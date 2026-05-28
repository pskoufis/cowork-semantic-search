"""Shared pst/mbox/msg unpack orchestration.

This used to live inline inside :func:`server.indexer.index_folder` as three
near-identical ``_ensure_*_unpacked`` helpers. It was extracted so the
``csemsearch`` CLI's ``unpack`` verb can call the same code path with
progress reporting, while the indexer continues to invoke it (controlled
by ``index_folder``'s ``unpack_first`` parameter).

Phase order is load-bearing:

* ``pst`` runs first because a .pst extract can yield .mbox or .msg files
  in its attachments tree, and we want those caught by the later passes.
* ``mbox`` runs second because mbox messages can carry .msg attachments.
* ``msg`` runs last.

Each per-archive unpacker (``ensure_pst_unpacked`` etc.) is mtime-guarded
so re-runs are cheap — the discovery walk still visits every archive but
the underlying extraction is a no-op when the sibling ``<stem>_unpacked``
tree is at least as fresh as the source archive.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

from mbox_handling.unpack import ensure_unpacked as ensure_mbox_unpacked
from msg_handling.unpack import ensure_unpacked as ensure_msg_unpacked
from pst_handling.unpack import ensure_unpacked as ensure_pst_unpacked
from server.exclusions import ExclusionRules


logger = logging.getLogger("cowork_semantic_search.unpacker")

PHASE_PST = "pst"
PHASE_MBOX = "mbox"
PHASE_MSG = "msg"
PHASES: tuple[str, ...] = (PHASE_PST, PHASE_MBOX, PHASE_MSG)


ProgressCallback = Callable[[str, int, int], None]


def run_unpack_passes(
    folder: Path,
    *,
    recursive: bool = True,
    exclusions: ExclusionRules | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Run the pst → mbox → msg unpack passes over ``folder``.

    Returns a dict keyed by phase name with per-phase counts:

        {
            "pst":  {"discovered": int, "failed": int, "elapsed_s": float},
            "mbox": {...},
            "msg":  {...},
        }

    The callback (if given) fires for every archive started in each phase
    and once more at the end of the phase with ``processed == total`` —
    including phases that discovered zero archives, so a CLI progress bar
    can always close out cleanly.
    """
    folder = Path(folder)
    out: dict[str, dict] = {}
    for phase in PHASES:
        out[phase] = _run_one_pass(
            folder,
            phase=phase,
            recursive=recursive,
            exclusions=exclusions,
            progress_callback=progress_callback,
        )
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _phase_unpacker(phase: str) -> Callable[[Path], object]:
    """Resolve a phase name to its per-archive unpacker. Indirection through
    module attributes keeps tests free to monkeypatch a specific unpacker
    (e.g. to simulate one .mbox failing) without losing the orchestration."""
    if phase == PHASE_PST:
        return ensure_pst_unpacked
    if phase == PHASE_MBOX:
        return ensure_mbox_unpacked
    if phase == PHASE_MSG:
        return ensure_msg_unpacked
    raise ValueError(f"unknown phase: {phase!r}")


def _phase_extension(phase: str) -> str:
    if phase == PHASE_PST:
        return ".pst"
    if phase == PHASE_MBOX:
        return ".mbox"
    if phase == PHASE_MSG:
        return ".msg"
    raise ValueError(f"unknown phase: {phase!r}")


def _run_one_pass(
    folder: Path,
    *,
    phase: str,
    recursive: bool,
    exclusions: ExclusionRules | None,
    progress_callback: ProgressCallback | None,
) -> dict:
    ext = _phase_extension(phase)
    archives = list(
        _discover_archives(folder, ext=ext, recursive=recursive, exclusions=exclusions)
    )
    started = time.time()
    total = len(archives)
    failed = 0
    for idx, archive in enumerate(archives):
        if progress_callback is not None:
            progress_callback(phase, idx, total)
        try:
            # Resolve at call time so tests that monkeypatch
            # ``server.unpacker.ensure_mbox_unpacked`` (etc.) see the patched
            # version — module-level lookup, not closure capture.
            _phase_unpacker(phase)(archive)
        except Exception as e:
            failed += 1
            logger.warning("failed to unpack %s: %s", archive, e)
            print(
                f"warning: failed to unpack {archive}: {e}",
                file=sys.stderr,
            )
    if progress_callback is not None:
        progress_callback(phase, total, total)
    return {
        "discovered": total,
        "failed": failed,
        "elapsed_s": time.time() - started,
    }


def _discover_archives(
    folder: Path,
    *,
    ext: str,
    recursive: bool,
    exclusions: ExclusionRules | None,
) -> Iterator[Path]:
    """Yield archive paths matching ``ext`` (case-insensitive) under
    ``folder``. Honours the same exclusion + recursive rules the inline
    ``_ensure_*_unpacked`` helpers used to use."""
    ext_lower = ext.lower()
    for dirpath, dirnames, filenames in os.walk(folder):
        dpath = Path(dirpath)
        if recursive:
            if exclusions is not None:
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not exclusions.is_excluded(dpath / d, folder, is_dir=True)
                ]
        else:
            dirnames[:] = []
        for name in filenames:
            if not name.lower().endswith(ext_lower):
                continue
            path = dpath / name
            if exclusions is not None and exclusions.is_excluded(
                path, folder, is_dir=False
            ):
                continue
            yield path
