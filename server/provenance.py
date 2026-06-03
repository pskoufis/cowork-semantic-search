"""Reverse-lookup: map an unpacked file back to its source archive.

The unpack scripts turn ``.pst`` / ``.mbox`` / ``.msg`` archives into trees of
per-message ``.txt`` files + materialized attachments. The output path is a
deterministic function of the source path, so the mapping is reversible. This
module performs that reverse lookup, returning a full :class:`SourceTrace`.

Two provenance sources, tried in order:

1. **The runlog ledger** (written by ``scripts/_runlog.py`` under
   ``<target_root>/_runlogs/*.jsonl``). Each ``item`` line records ``input_abs``
   (the absolute source archive), ``output`` (the unpacked tree/file), and
   ``kind``. This is authoritative — it records the true source even when the
   path convention can't reverse it (the mbox batch flattens subdirs and adds
   ``-1``/``-2`` collision suffixes that path-math alone can't undo).

2. **The path convention** as a fallback when no ledger covers the file
   (e.g. files unpacked in-place by the indexer, or a tree built without
   runlogs). PST and MSG are reconstructable; mbox is best-effort and refuses
   rather than guess when ambiguous.

The recorded ``output`` differs by format: PST → the ``<stem>_unpacked`` dir,
mbox → the ``<stem>``/``<stem>-N`` dir, MSG → the ``<stem>.txt`` file itself
(so MSG attachments, which share one ``attachments/`` dir across sibling
``.msg`` files, are matched by deriving the owning ``.txt`` from the
``<stem>__`` filename prefix).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_MSG_NUM_RE = re.compile(r"msg-(\d+)")
_COLLISION_SUFFIX_RE = re.compile(r"-\d+$")
_RUNLOG_SUBDIR = "_runlogs"


@dataclass(frozen=True)
class SourceTrace:
    """The resolved provenance of an unpacked file."""

    source_archive: Path
    """Absolute path to the originating ``.pst`` / ``.mbox`` / ``.msg``."""
    archive_type: str
    """``"pst"`` | ``"mbox"`` | ``"msg"``."""
    archive_exists: bool
    """Whether ``source_archive`` is currently present on disk."""
    method: str
    """How it was resolved: ``"runlog"`` | ``"path"``."""
    internal_folder: str | None
    """For PST, the ``Folder:`` header value; otherwise ``None``."""
    message_number: int | None
    """The ``NNNN`` from a ``msg-NNNN`` filename, when present."""
    attachment_of: Path | None
    """For an attachment, the owning message ``.txt``; else ``None``."""
    runlog_path: Path | None
    """Which ledger matched, when ``method == "runlog"``; else ``None``."""


def trace_source(
    file_path: Path | str,
    source_root: Path | str,
    target_root: Path | str,
) -> SourceTrace:
    """Map ``file_path`` (an unpacked ``.txt`` or extracted attachment under
    ``target_root``) back to its source archive under ``source_root``.

    Raises ``FileNotFoundError`` if ``file_path`` does not exist, ``ValueError``
    if it is not under ``target_root``, and ``LookupError`` if neither the
    runlog nor the path convention can resolve a source.
    """
    file_path = Path(file_path).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    target_root = Path(target_root).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    try:
        file_path.relative_to(target_root)
    except ValueError:
        raise ValueError(
            f"file {file_path} is not under target_root {target_root}"
        )

    match = _match_runlog(file_path, target_root)
    if match is not None:
        source_archive, kind, runlog_path = match
        internal_folder, message_number, attachment_of = _enrich(file_path)
        return SourceTrace(
            source_archive=source_archive,
            archive_type=kind,
            archive_exists=source_archive.exists(),
            method="runlog",
            internal_folder=internal_folder,
            message_number=message_number,
            attachment_of=attachment_of,
            runlog_path=runlog_path,
        )

    return _path_fallback(file_path, source_root, target_root)


# --- runlog matching -------------------------------------------------------


def _iter_runlog_items(target_root: Path) -> Iterator[tuple[Path, dict]]:
    """Yield ``(runlog_path, item_record)`` for every successful ``item`` line
    across all ledgers under ``<target_root>/_runlogs``. Malformed lines and
    failed items are skipped (the reader is deliberately tolerant, matching
    ``scripts/_runlog.py``'s own ledger loader)."""
    log_dir = target_root / _RUNLOG_SUBDIR
    if not log_dir.exists():
        return
    for f in sorted(log_dir.glob("*.jsonl")):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("event") == "item"
                and rec.get("output")
                and rec.get("status") != "fail"
                and rec.get("input_abs")
            ):
                yield f, rec


def _candidate_outputs(file_path: Path) -> set[Path]:
    """The set of recorded ``output`` values that could correspond to
    ``file_path``: the file itself, every ancestor directory (PST/mbox record a
    dir), and — for an MSG attachment — the owning ``.txt`` derived from the
    ``<stem>__`` prefix (MSG records the ``.txt``, not the attachment)."""
    candidates: set[Path] = {file_path, *file_path.parents}
    if file_path.parent.name == "attachments" and "__" in file_path.name:
        stem = file_path.name.split("__", 1)[0]
        candidates.add(file_path.parent.parent / f"{stem}.txt")
    return candidates


def _match_runlog(
    file_path: Path, target_root: Path
) -> tuple[Path, str, Path] | None:
    """Return ``(source_archive, kind, runlog_path)`` for the most specific
    ledger item whose ``output`` matches ``file_path``, or ``None``."""
    candidates = {p.resolve() for p in _candidate_outputs(file_path)}
    best: tuple[int, Path, str, Path] | None = None
    for runlog_path, rec in _iter_runlog_items(target_root):
        out = Path(rec["output"]).resolve()
        if out not in candidates:
            continue
        specificity = len(str(out))
        if best is None or specificity > best[0]:
            best = (
                specificity,
                Path(rec["input_abs"]).resolve(),
                rec.get("kind", ""),
                runlog_path,
            )
    if best is None:
        return None
    return best[1], best[2], best[3]


# --- enrichment ------------------------------------------------------------


def _owner_txt(file_path: Path) -> Path | None:
    """Locate the message ``.txt`` that owns ``file_path``.

    A body ``.txt`` owns itself. A PST/mbox attachment lives in
    ``attachments/msg-NNNN/`` next to a ``msg-NNNN_*.txt`` sibling. An MSG
    attachment lives flat in ``attachments/`` with a ``<stem>__`` prefix whose
    owner is ``<stem>.txt`` one level up.
    """
    if file_path.suffix.lower() == ".txt" and file_path.parent.name != "attachments":
        return file_path

    if file_path.parent.name == "attachments" and "__" in file_path.name:
        stem = file_path.name.split("__", 1)[0]
        cand = file_path.parent.parent / f"{stem}.txt"
        return cand if cand.exists() else None

    parent = file_path.parent
    m = _MSG_NUM_RE.search(parent.name)
    if parent.parent.name == "attachments" and m:
        folder_dir = parent.parent.parent
        matches = sorted(folder_dir.glob(f"msg-{int(m.group(1)):04d}_*.txt"))
        if matches:
            return matches[0]
    return None


def _enrich(file_path: Path) -> tuple[str | None, int | None, Path | None]:
    """Derive ``(internal_folder, message_number, attachment_of)`` for
    ``file_path`` from the surrounding unpacked tree."""
    owner = _owner_txt(file_path)
    is_attachment = owner is not None and owner != file_path
    attachment_of = owner.resolve() if is_attachment else None

    message_number = _first_msg_number(
        owner.name if owner is not None else file_path.name,
        file_path.parent.name,
    )

    internal_folder = None
    if owner is not None and owner.exists():
        internal_folder = _read_folder_header(owner)

    return internal_folder, message_number, attachment_of


def _first_msg_number(*names: str) -> int | None:
    for name in names:
        m = _MSG_NUM_RE.search(name)
        if m:
            return int(m.group(1))
    return None


def _read_folder_header(txt_path: Path) -> str | None:
    """Return the ``Folder:`` header value from a PST-unpacked ``.txt`` (it
    appears in the header block before the first blank line); ``None`` for
    mbox/msg outputs, which carry no such line."""
    try:
        with txt_path.open(encoding="utf-8", errors="replace") as fh:
            for _ in range(40):
                line = fh.readline()
                if line == "" or line.strip() == "":
                    break
                if line.startswith("Folder:"):
                    return line[len("Folder:"):].strip() or None
    except OSError:
        return None
    return None


# --- path-convention fallback ----------------------------------------------


def _path_fallback(
    file_path: Path, source_root: Path, target_root: Path
) -> SourceTrace:
    rel_parts = file_path.relative_to(target_root).parts

    unpacked = _unpacked_from_path(file_path, rel_parts, source_root)
    if unpacked is not None:
        return unpacked

    msg = _msg_from_path(file_path, source_root, target_root)
    if msg is not None:
        return msg

    return _mbox_from_path(file_path, rel_parts, source_root)


def _unpacked_from_path(
    file_path: Path, rel_parts: tuple[str, ...], source_root: Path
) -> SourceTrace | None:
    """Resolve a file living under a ``<stem>_unpacked`` directory.

    Both the PST batch script and the per-archive (in-place) unpack of *either*
    a ``.pst`` or a ``.mbox`` write ``<stem>_unpacked``, so the suffix alone
    doesn't tell the two formats apart. Probe the source next to the tree:
    prefer whichever of ``<stem>.pst`` / ``<stem>.mbox`` exists, defaulting to
    ``pst`` (the batch convention) when neither is present.
    """
    for i, part in enumerate(rel_parts):
        if not part.endswith("_unpacked"):
            continue
        stem = part[: -len("_unpacked")]
        rel_parent = Path(*rel_parts[:i])
        base = source_root / rel_parent
        pst = (base / f"{stem}.pst").resolve()
        mbox = (base / f"{stem}.mbox").resolve()
        if not pst.exists() and mbox.exists():
            source, kind, exists = mbox, "mbox", True
        else:
            source, kind, exists = pst, "pst", pst.exists()
        folder, num, att = _enrich(file_path)
        return SourceTrace(
            source_archive=source,
            archive_type=kind,
            archive_exists=exists,
            method="path",
            internal_folder=folder,
            message_number=num,
            attachment_of=att,
            runlog_path=None,
        )
    return None


def _msg_from_path(
    file_path: Path, source_root: Path, target_root: Path
) -> SourceTrace | None:
    """MSG preserves ``<rel>`` and names outputs ``<stem>.txt`` +
    ``attachments/<stem>__*``. Confirmed by probing that ``<stem>.msg`` exists
    in the source — that probe also disambiguates MSG from mbox outputs."""
    if file_path.parent.name == "attachments" and "__" in file_path.name:
        stem = file_path.name.split("__", 1)[0]
        owner_dir = file_path.parent.parent
    elif file_path.suffix.lower() == ".txt":
        stem = file_path.stem
        owner_dir = file_path.parent
    else:
        return None

    rel_dir = owner_dir.relative_to(target_root)
    candidate = (source_root / rel_dir / f"{stem}.msg").resolve()
    if not candidate.exists():
        return None

    folder, num, att = _enrich(file_path)
    return SourceTrace(
        source_archive=candidate,
        archive_type="msg",
        archive_exists=True,
        method="path",
        internal_folder=folder,
        message_number=num,
        attachment_of=att,
        runlog_path=None,
    )


def _mbox_from_path(
    file_path: Path, rel_parts: tuple[str, ...], source_root: Path
) -> SourceTrace:
    """mbox outputs are flattened to ``<target>/<stem>/...`` with ``-N``
    collision suffixes, so ``<rel>`` is lost. Best-effort: take the first path
    segment as the stem and search the source for a unique ``<stem>.mbox``.
    Refuse (``LookupError``) rather than guess when ambiguous — runlogs are the
    supported path for mbox."""
    stem = rel_parts[0] if rel_parts else ""
    source = _unique_mbox(source_root, stem)
    if source is None:
        raise LookupError(
            f"cannot resolve an mbox source for {file_path} by path alone "
            f"(stem {stem!r} is missing or ambiguous in {source_root}); a "
            "runlog ledger is required to trace flattened mbox outputs."
        )

    folder, num, att = _enrich(file_path)
    return SourceTrace(
        source_archive=source.resolve(),
        archive_type="mbox",
        archive_exists=True,
        method="path",
        internal_folder=folder,
        message_number=num,
        attachment_of=att,
        runlog_path=None,
    )


def _unique_mbox(source_root: Path, stem: str) -> Path | None:
    """Return the sole ``<stem>.mbox`` under ``source_root``, trying the stem
    as-is then with a trailing ``-N`` collision suffix stripped. ``None`` if
    absent or non-unique."""
    for candidate_stem in _dedupe([stem, _COLLISION_SUFFIX_RE.sub("", stem)]):
        matches = list(source_root.rglob(f"{candidate_stem}.mbox"))
        if len(matches) == 1:
            return matches[0]
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
