"""Batch-unpack every ``.mbox`` file under a folder into a chosen output tree.

Usage:
    python scripts/unpack_mbox_folder.py <input-folder> <output-folder>

For each ``.mbox`` file found (recursively, case-insensitive), the script
creates ``<output>/<mbox-stem>/`` and delegates the actual unpacking to
``mbox_handling.unpack.ensure_unpacked``. The output tree per mbox is the
standard one: ``thread-NNNN_slug/`` folders containing message ``.txt``
files and a per-message ``attachments/`` subfolder, plus ``_unthreaded/``
for orphans.

Same-stem collisions across subfolders are disambiguated with ``-1``,
``-2``, ... suffixes. Per-mbox errors are logged to stderr and the batch
continues; the exit code is non-zero if at least one mbox failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow direct invocation (``python scripts/unpack_mbox_folder.py``) in
# addition to ``python -m scripts.unpack_mbox_folder`` by putting the
# repo root on sys.path before importing project code.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mbox_handling.unpack import ensure_unpacked  # noqa: E402
from scripts._runlog import RunLog  # noqa: E402
from scripts import _runlog  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src = Path(args.input_folder).expanduser().resolve()
    dst = Path(args.output_folder).expanduser().resolve()

    if not src.exists():
        print(f"error: input folder not found: {src}", file=sys.stderr)
        return 1
    if not src.is_dir():
        print(f"error: input path is not a directory: {src}", file=sys.stderr)
        return 1
    if _is_inside(dst, src):
        print(
            f"error: output folder {dst} is inside input folder {src}; "
            "pick a target outside the input tree.",
            file=sys.stderr,
        )
        return 1

    dst.mkdir(parents=True, exist_ok=True)

    mboxes = _discover_mboxes(src)
    print(
        f"Found {len(mboxes)} mbox file(s) under {src}",
        file=sys.stderr,
    )
    rl = RunLog(
        "unpack_mbox_folder",
        input_root=src,
        output_root=dst,
        argv=argv if argv is not None else sys.argv[1:],
        log_dir=args.log_dir,
        force=args.force,
        enabled=not args.no_runlog,
    )
    succeeded = 0
    skipped = 0
    failed = 0
    used_names: set[str] = set()
    started = time.monotonic()

    total = len(mboxes)
    for i, mbox in enumerate(mboxes, start=1):
        rel = mbox.relative_to(src)
        # Idempotent re-run: skip an mbox already unpacked into a still-present
        # output dir. Reserve that dir's name so a different mbox can't pick it.
        if (done := rl.done_output(mbox)) is not None:
            skipped += 1
            used_names.add(done.name)
            print(f"[{i}/{total}] skip (done) {rel}", file=sys.stderr)
            rl.record(mbox, kind="mbox", output=done, status="skip")
            continue
        # Reuse the prior target when reprocessing (force / changed) so we
        # don't mint a fresh `<stem>-N/` next to the old one (bug #3).
        target = rl.planned_output(mbox) or _pick_target(dst, mbox.stem, used_names)
        used_names.add(target.name)
        print(
            f"[{i}/{total}] unpacking {rel} -> {target.name}",
            file=sys.stderr,
        )
        item_started = time.monotonic()
        try:
            ensure_unpacked(mbox, target=target)
        except Exception as exc:  # noqa: BLE001 — per-file isolation by design
            failed += 1
            print(
                f"error: failed to unpack {mbox}: {exc}",
                file=sys.stderr,
            )
            # Record the intended target so a retry reuses it instead of
            # allocating a fresh `-N` dir.
            rl.record(mbox, kind="mbox", output=target, status="fail", error=exc)
            continue
        succeeded += 1
        rl.record(mbox, kind="mbox", output=target, status="ok",
                  duration_s=round(time.monotonic() - item_started, 3))

    elapsed = time.monotonic() - started
    skipped_part = f", {skipped} skipped" if skipped else ""
    print(
        f"Processed {total} mbox file(s) in {elapsed:.1f}s: "
        f"{succeeded} succeeded, {failed} failed{skipped_part}.",
        file=sys.stderr,
    )
    exit_code = 0 if failed == 0 else 1
    rl.finish(exit_code=exit_code)
    return exit_code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-unpack every .mbox under <input-folder> into "
            "<output-folder>/<mbox-stem>/ subdirectories."
        )
    )
    parser.add_argument("input_folder", help="Folder to scan for .mbox files")
    parser.add_argument(
        "output_folder",
        help="Folder to write per-mbox unpacked trees into",
    )
    _runlog.add_args(parser)
    return parser.parse_args(argv)


def _discover_mboxes(root: Path) -> list[Path]:
    """Return every regular file under ``root`` with a ``.mbox`` extension
    (case-insensitive), sorted deterministically by relative path."""
    results: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".mbox":
            results.append(path)
    results.sort(key=lambda p: str(p.relative_to(root)))
    return results


def _pick_target(out_root: Path, stem: str, used: set[str]) -> Path:
    """Pick ``<out_root>/<stem>``, falling back to ``<stem>-1/-2/...`` when
    already used in this run or already non-empty on disk."""
    candidate_name = stem
    if candidate_name not in used and not _is_in_use(out_root / candidate_name):
        return out_root / candidate_name
    for n in range(1, 100_000):
        candidate_name = f"{stem}-{n}"
        candidate = out_root / candidate_name
        if candidate_name not in used and not _is_in_use(candidate):
            return candidate
    raise RuntimeError(
        f"could not find an unused output subdir for stem {stem!r} under {out_root}"
    )


def _is_in_use(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
