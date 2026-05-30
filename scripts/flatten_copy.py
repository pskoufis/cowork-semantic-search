"""Recursively copy every regular file from a source tree into a flat target.

Usage:
    python scripts/flatten_copy.py <source-folder> <target-folder>

Walks ``<source-folder>`` recursively and copies every regular file
(symlinks skipped) into ``<target-folder>`` with no subdirectory
structure. Name collisions are resolved by appending ``_1``, ``_2``, ...
to the stem (e.g. ``foo.pdf`` → ``foo_1.pdf``); the first occurrence
keeps the plain name.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Allow direct invocation (``python scripts/flatten_copy.py``) by putting the
# repo root on sys.path before importing the sibling run-log helper.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._runlog import RunLog  # noqa: E402
from scripts import _runlog  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src = Path(args.source_folder).expanduser().resolve()
    dst = Path(args.target_folder).expanduser().resolve()

    if not src.exists():
        print(f"error: source folder not found: {src}", file=sys.stderr)
        return 1
    if not src.is_dir():
        print(f"error: source path is not a directory: {src}", file=sys.stderr)
        return 1
    if _is_inside(dst, src):
        print(
            f"error: target folder {dst} is inside source folder {src}; "
            "pick a target outside the source tree.",
            file=sys.stderr,
        )
        return 1

    dst.mkdir(parents=True, exist_ok=True)

    files = list(_iter_files(src))
    print(
        f"Found {len(files)} regular file(s) under {src}",
        file=sys.stderr,
    )
    rl = RunLog(
        "flatten_copy",
        input_root=src,
        output_root=dst,
        argv=argv if argv is not None else sys.argv[1:],
        log_dir=args.log_dir,
        force=args.force,
        enabled=not args.no_runlog,
    )
    used_names: set[str] = set()
    copied = 0
    skipped = 0
    failed = 0
    started = time.monotonic()
    exit_code = 0

    total = len(files)
    for i, src_file in enumerate(files, start=1):
        rel = src_file.relative_to(src)
        # Idempotent re-run: skip an input already copied (output present,
        # input unchanged). This is what stops a second run from re-copying
        # the whole tree under `_1`/`_2` names.
        if (done := rl.done_output(src_file)) is not None:
            skipped += 1
            print(f"[{i}/{total}] skip (done) {rel}", file=sys.stderr)
            rl.record(src_file, kind="file", output=done, status="skip")
            continue
        dst_name = _pick_dst_name(src_file.name, dst, used_names)
        used_names.add(dst_name)
        print(f"[{i}/{total}] copying {rel} -> {dst_name}", file=sys.stderr)
        item_started = time.monotonic()
        try:
            shutil.copy2(src_file, dst / dst_name)
        except Exception as exc:  # noqa: BLE001 — per-file isolation by design
            failed += 1
            # Drop a half-written file so a re-run starts clean.
            try:
                (dst / dst_name).unlink(missing_ok=True)
            except OSError:
                pass
            print(
                f"error: failed to copy {rel}: {exc}",
                file=sys.stderr,
            )
            rl.record(src_file, kind="file", output=None, status="fail",
                      error=exc)
            continue
        copied += 1
        rl.record(src_file, kind="file", output=dst / dst_name, status="ok",
                  duration_s=round(time.monotonic() - item_started, 3))

    elapsed = time.monotonic() - started
    skipped_part = f" {skipped} skipped," if skipped else ""
    print(
        f"Copied {copied} file(s) into {dst} ({failed} failed)"
        f"{skipped_part} in {elapsed:.1f}s.",
        file=sys.stderr,
    )
    exit_code = 0 if failed == 0 else 1
    rl.finish(exit_code=exit_code)
    return exit_code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively copy every file from <source-folder> into "
            "<target-folder> as a flat tree."
        )
    )
    parser.add_argument("source_folder", help="Folder to copy files from")
    parser.add_argument("target_folder", help="Folder to copy files into (flat)")
    _runlog.add_args(parser)
    return parser.parse_args(argv)


def _iter_files(root: Path):
    """Yield regular files under ``root`` (recursive). Symlinks are skipped
    to avoid loops and dangling targets. Sort by relative path for
    deterministic collision order."""
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            found.append(path)
    found.sort(key=lambda p: str(p.relative_to(root)))
    yield from found


def _pick_dst_name(name: str, dst: Path, used: set[str]) -> str:
    """Return a destination filename that is unused both on disk and in
    ``used``. On collision, append ``_1``, ``_2``, ... to the stem."""
    if name not in used and not (dst / name).exists():
        return name
    stem, dot, suffix = name.partition(".")
    # `partition` keeps any extension (including multi-dot like ".tar.gz")
    # in `suffix` with no leading dot stripped — we restore the dot below.
    suffix = f".{suffix}" if dot else ""
    for n in range(1, 1_000_000):
        candidate = f"{stem}_{n}{suffix}"
        if candidate not in used and not (dst / candidate).exists():
            return candidate
    raise RuntimeError(f"could not find an unused destination name for {name!r}")


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
