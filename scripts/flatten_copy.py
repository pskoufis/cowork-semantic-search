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
    used_names: set[str] = set()
    copied = 0
    failed = 0
    started = time.monotonic()

    total = len(files)
    for i, src_file in enumerate(files, start=1):
        rel = src_file.relative_to(src)
        dst_name = _pick_dst_name(src_file.name, dst, used_names)
        used_names.add(dst_name)
        print(f"[{i}/{total}] copying {rel} -> {dst_name}", file=sys.stderr)
        try:
            shutil.copy2(src_file, dst / dst_name)
        except Exception as exc:  # noqa: BLE001 — per-file isolation by design
            failed += 1
            print(
                f"error: failed to copy {rel}: {exc}",
                file=sys.stderr,
            )
            continue
        copied += 1

    elapsed = time.monotonic() - started
    print(
        f"Copied {copied} file(s) into {dst} ({failed} failed) in {elapsed:.1f}s.",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively copy every file from <source-folder> into "
            "<target-folder> as a flat tree."
        )
    )
    parser.add_argument("source_folder", help="Folder to copy files from")
    parser.add_argument("target_folder", help="Folder to copy files into (flat)")
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
