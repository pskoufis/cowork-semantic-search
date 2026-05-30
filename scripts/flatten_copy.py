"""Recursively copy every regular file from a source tree into a flat target.

Usage:
    python scripts/flatten_copy.py <source-folder> <target-folder>
        [--exclude-ext .ics] [--exclude-ext .tmp,.bak]

Walks ``<source-folder>`` recursively and copies every regular file
(symlinks skipped) into ``<target-folder>`` with no subdirectory
structure. Name collisions are resolved by appending ``_1``, ``_2``, ...
to the stem (e.g. ``foo.pdf`` → ``foo_1.pdf``); the first occurrence
keeps the plain name.

``--exclude-ext`` skips files by extension (case-insensitive, matched on
the last suffix). It is repeatable and accepts comma-separated lists; a
leading dot is optional (``ics`` and ``.ics`` are equivalent). Excluded
files are filtered out before copying, so they are neither counted nor
recorded in the run-log.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import unicodedata
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

    exclude_exts = _normalize_exts(args.exclude_ext)
    if exclude_exts:
        print(
            f"Excluding extensions: {', '.join(sorted(exclude_exts))}",
            file=sys.stderr,
        )

    files = list(_iter_files(src, exclude_exts))
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
        # Reuse the prior destination when reprocessing a known input (--force,
        # or an edited source) so we overwrite in place instead of minting an
        # `a_1.txt` duplicate. New inputs (ledger empty for them) still get
        # collision-suffixed against existing names.
        dst_name: str | None = None
        item_started = time.monotonic()
        try:
            # Name picking is inside the try so a single pathological input
            # (e.g. a name the filesystem rejects) fails that one item instead
            # of aborting the whole run.
            planned = rl.planned_output(src_file)
            dst_name = (
                planned.name if planned is not None
                else _pick_dst_name(src_file.name, dst, used_names)
            )
            used_names.add(dst_name)
            print(f"[{i}/{total}] copying {rel} -> {dst_name}", file=sys.stderr)
            shutil.copy2(src_file, dst / dst_name)
        except Exception as exc:  # noqa: BLE001 — per-file isolation by design
            failed += 1
            # Drop a half-written file so a re-run starts clean.
            if dst_name is not None:
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
    parser.add_argument(
        "--exclude-ext",
        action="append",
        default=None,
        metavar="EXT[,EXT...]",
        help=(
            "Extension(s) to skip, case-insensitive (e.g. .ics). Repeatable "
            "and comma-separated; a leading dot is optional."
        ),
    )
    _runlog.add_args(parser)
    return parser.parse_args(argv)


def _normalize_exts(values: list[str] | None) -> set[str]:
    """Turn ``--exclude-ext`` values into a set of lowercase, dot-prefixed
    extensions. Handles repeated flags and comma-separated lists; ``ics`` and
    ``.ics`` are equivalent; blanks are dropped."""
    exts: set[str] = set()
    for value in values or []:
        for part in value.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if not part.startswith("."):
                part = "." + part
            exts.add(part)
    return exts


def _iter_files(root: Path, exclude_exts: set[str] | None = None):
    """Yield regular files under ``root`` (recursive). Symlinks are skipped
    to avoid loops and dangling targets. Files whose last suffix is in
    ``exclude_exts`` (case-insensitive) are omitted. Sort by relative path for
    deterministic collision order."""
    exclude_exts = exclude_exts or set()
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            if path.suffix.lower() in exclude_exts:
                continue
            found.append(path)
    found.sort(key=lambda p: str(p.relative_to(root)))
    yield from found


# A single filename component is byte-limited by the filesystem (255 on
# macOS/APFS and ext4). macOS stores names NFD-normalised, so we measure the
# NFD/UTF-8 byte length and keep headroom below the limit for that expansion
# plus a collision suffix, so the assembled name always fits.
_FALLBACK_NAME_MAX = 255
_NFD_HEADROOM = 8
_COLLISION_RESERVE = len("_999999")
# Cap a "suffix" that is really path junk (e.g. a trailing "...") so it can't
# eat the whole budget; a real extension is short.
_MAX_SUFFIX_BYTES = 16


def _name_byte_limit(dst: Path) -> int:
    try:
        limit = os.pathconf(dst, "PC_NAME_MAX")
    except (OSError, ValueError, AttributeError):
        limit = _FALLBACK_NAME_MAX
    return limit if limit and limit > 0 else _FALLBACK_NAME_MAX


def _nfd_len(s: str) -> int:
    return len(unicodedata.normalize("NFD", s).encode("utf-8"))


def _truncate_to_bytes(s: str, limit: int) -> str:
    """Longest prefix of ``s`` whose NFD/UTF-8 byte length is <= ``limit``."""
    if _nfd_len(s) <= limit:
        return s
    out = s
    while out and _nfd_len(out) > limit:
        out = out[:-1]
    return out


def _pick_dst_name(name: str, dst: Path, used: set[str]) -> str:
    """Return a destination filename that is unused both on disk and in
    ``used`` and that fits the filesystem's per-name byte limit. Over-long
    names are truncated (keeping the real extension); on collision, append
    ``_1``, ``_2``, ... to the stem."""
    limit = _name_byte_limit(dst)
    # Split the real (last) extension off the original name once, discarding an
    # absurdly long "extension" that is really path junk.
    suffix = Path(name).suffix
    if _nfd_len(suffix) > _MAX_SUFFIX_BYTES:
        suffix = ""
    stem = name[: len(name) - len(suffix)]
    stem_budget = limit - _nfd_len(suffix) - _COLLISION_RESERVE - _NFD_HEADROOM
    if stem_budget < 1:
        suffix = ""
        stem_budget = max(1, limit - _COLLISION_RESERVE - _NFD_HEADROOM)
        stem = name
    stem = _truncate_to_bytes(stem, stem_budget)
    base = f"{stem}{suffix}"
    if base not in used and not (dst / base).exists():
        return base
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
