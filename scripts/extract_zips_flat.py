"""Recursively flat-extract every ``.zip`` under a folder.

Usage:
    python scripts/extract_zips_flat.py <root-folder> [--dry-run]

For each ``foo.zip`` found (recursively, case-insensitive):

* Every file inside the zip is written to ``foo.zip``'s parent
  directory using **only its basename** — subdirectory structure inside
  the zip is dropped. Files with the same basename (either coming from
  different paths inside the zip, or already present in the parent dir)
  get ``-1``, ``-2``, ... inserted before the extension to avoid
  overwriting anything.
* Directory entries inside the zip are ignored.
* Taking only the basename also acts as the path-traversal defense:
  archive members can't escape the parent dir because their path
  components are discarded.

Idempotency:
    On success, a hidden sibling marker ``.<zipname>.flat_extracted``
    is written next to the zip. Re-runs skip zips whose marker already
    exists. To force a re-extraction, delete the marker manually.

Audit log:
    Each run writes ``<root>/extract_zips_flat-<UTC-timestamp>.log``
    listing every file created during the run (absolute paths), one
    per line, with comments noting the source zip. Use this to review
    or batch-delete extracted files later, e.g.::

        grep -v '^#' <log> | xargs rm

Per-zip errors are logged to stderr and the batch continues. Exit code
is non-zero if at least one zip failed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
import zipfile
from pathlib import Path

MARKER_SUFFIX = ".flat_extracted"  # full marker name: .<zipname>.flat_extracted
LOG_PREFIX = "extract_zips_flat-"
LOG_SUFFIX = ".log"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root_folder).expanduser().resolve()

    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    zips = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip")
    if not zips:
        print(f"No .zip files found under {root}")
        return 0

    print(f"Found {len(zips)} zip file(s) under {root}")

    extracted = 0
    skipped = 0
    errors = 0
    total_files = 0
    total_renamed = 0
    # log_entries: list of (absolute_output_path, source_zip_rel_to_root)
    log_entries: list[tuple[Path, Path]] = []

    for zip_path in zips:
        rel = zip_path.relative_to(root)
        marker = _marker_path(zip_path)

        if marker.exists():
            print(f"SKIP {rel} (already flat-extracted)")
            skipped += 1
            continue

        try:
            written, renamed = _flat_extract(zip_path, dry_run=args.dry_run)
        except (zipfile.BadZipFile, RuntimeError, OSError, ValueError) as exc:
            print(f"ERROR {rel}: {exc}", file=sys.stderr)
            errors += 1
            continue

        if not args.dry_run:
            try:
                marker.write_text(zip_path.name, encoding="utf-8")
            except OSError as exc:
                print(f"ERROR {rel}: extraction OK but marker write failed: {exc}", file=sys.stderr)
                errors += 1
                # don't continue — count the extraction below

        renamed_note = f" ({renamed} renamed for collision)" if renamed else ""
        verb = "DRY-RUN" if args.dry_run else "OK"
        print(f"{verb} {rel} -> {len(written)} file(s){renamed_note}")
        extracted += 1
        total_files += len(written)
        total_renamed += renamed
        log_entries.extend((p, rel) for p in written)

    log_path: Path | None = None
    if not args.dry_run and log_entries:
        log_path = _write_log(root, log_entries)

    parts = [f"{extracted} extracted"]
    if skipped:
        parts.append(f"{skipped} skipped")
    parts.append(f"{total_files} files written")
    if total_renamed:
        parts.append(f"{total_renamed} renamed for collision")
    parts.append(f"{errors} errors")
    print(f"\nSummary: {', '.join(parts)}")
    if log_path is not None:
        print(f"Log: {log_path}")
    return 1 if errors else 0


def _flat_extract(zip_path: Path, dry_run: bool) -> tuple[list[Path], int]:
    """Extract zip's files flat into its parent dir.

    Returns (list of absolute paths written, count of collision renames).
    During dry_run, no files are written but the returned paths reflect
    what WOULD be written, with collision detection accounting for both
    pre-existing files and earlier entries of this same zip.
    """
    parent = zip_path.parent
    written: list[Path] = []
    # used: track basenames we've consumed during this zip's extraction,
    # so a later entry doesn't pick the same name as an earlier one even
    # before the file is on disk (matters during dry-run; matters during
    # real run too since we want consistent counting).
    used: set[str] = set()
    renames = 0

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            raw_name = Path(member.filename).name
            if not raw_name:
                continue

            dest = _pick_dest(parent, raw_name, used)
            if dest.name != raw_name:
                renames += 1
            used.add(dest.name)

            if not dry_run:
                # Stream the member to avoid loading huge files into memory.
                with zf.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)

            written.append(dest.resolve())

    return written, renames


def _pick_dest(parent: Path, basename: str, used: set[str]) -> Path:
    """Return a non-colliding destination path in ``parent`` for ``basename``.

    Inserts ``-1``, ``-2``, ... before the file extension on collision.
    Considers both filesystem presence and previously-chosen names in
    ``used``.
    """
    stem, ext = _split_name(basename)
    candidate_name = basename
    i = 0
    while candidate_name in used or (parent / candidate_name).exists():
        i += 1
        candidate_name = f"{stem}-{i}{ext}"
    return parent / candidate_name


def _split_name(basename: str) -> tuple[str, str]:
    """Split a basename into (stem, ext) where ext includes the leading dot.

    Dot-files (``.bashrc``) are treated as having no extension so the
    suffix lands at the end (``.bashrc-1``), not in the middle.
    """
    if basename.startswith(".") and basename.count(".") == 1:
        return basename, ""
    p = Path(basename)
    if not p.suffix:
        return basename, ""
    return p.stem, p.suffix


def _marker_path(zip_path: Path) -> Path:
    """Return the hidden sibling marker path for ``zip_path``."""
    return zip_path.with_name(f".{zip_path.name}{MARKER_SUFFIX}")


def _write_log(root: Path, entries: list[tuple[Path, Path]]) -> Path:
    """Write the audit log for this run and return its path."""
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = root / f"{LOG_PREFIX}{timestamp}{LOG_SUFFIX}"
    lines = [
        f"# extract_zips_flat run at {timestamp}",
        f"# root: {root}",
        f"# {len(entries)} file(s) written. Source zip noted on each line.",
        "# To delete every file listed here:",
        "#     grep -v '^#' <this-file> | cut -f1 | xargs rm",
        "",
    ]
    for output_path, source_zip_rel in entries:
        # TSV: <absolute output path>\t<source zip relative to root>
        lines.append(f"{output_path}\t{source_zip_rel}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively flat-extract every .zip under a folder. Files "
            "are written into the zip's parent dir using only their "
            "basename; collisions get -1, -2, ... before the extension. "
            "Re-runs are skipped via a hidden sibling marker. A "
            "consolidated audit log is written at "
            "<root>/extract_zips_flat-<timestamp>.log so you can review "
            "or delete extracted files later."
        ),
    )
    parser.add_argument("root_folder", help="Folder to scan recursively for .zip files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be extracted without writing anything (no log either).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
