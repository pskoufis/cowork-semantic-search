"""Recursively extract every ``.zip`` under a folder into a sibling dir.

Usage:
    python scripts/extract_zips.py <root-folder> [--dry-run] [--overwrite]

For each ``foo.zip`` found (recursively, case-insensitive), extracts into
``foo_unzip/`` next to the zip (same parent). The ``_unzip`` suffix makes
it obvious the folder came from this script.

Idempotent & non-destructive:
    Each extracted folder is marked with a hidden file
    ``.extract_zips_source`` containing the source zip's filename. On
    re-run, the script uses this marker to decide:

    * Marker matches the source zip -> already extracted, SKIP it
      (no duplicate ``-N`` copy is created). With ``--overwrite``, the
      folder is removed and re-extracted instead.
    * Folder exists with no marker (organic / user content), or marker
      for a *different* zip (e.g. ``foo.zip`` vs ``foo.ZIP`` on a
      case-sensitive filesystem) -> try ``foo_unzip-1/``, ``foo_unzip-2/``,
      ... until a free slot or a matching marker is found.

    The script never deletes an organic folder, even with ``--overwrite``.

Per-zip errors are logged to stderr and the batch continues. Exit code is
non-zero if at least one zip failed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

TARGET_SUFFIX = "_unzip"
MARKER_NAME = ".extract_zips_source"
MAX_SUFFIX = 999  # safety cap on -1, -2, ... search


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
    renamed = 0
    skipped = 0
    overwritten = 0
    errors = 0

    for zip_path in zips:
        source_name = zip_path.name  # e.g. "foo.zip" or "foo.ZIP"
        base_target = zip_path.with_name(zip_path.stem + TARGET_SUFFIX)
        rel = zip_path.relative_to(root)

        try:
            action, target = _resolve_target(base_target, source_name, args.overwrite)
        except RuntimeError as exc:
            print(f"ERROR {rel}: {exc}", file=sys.stderr)
            errors += 1
            continue

        if action == "skip":
            print(f"SKIP {rel} (already extracted at {target.name}/)")
            skipped += 1
            continue

        if action == "extract" and target != base_target:
            print(
                f"NOTE {rel}: {base_target.name}/ holds other content, "
                f"using {target.name}/ instead"
            )
            renamed += 1

        if args.dry_run:
            verb = "REPLACE" if action == "replace" else "EXTRACT"
            print(f"DRY-RUN {verb} {rel} -> {target.name}/")
            if action == "replace":
                overwritten += 1
            extracted += 1
            continue

        if action == "replace":
            try:
                shutil.rmtree(target)
            except OSError as exc:
                print(f"ERROR {rel}: cannot remove target: {exc}", file=sys.stderr)
                errors += 1
                continue
            overwritten += 1

        try:
            _extract_safely(zip_path, target)
            _write_marker(target, source_name)
        except (zipfile.BadZipFile, RuntimeError, OSError, ValueError) as exc:
            print(f"ERROR {rel}: {exc}", file=sys.stderr)
            # Only safe to clean up dirs we just created. action == "replace"
            # has already deleted the prior dir; "extract" creates a fresh
            # one. Either way the dir at `target` is ours to remove.
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            errors += 1
            continue

        print(f"OK {rel} -> {target.name}/")
        extracted += 1

    parts = [f"{extracted} extracted"]
    if renamed:
        parts.append(f"{renamed} renamed to avoid collision")
    if skipped:
        parts.append(f"{skipped} skipped (already extracted)")
    if overwritten:
        parts.append(f"{overwritten} overwritten")
    parts.append(f"{errors} errors")
    print(f"\nSummary: {', '.join(parts)}")
    return 1 if errors else 0


def _resolve_target(
    base: Path, source_name: str, overwrite: bool
) -> tuple[str, Path]:
    """Decide where (and how) to extract this zip.

    Walks ``base``, ``base-1``, ``base-2``, ... and inspects each
    candidate's marker file:

    * Candidate does not exist -> ("extract", candidate).
    * Candidate's marker matches ``source_name`` -> already extracted from
      this zip. Returns ("replace", candidate) if ``overwrite``, else
      ("skip", candidate).
    * Candidate exists with no marker, or a marker for a different zip
      (organic or different source) -> never touched, walk on.
    """
    for i in range(MAX_SUFFIX + 1):
        candidate = base if i == 0 else base.with_name(f"{base.name}-{i}")
        if not candidate.exists():
            return ("extract", candidate)
        marker = _read_marker(candidate)
        if marker == source_name:
            return ("replace" if overwrite else "skip", candidate)
        # Organic folder or marker for a different zip — keep walking.
    raise RuntimeError(
        f"exceeded {MAX_SUFFIX} suffix candidates for {base.name}; "
        "refusing to keep searching"
    )


def _read_marker(target: Path) -> str | None:
    """Return the source filename recorded in ``target``'s marker, if any."""
    marker = target / MARKER_NAME
    if not marker.is_file():
        return None
    try:
        content = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return content or None


def _write_marker(target: Path, source_name: str) -> None:
    (target / MARKER_NAME).write_text(source_name, encoding="utf-8")


def _extract_safely(zip_path: Path, target: Path) -> None:
    """Extract zip into target, rejecting entries that escape target."""
    target.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            # Resolve the would-be destination and ensure it stays inside target.
            dest = (target / member.filename).resolve()
            try:
                dest.relative_to(target_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"unsafe path in archive: {member.filename!r}"
                ) from exc
        zf.extractall(target)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively extract every .zip under a folder into a sibling "
            "dir named <stem>_unzip/. Idempotent: re-running skips zips "
            "already extracted (tracked via a hidden marker file). "
            "Renames to -1, -2, ... only when an organic folder or a "
            "different zip already occupies the slot. Never deletes "
            "organic folders, even with --overwrite."
        ),
    )
    parser.add_argument("root_folder", help="Folder to scan recursively for .zip files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be extracted without writing anything.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-extract zips that were already extracted by a previous run "
            "(marker matches). Only touches folders this script created — "
            "organic folders are still left alone."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
