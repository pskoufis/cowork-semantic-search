"""Sort a flat folder of files into buckets for downstream indexing.

Usage:
    python scripts/split_for_indexing.py INPUT_DIR OUTPUT_DIR [--batch-size 10000]

The input is assumed to be a flat folder (subdirectories are ignored).
Each file is copied (not moved) into one of four buckets under
``OUTPUT_DIR``:

* ``BATCHES/`` — text-like (.txt .htm .html .rtf) up to 20 MiB and
  document-like (.pdf .doc .docx .eml .msg .wpd) up to 500 MiB are
  written into 10K-file ``batch_NNN.zip`` archives.
* ``TOO_LARGE/`` — files of those extensions that exceed the size cap.
* ``TO_PDF/`` — image extensions (jpg/jpeg/png/tif/tiff/bmp/gif/heic/
  heif/webp/jp2) regardless of size.
* ``OTHER_EXTENSIONS/`` — everything else, including files with no
  extension.

Extension matching is case-insensitive. Dotfiles (``.DS_Store`` etc.)
and subdirectories in the input are silently skipped. ``OUTPUT_DIR``
must not exist or must be empty.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

MIB = 1024 * 1024
TEXT_LIMIT = 20 * MIB
DOC_LIMIT = 500 * MIB

TEXT_EXTS = {".txt", ".htm", ".html", ".rtf"}
DOC_EXTS = {".pdf", ".doc", ".docx", ".eml", ".msg", ".wpd"}
IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".webp",
    ".jp2",
}

BATCHES = "BATCHES"
TOO_LARGE = "TOO_LARGE"
TO_PDF = "TO_PDF"
OTHER = "OTHER_EXTENSIONS"

DEFAULT_BATCH_SIZE = 10_000


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src = Path(args.input_dir).expanduser().resolve()
    dst = Path(args.output_dir).expanduser().resolve()

    if not src.exists() or not src.is_dir():
        print(f"ERROR: input dir {src} does not exist or is not a directory", file=sys.stderr)
        return 2
    if dst.exists():
        if not dst.is_dir():
            print(f"ERROR: output path {dst} exists and is not a directory", file=sys.stderr)
            return 2
        if any(dst.iterdir()):
            print(f"ERROR: output dir {dst} is not empty", file=sys.stderr)
            return 2

    for sub in (BATCHES, TOO_LARGE, TO_PDF, OTHER):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    counts = {BATCHES: 0, TOO_LARGE: 0, TO_PDF: 0, OTHER: 0}
    bytes_per_bucket = {BATCHES: 0, TOO_LARGE: 0, TO_PDF: 0, OTHER: 0}

    progress = _make_progress()
    writer = _BatchWriter(dst / BATCHES, args.batch_size)
    interrupted = False

    try:
        with os.scandir(src) as it:
            for entry in it:
                try:
                    if entry.name.startswith("."):
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    size = entry.stat(follow_symlinks=False).st_size
                except OSError as exc:
                    print(f"WARN: skipping {entry.path}: {exc}", file=sys.stderr)
                    continue

                bucket = _classify(entry.name, size)
                try:
                    if bucket is BATCHES:
                        writer.add(entry.path, entry.name)
                    else:
                        shutil.copy2(entry.path, dst / bucket / entry.name)
                except (PermissionError, FileNotFoundError) as exc:
                    print(f"WARN: failed to copy {entry.path}: {exc}", file=sys.stderr)
                    continue

                counts[bucket] += 1
                bytes_per_bucket[bucket] += size
                progress.tick(counts, writer.current_name())
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — closing current batch and exiting.", file=sys.stderr)
    finally:
        writer.close()
        progress.close()

    _print_summary(counts, bytes_per_bucket, writer.batch_count())
    if interrupted:
        return 130
    return 0


def _classify(name: str, size: int) -> str:
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXTS:
        return BATCHES if size <= TEXT_LIMIT else TOO_LARGE
    if ext in DOC_EXTS:
        return BATCHES if size <= DOC_LIMIT else TOO_LARGE
    if ext in IMAGE_EXTS:
        return TO_PDF
    return OTHER


class _BatchWriter:
    """Lazy 10K-per-zip writer rolling batch_001.zip, batch_002.zip, ..."""

    def __init__(self, batches_dir: Path, batch_size: int) -> None:
        self._dir = batches_dir
        self._batch_size = batch_size
        self._index = 0
        self._files_in_current = 0
        self._zf: zipfile.ZipFile | None = None

    def add(self, src_path: str, arcname: str) -> None:
        if self._zf is None or self._files_in_current >= self._batch_size:
            self._roll()
        assert self._zf is not None
        self._zf.write(src_path, arcname=arcname)
        self._files_in_current += 1

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def batch_count(self) -> int:
        return self._index

    def current_name(self) -> str | None:
        if self._index == 0:
            return None
        return f"batch_{self._index:03d}.zip"

    def _roll(self) -> None:
        if self._zf is not None:
            self._zf.close()
        self._index += 1
        self._files_in_current = 0
        path = self._dir / f"batch_{self._index:03d}.zip"
        self._zf = zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED)


class _Progress:
    """tqdm-backed progress with a plain-text fallback every N files."""

    _FALLBACK_EVERY = 500

    def __init__(self) -> None:
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:
            self._bar = None
        else:
            self._bar = tqdm(total=None, unit="file")
        self._seen = 0

    def tick(self, counts: dict[str, int], cur_zip: str | None) -> None:
        self._seen += 1
        if self._bar is not None:
            self._bar.update(1)
            self._bar.set_postfix(
                batches=counts[BATCHES],
                too_large=counts[TOO_LARGE],
                to_pdf=counts[TO_PDF],
                other=counts[OTHER],
                cur_zip=cur_zip or "-",
            )
        elif self._seen % self._FALLBACK_EVERY == 0:
            print(f"processed {self._seen} files", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _make_progress() -> _Progress:
    return _Progress()


def _print_summary(counts: dict[str, int], byte_counts: dict[str, int], batches: int) -> None:
    total = sum(counts.values())
    print()
    print(f"Done. {total} files processed across {batches} batch zip(s).")
    for name in (BATCHES, TOO_LARGE, TO_PDF, OTHER):
        mib = byte_counts[name] / MIB
        print(f"  {name:<17} {counts[name]:>8} files  {mib:>10.1f} MiB")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sort a flat folder of files into BATCHES/ (zipped 10K-file "
            "chunks of in-spec text/document files), TOO_LARGE/, TO_PDF/, "
            "and OTHER_EXTENSIONS/ buckets for downstream indexing."
        ),
    )
    parser.add_argument("input_dir", help="Flat folder of files to sort.")
    parser.add_argument(
        "output_dir",
        help="Destination root. Must not exist, or must exist and be empty.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Files per batch zip (default {DEFAULT_BATCH_SIZE}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
