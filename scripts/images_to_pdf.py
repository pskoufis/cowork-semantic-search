"""Convert a flat folder of image files into one PDF per image.

Usage:
    python scripts/images_to_pdf.py INPUT_DIR OUTPUT_DIR

Each image directly under ``INPUT_DIR`` is converted into ``<stem>.pdf`` under
``OUTPUT_DIR``. The originals are left untouched. The input is treated as flat:
subdirectories and dotfiles are skipped (no recursion).

The set of recognised image extensions is the same set
``scripts/split_for_indexing.py`` routes into its ``TO_PDF/`` bucket:
``.jpg .jpeg .png .tif .tiff .bmp .gif .heic .heif .webp .jp2`` (matched
case-insensitively). Anything else is skipped with a ``WARN``.

Behaviour notes:

* Modes that a PDF cannot store directly (RGBA / LA / P) are flattened to RGB,
  compositing any alpha channel onto a white background.
* Multi-frame inputs (multi-page TIFF, animated GIF) convert the **first frame
  only** and emit a ``WARN`` noting the dropped frames.
* If two inputs map to the same stem (``foo.jpg`` + ``foo.png``), the second and
  later get a numeric suffix (``foo-1.pdf``, ``foo-2.pdf``) and a ``WARN`` — no
  silent overwrite.

A single unreadable or corrupt image logs a ``WARN`` and is skipped; it never
aborts the run. ``OUTPUT_DIR`` must not exist or must be empty.

Requires the ``images`` optional dependencies (``pip install -e '.[images]'``):
Pillow and pillow-heif.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

# Pillow image modes that PDF output cannot represent directly and must be
# flattened to RGB before saving.
_NON_RGB_MODES = {"RGBA", "LA", "P", "PA"}


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

    # Import lazily so --help and the pre-flight checks work without the
    # optional image dependencies installed.
    try:
        from PIL import Image
        from pillow_heif import register_heif_opener
    except ImportError as exc:
        print(
            f"ERROR: image dependencies missing ({exc}). "
            "Install with: pip install -e '.[images]'",
            file=sys.stderr,
        )
        return 2
    register_heif_opener()

    dst.mkdir(parents=True, exist_ok=True)

    converted = skipped = failed = 0
    used_stems: set[str] = set()
    progress = _make_progress()

    interrupted = False
    try:
        with os.scandir(src) as it:
            for entry in it:
                try:
                    if entry.name.startswith("."):
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError as exc:
                    print(f"WARN: skipping {entry.path}: {exc}", file=sys.stderr)
                    skipped += 1
                    continue

                if Path(entry.name).suffix.lower() not in IMAGE_EXTS:
                    skipped += 1
                    continue

                out_path = _unique_out_path(dst, entry.name, used_stems)
                if _convert_one(Image, entry.path, out_path):
                    converted += 1
                else:
                    failed += 1
                progress.tick(converted, skipped, failed)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — exiting.", file=sys.stderr)
    finally:
        progress.close()

    _print_summary(converted, skipped, failed)
    if interrupted:
        return 130
    return 0


def _unique_out_path(dst: Path, src_name: str, used_stems: set[str]) -> Path:
    """Reserve a collision-free ``<stem>.pdf`` path under ``dst``.

    Records the chosen stem in ``used_stems`` so the next caller mapping to the
    same stem gets a ``-1``, ``-2``, ... suffix instead of overwriting.
    """
    stem = Path(src_name).stem
    candidate = stem
    n = 0
    while candidate in used_stems:
        n += 1
        candidate = f"{stem}-{n}"
    if n:
        print(
            f"WARN: name collision for {src_name!r}; writing {candidate}.pdf",
            file=sys.stderr,
        )
    used_stems.add(candidate)
    return dst / f"{candidate}.pdf"


def _convert_one(Image, src_path: str, out_path: Path) -> bool:
    """Convert a single image file to ``out_path``. Returns success."""
    from PIL import ImageOps

    try:
        with Image.open(src_path) as img:
            n_frames = getattr(img, "n_frames", 1)
            if n_frames > 1:
                print(
                    f"WARN: {src_path} has {n_frames} frames; converting first frame only",
                    file=sys.stderr,
                )
                img.seek(0)
            # Honour the EXIF orientation tag (phone photos / scans set it) so
            # the PDF isn't written sideways; Pillow's PDF save ignores it.
            # Must come after frame selection — it returns a flat single-frame copy.
            img = ImageOps.exif_transpose(img)
            if img.mode in _NON_RGB_MODES:
                img = _flatten_to_rgb(Image, img)
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(out_path, "PDF")
        return True
    except Exception as exc:  # noqa: BLE001 — one bad file must not abort the run
        print(f"WARN: skipping {src_path}: {exc}", file=sys.stderr)
        # Drop a half-written PDF if save() failed partway.
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _flatten_to_rgb(Image, img):
    """Composite an alpha-bearing or palette image onto white, return RGB."""
    rgba = img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


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

    def tick(self, converted: int, skipped: int, failed: int) -> None:
        self._seen += 1
        if self._bar is not None:
            self._bar.update(1)
            self._bar.set_postfix(converted=converted, skipped=skipped, failed=failed)
        elif self._seen % self._FALLBACK_EVERY == 0:
            print(f"processed {self._seen} files", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _make_progress() -> _Progress:
    return _Progress()


def _print_summary(converted: int, skipped: int, failed: int) -> None:
    print()
    print(
        f"Done. {converted} converted, {skipped} skipped (non-image), "
        f"{failed} failed."
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a flat folder of image files into one PDF per image. "
            "Recognises the same image extensions split_for_indexing.py routes "
            "into TO_PDF/."
        ),
    )
    parser.add_argument("input_dir", help="Flat folder of image files.")
    parser.add_argument(
        "output_dir",
        help="Destination for PDFs. Must not exist, or must exist and be empty.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
