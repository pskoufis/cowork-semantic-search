"""Tests for scripts/images_to_pdf.py — convert a flat folder of images to PDFs."""

from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF, used here only to inspect output PDFs
import pytest
from PIL import Image


def _runlog_items(dst: Path) -> list[dict]:
    logs = sorted((dst / "_runlogs").glob("images_to_pdf-*.jsonl"))
    assert logs, "no run-log written"
    items = []
    for line in logs[-1].read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["event"] == "item":
            items.append(rec)
    return items


def _make_image(path: Path, mode: str = "RGB", size: tuple[int, int] = (8, 8)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = {"RGB": (200, 30, 30), "RGBA": (200, 30, 30, 128), "P": 5, "L": 128}[mode]
    img = Image.new(mode, size, color)
    fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
           ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP"}[path.suffix.lower()]
    # JPEG cannot store alpha/palette; coerce to RGB for that format.
    if fmt == "JPEG" and mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(path, fmt)
    return path


def _make_multipage_tiff(path: Path, pages: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.new("RGB", (8, 8), (i * 40, 0, 0)) for i in range(pages)]
    frames[0].save(path, "TIFF", save_all=True, append_images=frames[1:])
    return path


def _pdf_page_count(path: Path) -> int:
    with fitz.open(path) as doc:
        return doc.page_count


# --- core conversion ---------------------------------------------------------


def test_png_converts_to_single_page_pdf(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "photo.png", "RGB")

    assert mod.main([str(src), str(dst)]) == 0

    out = dst / "photo.pdf"
    assert out.is_file()
    assert _pdf_page_count(out) == 1


def test_jpeg_converts(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "scan.jpg", "RGB")

    assert mod.main([str(src), str(dst)]) == 0
    assert (dst / "scan.pdf").is_file()


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    # Landscape pixels (wide=40, tall=10) tagged Orientation=6 ("rotate 90° CW
    # to display"): a correct converter renders it portrait (10 x 40).
    img = Image.new("RGB", (40, 10), (0, 120, 255))
    exif = img.getexif()
    exif[274] = 6  # 274 = Orientation tag
    img.save(src / "rotated.jpg", "JPEG", exif=exif)

    assert mod.main([str(src), str(dst)]) == 0

    with fitz.open(dst / "rotated.pdf") as doc:
        rect = doc[0].rect
    # Portrait after honouring orientation: height > width.
    assert rect.height > rect.width


def test_rgba_png_is_flattened_not_crashed(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "alpha.png", "RGBA")

    assert mod.main([str(src), str(dst)]) == 0
    assert (dst / "alpha.pdf").is_file()


def test_palette_image_converts(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "pal.png", "P")

    assert mod.main([str(src), str(dst)]) == 0
    assert (dst / "pal.pdf").is_file()


# --- multi-frame handling ----------------------------------------------------


def test_multipage_tiff_yields_multipage_pdf(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_multipage_tiff(src / "fax.tiff", pages=3)

    assert mod.main([str(src), str(dst)]) == 0

    out = dst / "fax.pdf"
    assert out.is_file()
    # All three TIFF pages are preserved as PDF pages.
    assert _pdf_page_count(out) == 3


def test_single_frame_image_yields_single_page_pdf(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "one.png", "RGB")

    assert mod.main([str(src), str(dst)]) == 0
    assert _pdf_page_count(dst / "one.pdf") == 1


def test_animated_gif_yields_multipage_pdf(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    # Visually distinct frames so the GIF encoder doesn't merge them into one.
    frames = [Image.new("RGB", (8, 8), (i * 60, 0, 0)) for i in range(4)]
    frames[0].save(src / "anim.gif", "GIF", save_all=True, append_images=frames[1:])

    assert mod.main([str(src), str(dst)]) == 0
    assert _pdf_page_count(dst / "anim.pdf") == 4


def test_heic_converts(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    Image.new("RGB", (16, 16), (10, 200, 90)).save(src / "phone.heic")

    assert mod.main([str(src), str(dst)]) == 0

    out = dst / "phone.pdf"
    assert out.is_file()
    assert _pdf_page_count(out) == 1


# --- naming ------------------------------------------------------------------


def test_name_collision_gets_numeric_suffix(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "foo.jpg", "RGB")
    _make_image(src / "foo.png", "RGB")

    assert mod.main([str(src), str(dst)]) == 0

    produced = sorted(p.name for p in dst.glob("*.pdf"))
    assert produced == ["foo-1.pdf", "foo.pdf"]


def test_extension_match_is_case_insensitive(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "PHOTO.JPG", "RGB")

    assert mod.main([str(src), str(dst)]) == 0
    assert (dst / "PHOTO.pdf").is_file()


# --- skipping ----------------------------------------------------------------


def test_non_image_extension_is_skipped(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "keep.png", "RGB")
    (src / "notes.txt").write_text("hello")

    assert mod.main([str(src), str(dst)]) == 0

    assert (dst / "keep.pdf").is_file()
    assert not (dst / "notes.pdf").exists()
    assert not (dst / "notes.txt.pdf").exists()
    assert sorted(p.name for p in dst.glob("*.pdf")) == ["keep.pdf"]


def test_dotfiles_and_subdirs_are_skipped(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "real.png", "RGB")
    _make_image(src / ".hidden.png", "RGB")
    _make_image(src / "nested" / "buried.png", "RGB")

    assert mod.main([str(src), str(dst)]) == 0

    assert sorted(p.name for p in dst.glob("*.pdf")) == ["real.pdf"]


def test_corrupt_image_is_skipped_run_continues(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "good.png", "RGB")
    (src / "broken.png").write_bytes(b"not a real png")

    assert mod.main([str(src), str(dst)]) == 0

    # Good file still converted; the corrupt one produced no PDF.
    assert (dst / "good.pdf").is_file()
    assert not (dst / "broken.pdf").exists()


# --- pre-flight --------------------------------------------------------------


def test_rejects_non_empty_output_dir(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (dst / "preexisting.pdf").write_bytes(b"x")

    assert mod.main([str(src), str(dst)]) == 2


def test_accepts_pre_existing_empty_output_dir(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    _make_image(src / "ok.png", "RGB")

    assert mod.main([str(src), str(dst)]) == 0
    assert (dst / "ok.pdf").is_file()


def test_rejects_missing_input_dir(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "nope", tmp_path / "dst"
    assert mod.main([str(src), str(dst)]) == 2


# --- run-log + idempotent re-run --------------------------------------------


def test_runlog_records_conversions(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "a.png", "RGB")

    assert mod.main([str(src), str(dst)]) == 0
    items = _runlog_items(dst)
    by_input = {it["input"]: it for it in items}
    assert by_input["a.png"]["status"] == "ok"
    assert by_input["a.png"]["output"].endswith("a.pdf")


def test_rerun_skips_already_converted(tmp_path: Path) -> None:
    """A second run skips done images — no rewrite, no duplicate PDFs, and the
    'output not empty' guard does not fire on our own prior output."""
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "a.png", "RGB")
    _make_image(src / "b.png", "RGB")

    assert mod.main([str(src), str(dst)]) == 0
    before = {p.name: p.stat().st_mtime_ns for p in dst.glob("*.pdf")}

    assert mod.main([str(src), str(dst)]) == 0
    after = {p.name: p.stat().st_mtime_ns for p in dst.glob("*.pdf")}

    assert after == before  # untouched — skipped, not rewritten
    assert {it["status"] for it in _runlog_items(dst)} == {"skip"}


def test_retry_after_fix_converts_only_the_fixed_image(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "good.png", "RGB")
    (src / "broken.png").write_bytes(b"not a real png")

    assert mod.main([str(src), str(dst)]) == 0  # corrupt image just skipped
    assert not (dst / "broken.pdf").exists()

    # "Fix" the broken image with real content and re-run.
    _make_image(src / "broken.png", "RGB")
    assert mod.main([str(src), str(dst)]) == 0

    assert (dst / "broken.pdf").is_file()
    by_input = {it["input"]: it["status"] for it in _runlog_items(dst)}
    assert by_input["broken.png"] == "ok"
    assert by_input["good.png"] == "skip"


def test_force_overwrites_in_place_no_duplicates(tmp_path: Path) -> None:
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_image(src / "a.png", "RGB")

    assert mod.main([str(src), str(dst)]) == 0
    assert mod.main([str(src), str(dst), "--force"]) == 0

    # Reprocessed in place — still exactly one PDF, no a-1.pdf.
    assert sorted(p.name for p in dst.glob("*.pdf")) == ["a.pdf"]
    assert {it["status"] for it in _runlog_items(dst)} == {"ok"}


def test_rejects_foreign_non_empty_output_still(tmp_path: Path) -> None:
    """The 'not empty' guard still protects an output dir we did not create
    (no prior run-log present)."""
    from scripts import images_to_pdf as mod

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (dst / "preexisting.pdf").write_bytes(b"x")

    assert mod.main([str(src), str(dst)]) == 2
