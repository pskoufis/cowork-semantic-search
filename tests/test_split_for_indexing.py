"""Tests for scripts/split_for_indexing.py — flat-folder sorter."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest


def _write(p: Path, size: int = 1) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    return p


def _list_zip(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return sorted(zf.namelist())


# --- size-bound classification -----------------------------------------------


def test_text_like_under_limit_goes_to_batches(tmp_path: Path, monkeypatch) -> None:
    from scripts import split_for_indexing as mod

    monkeypatch.setattr(mod, "TEXT_LIMIT", 1024)
    monkeypatch.setattr(mod, "DOC_LIMIT", 1024)

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "note.txt", 100)
    _write(src / "page.html", 100)
    _write(src / "page.htm", 100)
    _write(src / "doc.rtf", 100)

    assert mod.main([str(src), str(dst)]) == 0

    batches = sorted((dst / "BATCHES").glob("*.zip"))
    assert len(batches) == 1
    assert _list_zip(batches[0]) == ["doc.rtf", "note.txt", "page.htm", "page.html"]
    assert list((dst / "TOO_LARGE").iterdir()) == []


def test_text_like_at_exact_limit_is_in_spec(tmp_path: Path, monkeypatch) -> None:
    from scripts import split_for_indexing as mod

    monkeypatch.setattr(mod, "TEXT_LIMIT", 100)
    monkeypatch.setattr(mod, "DOC_LIMIT", 100)

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "boundary.txt", 100)

    assert mod.main([str(src), str(dst)]) == 0

    batches = sorted((dst / "BATCHES").glob("*.zip"))
    assert len(batches) == 1
    assert _list_zip(batches[0]) == ["boundary.txt"]


def test_text_like_over_limit_goes_to_too_large(tmp_path: Path, monkeypatch) -> None:
    from scripts import split_for_indexing as mod

    monkeypatch.setattr(mod, "TEXT_LIMIT", 100)
    monkeypatch.setattr(mod, "DOC_LIMIT", 100)

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "big.txt", 101)

    assert mod.main([str(src), str(dst)]) == 0

    assert list((dst / "BATCHES").iterdir()) == []
    too_large = sorted(p.name for p in (dst / "TOO_LARGE").iterdir())
    assert too_large == ["big.txt"]


def test_document_like_under_limit_goes_to_batches(tmp_path: Path, monkeypatch) -> None:
    from scripts import split_for_indexing as mod

    monkeypatch.setattr(mod, "TEXT_LIMIT", 1024)
    monkeypatch.setattr(mod, "DOC_LIMIT", 1024)

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    for name in ("a.pdf", "b.doc", "c.docx", "d.eml", "e.msg", "f.wpd"):
        _write(src / name, 200)

    assert mod.main([str(src), str(dst)]) == 0

    batches = sorted((dst / "BATCHES").glob("*.zip"))
    assert len(batches) == 1
    assert _list_zip(batches[0]) == ["a.pdf", "b.doc", "c.docx", "d.eml", "e.msg", "f.wpd"]


def test_document_like_over_limit_goes_to_too_large(tmp_path: Path, monkeypatch) -> None:
    from scripts import split_for_indexing as mod

    monkeypatch.setattr(mod, "TEXT_LIMIT", 50)
    monkeypatch.setattr(mod, "DOC_LIMIT", 100)

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "huge.pdf", 200)

    assert mod.main([str(src), str(dst)]) == 0

    assert list((dst / "BATCHES").iterdir()) == []
    too_large = sorted(p.name for p in (dst / "TOO_LARGE").iterdir())
    assert too_large == ["huge.pdf"]


# --- non-batch buckets -------------------------------------------------------


def test_images_go_to_to_pdf(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    for name in (
        "a.jpg",
        "b.jpeg",
        "c.png",
        "d.tif",
        "e.tiff",
        "f.bmp",
        "g.gif",
        "h.heic",
        "i.heif",
        "j.webp",
        "k.jp2",
    ):
        _write(src / name, 50)

    assert mod.main([str(src), str(dst)]) == 0

    names = sorted(p.name for p in (dst / "TO_PDF").iterdir())
    assert names == [
        "a.jpg",
        "b.jpeg",
        "c.png",
        "d.tif",
        "e.tiff",
        "f.bmp",
        "g.gif",
        "h.heic",
        "i.heif",
        "j.webp",
        "k.jp2",
    ]
    assert list((dst / "BATCHES").iterdir()) == []


def test_unknown_extension_goes_to_other(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "weird.xyz", 10)
    _write(src / "script.sh", 10)

    assert mod.main([str(src), str(dst)]) == 0

    names = sorted(p.name for p in (dst / "OTHER_EXTENSIONS").iterdir())
    assert names == ["script.sh", "weird.xyz"]


def test_file_without_extension_goes_to_other(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "README", 10)
    _write(src / "Makefile", 10)

    assert mod.main([str(src), str(dst)]) == 0

    names = sorted(p.name for p in (dst / "OTHER_EXTENSIONS").iterdir())
    assert names == ["Makefile", "README"]


# --- edge cases --------------------------------------------------------------


def test_extension_match_is_case_insensitive(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "shouty.PDF", 50)
    _write(src / "PHOTO.JPG", 50)

    assert mod.main([str(src), str(dst)]) == 0

    batches = sorted((dst / "BATCHES").glob("*.zip"))
    assert len(batches) == 1
    assert _list_zip(batches[0]) == ["shouty.PDF"]
    assert sorted(p.name for p in (dst / "TO_PDF").iterdir()) == ["PHOTO.JPG"]


def test_dotfiles_are_skipped(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / ".DS_Store", 10)
    _write(src / ".hidden.txt", 10)
    _write(src / "kept.txt", 10)

    assert mod.main([str(src), str(dst)]) == 0

    batches = sorted((dst / "BATCHES").glob("*.zip"))
    assert len(batches) == 1
    assert _list_zip(batches[0]) == ["kept.txt"]
    # Confirm no dotfile leaked into any bucket.
    for sub in ("BATCHES", "TOO_LARGE", "TO_PDF", "OTHER_EXTENSIONS"):
        for p in (dst / sub).iterdir():
            assert not p.name.startswith(".")


def test_subdirectories_in_input_are_skipped(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "top.txt", 10)
    (src / "nested").mkdir()
    _write(src / "nested" / "buried.txt", 10)

    assert mod.main([str(src), str(dst)]) == 0

    batches = sorted((dst / "BATCHES").glob("*.zip"))
    assert len(batches) == 1
    assert _list_zip(batches[0]) == ["top.txt"]


# --- batch rolling -----------------------------------------------------------


def test_batches_roll_at_configured_size(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    for i in range(7):
        _write(src / f"f{i:02d}.txt", 10)

    assert mod.main([str(src), str(dst), "--batch-size", "3"]) == 0

    batches = sorted((dst / "BATCHES").glob("*.zip"))
    assert [b.name for b in batches] == ["batch_001.zip", "batch_002.zip", "batch_003.zip"]
    sizes = [len(_list_zip(b)) for b in batches]
    assert sizes == [3, 3, 1]
    # All seven files survived across the three zips, with no duplicates.
    all_names = []
    for b in batches:
        all_names.extend(_list_zip(b))
    assert sorted(all_names) == [f"f{i:02d}.txt" for i in range(7)]


# --- pre-flight --------------------------------------------------------------


def test_rejects_non_empty_output_dir(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    _write(dst / "preexisting.txt", 1)

    assert mod.main([str(src), str(dst)]) == 2


def test_accepts_pre_existing_empty_output_dir(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    _write(src / "ok.txt", 10)

    assert mod.main([str(src), str(dst)]) == 0
    batches = sorted((dst / "BATCHES").glob("*.zip"))
    assert len(batches) == 1


def test_rejects_missing_input_dir(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "nope"
    dst = tmp_path / "dst"

    assert mod.main([str(src), str(dst)]) == 2


def test_arcname_is_basename_only(tmp_path: Path) -> None:
    from scripts import split_for_indexing as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "shallow.txt", 10)

    assert mod.main([str(src), str(dst)]) == 0
    batches = sorted((dst / "BATCHES").glob("*.zip"))
    with zipfile.ZipFile(batches[0]) as zf:
        for info in zf.infolist():
            assert info.filename == "shallow.txt"
            assert "/" not in info.filename
