"""Tests for scripts/ppt_to_pdf_folder.py — standalone batch PowerPoint→PDF
converter (recursive, mirror-layout, run-log backed).

LibreOffice is not available in the test environment, so the real conversion
boundary ``_convert_one`` is monkeypatched to write a fake PDF (or raise) —
the same fake-the-downstream pattern as tests/test_unpack_mbox_folder.py. The
unit under test is the script's discovery + mirror layout + collision +
idempotency + error-isolation behaviour, not LibreOffice itself.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


# -- helpers --------------------------------------------------------------


def _runlog_items(dst: Path, script: str = "ppt_to_pdf_folder") -> list[dict]:
    logs = sorted((dst / "_runlogs").glob(f"{script}-*.jsonl"))
    assert logs, "no run-log written"
    items = []
    for line in logs[-1].read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["event"] == "item":
            items.append(rec)
    return items


def _drop(path: Path, content: bytes = b"\x00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _dummy_soffice(tmp_path: Path) -> str:
    """A real on-disk file so _find_soffice(--soffice ...) accepts it; it is
    never executed because _convert_one is faked."""
    p = tmp_path / "soffice"
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return str(p)


def _install_fake_convert(monkeypatch, reader=None):
    """Patch ppt_to_pdf_folder._convert_one. Default writes a fake PDF; pass a
    custom ``reader(src_path, out_path, *, soffice, timeout)`` to inject
    failures."""
    from scripts import ppt_to_pdf_folder

    def default(src_path, out_path, *, soffice, timeout):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4 fake\n")

    monkeypatch.setattr(ppt_to_pdf_folder, "_convert_one", reader or default)


# -- discovery + mirror layout -------------------------------------------


def test_mirrors_subdir_layout_and_leaves_source_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop(src / "a" / "talk.pptx")
    _drop(src / "b" / "deep" / "deck.ppt")
    _drop(src / "a" / "notes.txt")  # non-presentation → ignored
    _install_fake_convert(monkeypatch)

    rc = main([str(src), str(dst), "--soffice", _dummy_soffice(tmp_path)])
    assert rc == 0

    assert (dst / "a" / "talk.pdf").is_file()
    assert (dst / "b" / "deep" / "deck.pdf").is_file()
    assert not (dst / "a" / "notes.pdf").exists()
    # Source untouched.
    assert (src / "a" / "talk.pptx").is_file()
    assert not (src / "a" / "talk.pdf").exists()


def test_full_extension_set_case_insensitive(tmp_path: Path, monkeypatch) -> None:
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    for name in (
        "a.PPT", "b.Pptx", "c.pps", "d.ppsx", "e.pot",
        "f.potx", "g.pptm", "h.ppsm", "i.potm", "j.odp",
    ):
        _drop(src / name)
    _drop(src / "skip.docx")  # not a presentation
    _install_fake_convert(monkeypatch)

    rc = main([str(src), str(dst), "--soffice", _dummy_soffice(tmp_path)])
    assert rc == 0

    produced = {p.name for p in dst.rglob("*.pdf")}
    assert produced == {
        "a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf",
        "f.pdf", "g.pdf", "h.pdf", "i.pdf", "j.pdf",
    }
    assert not (dst / "skip.pdf").exists()


def test_stem_collision_in_same_dir_gets_suffix(tmp_path: Path, monkeypatch) -> None:
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop(src / "talk.ppt")
    _drop(src / "talk.pptx")
    _install_fake_convert(monkeypatch)

    rc = main([str(src), str(dst), "--soffice", _dummy_soffice(tmp_path)])
    assert rc == 0
    pdfs = sorted(p.name for p in dst.glob("*.pdf"))
    assert pdfs == ["talk-1.pdf", "talk.pdf"]  # no overwrite


# -- idempotent re-run ----------------------------------------------------


def test_rerun_skips_done_and_writes_nothing_new(tmp_path: Path, monkeypatch) -> None:
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop(src / "a" / "talk.pptx")
    _drop(src / "deck.ppt")
    soffice = _dummy_soffice(tmp_path)
    _install_fake_convert(monkeypatch)

    assert main([str(src), str(dst), "--soffice", soffice]) == 0
    snapshot = {p: p.stat().st_mtime_ns for p in dst.rglob("*.pdf")}

    # Tripwire: second run must not convert anything.
    from scripts import ppt_to_pdf_folder

    def tripwire(*a, **k):
        raise AssertionError("_convert_one should not run on idempotent skip")

    monkeypatch.setattr(ppt_to_pdf_folder, "_convert_one", tripwire)

    assert main([str(src), str(dst), "--soffice", soffice]) == 0
    assert {it["status"] for it in _runlog_items(dst)} == {"skip"}
    # No new files, none rewritten.
    assert {p: p.stat().st_mtime_ns for p in dst.rglob("*.pdf")} == snapshot


def test_retry_after_fix_only_reprocesses_the_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import ppt_to_pdf_folder
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop(src / "good.pptx")
    _drop(src / "bad.pptx")
    soffice = _dummy_soffice(tmp_path)

    def run1(src_path, out_path, *, soffice, timeout):
        if Path(src_path).name == "bad.pptx":
            raise ppt_to_pdf_folder.ConversionError("boom")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4 fake\n")

    monkeypatch.setattr(ppt_to_pdf_folder, "_convert_one", run1)
    assert main([str(src), str(dst), "--soffice", soffice]) == 1  # one failure

    # "Fix" it: now everything converts. Re-run.
    converted: list[str] = []

    def run2(src_path, out_path, *, soffice, timeout):
        converted.append(Path(src_path).name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4 fake\n")

    monkeypatch.setattr(ppt_to_pdf_folder, "_convert_one", run2)
    assert main([str(src), str(dst), "--soffice", soffice]) == 0

    assert converted == ["bad.pptx"]  # good.pptx was skipped (already done)
    statuses = {it["input"]: it["status"] for it in _runlog_items(dst)}
    assert statuses == {"good.pptx": "skip", "bad.pptx": "ok"}


# -- per-file error isolation --------------------------------------------


def test_one_failure_does_not_abort_batch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scripts import ppt_to_pdf_folder
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop(src / "good.pptx")
    _drop(src / "bad.pptx")

    def reader(src_path, out_path, *, soffice, timeout):
        if Path(src_path).name == "bad.pptx":
            raise ppt_to_pdf_folder.ConversionError("intentional")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4 fake\n")

    monkeypatch.setattr(ppt_to_pdf_folder, "_convert_one", reader)
    rc = main([str(src), str(dst), "--soffice", _dummy_soffice(tmp_path)])
    assert rc == 1  # ≥1 failure → non-zero

    assert (dst / "good.pdf").is_file()
    assert not (dst / "bad.pdf").exists()
    err = capsys.readouterr().err
    assert "bad.pptx" in err
    statuses = {it["input"]: it["status"] for it in _runlog_items(dst)}
    assert statuses == {"good.pptx": "ok", "bad.pptx": "fail"}


# -- soffice detection ----------------------------------------------------


def test_find_soffice_honours_explicit_arg(tmp_path: Path) -> None:
    from scripts.ppt_to_pdf_folder import _find_soffice

    dummy = _dummy_soffice(tmp_path)
    assert _find_soffice(dummy) == dummy


def test_find_soffice_explicit_missing_is_none(tmp_path: Path) -> None:
    from scripts.ppt_to_pdf_folder import _find_soffice

    assert _find_soffice(str(tmp_path / "nope")) is None


def test_find_soffice_uses_env(tmp_path: Path, monkeypatch) -> None:
    from scripts import ppt_to_pdf_folder

    dummy = _dummy_soffice(tmp_path)
    monkeypatch.setenv("SOFFICE", dummy)
    # No PATH soffice in the test env; env var should win.
    monkeypatch.setattr(ppt_to_pdf_folder.shutil, "which", lambda *_: None)
    assert ppt_to_pdf_folder._find_soffice(None) == dummy


def test_find_soffice_uses_path(tmp_path: Path, monkeypatch) -> None:
    from scripts import ppt_to_pdf_folder

    monkeypatch.delenv("SOFFICE", raising=False)
    monkeypatch.setattr(
        ppt_to_pdf_folder.shutil, "which",
        lambda name: "/usr/bin/soffice" if name == "soffice" else None,
    )
    assert ppt_to_pdf_folder._find_soffice(None) == "/usr/bin/soffice"


def test_missing_engine_errors_clearly(tmp_path: Path, monkeypatch, capsys) -> None:
    from scripts import ppt_to_pdf_folder
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop(src / "talk.pptx")
    monkeypatch.setattr(ppt_to_pdf_folder, "_find_soffice", lambda arg: None)

    rc = main([str(src), str(dst)])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "libreoffice" in err or "soffice" in err
    assert not (dst / "talk.pdf").exists()


# -- guards ---------------------------------------------------------------


def test_input_must_be_existing_directory(tmp_path: Path) -> None:
    from scripts.ppt_to_pdf_folder import main

    rc = main([str(tmp_path / "nope"), str(tmp_path / "out"),
               "--soffice", _dummy_soffice(tmp_path)])
    assert rc != 0


def test_output_inside_input_refused(tmp_path: Path, monkeypatch) -> None:
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    src.mkdir()
    _drop(src / "talk.pptx")
    _install_fake_convert(monkeypatch)

    inside = src / "out"
    rc = main([str(src), str(inside), "--soffice", _dummy_soffice(tmp_path)])
    assert rc != 0
    assert not inside.exists() or not any(inside.rglob("*.pdf"))


# -- dry run --------------------------------------------------------------


def test_dry_run_lists_targets_and_writes_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scripts import ppt_to_pdf_folder
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop(src / "a" / "talk.pptx")

    def boom(*a, **k):
        raise AssertionError("dry-run must not convert")

    monkeypatch.setattr(ppt_to_pdf_folder, "_convert_one", boom)
    rc = main([str(src), str(dst), "--dry-run"])
    assert rc == 0

    assert not list(dst.rglob("*.pdf"))
    assert not (dst / "_runlogs").exists()
    captured = capsys.readouterr()
    assert "talk.pptx" in captured.out + captured.err


# -- summary --------------------------------------------------------------


def test_summary_line_reports_counts(tmp_path: Path, monkeypatch, capsys) -> None:
    from scripts.ppt_to_pdf_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop(src / "talk.pptx")
    _install_fake_convert(monkeypatch)

    assert main([str(src), str(dst), "--soffice", _dummy_soffice(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "Converted 1" in err


# -- real-engine smoke test (auto-skipped without LibreOffice / python-pptx) --


def test_real_soffice_converts_a_pptx(tmp_path: Path) -> None:
    """Exercise the genuine LibreOffice path end-to-end when the engine and
    python-pptx are both available; otherwise skip. Proves _convert_one's real
    soffice invocation and the temp-dir → move plumbing actually produce a PDF."""
    import pytest

    from scripts.ppt_to_pdf_folder import _find_soffice, main

    soffice = _find_soffice(None)
    if soffice is None:
        pytest.skip("LibreOffice not installed")
    pptx = pytest.importorskip("pptx")  # python-pptx, to author a real input

    src = tmp_path / "src"
    src.mkdir()
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Hello smoke test"
    prs.save(str(src / "deck.pptx"))

    dst = tmp_path / "dst"
    rc = main([str(src), str(dst), "--soffice", soffice])
    assert rc == 0
    out = dst / "deck.pdf"
    assert out.is_file() and out.stat().st_size > 0
    assert out.read_bytes()[:5] == b"%PDF-"
