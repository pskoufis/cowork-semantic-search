"""Real-binary-.msg integration tests.

The unit tests in test_msg_unpack.py monkeypatch read_message; this file
exercises the actual extract-msg parser against the two real .msg fixtures
in example-msg-files/. Skipped when extract-msg isn't installed.

Locks in the date-fallback fix for messages without clientSubmitTime
(strangeDate.msg) and verifies attachment bytes round-trip exactly
(unicode.msg).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

pytest.importorskip("extract_msg")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "example-msg-files"
EXPECTED_DIR = FIXTURES_DIR / "expected-outputs"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.fixture
def scratch_src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for name in ("unicode.msg", "strangeDate.msg"):
        shutil.copy2(FIXTURES_DIR / name, src / name)
    return src


def test_unicode_msg_extracts_body_and_byte_equal_attachments(
    scratch_src: Path, tmp_path: Path
) -> None:
    from msg_handling.unpack import ensure_unpacked

    dst = tmp_path / "dst"
    ensure_unpacked(scratch_src / "unicode.msg", target=dst)

    txt = (dst / "unicode.txt").read_text(encoding="utf-8")
    assert "Subject: Test for TIF files" in txt
    assert "Brian Zhou" in txt
    assert "This is a test email to experiment with the MS Outlook MSG Extractor" in txt
    # Header line is present and populated (regression for the date-fallback fix).
    date_line = next(line for line in txt.splitlines() if line.startswith("Date:"))
    assert "2013" in date_line, date_line

    # Both .tif attachments round-trip byte-for-byte against expected outputs.
    expected_outer = EXPECTED_DIR / "2013-11-18_0026 Test for TIF files"
    for name in ("import OleFileIO.tif", "raised value error.tif"):
        ours = dst / "attachments" / f"unicode__{name}"
        theirs = expected_outer / name
        assert ours.is_file(), f"missing {ours}"
        assert _md5(ours) == _md5(theirs), f"byte mismatch on {name}"


def test_strange_date_msg_falls_back_to_creation_time(
    scratch_src: Path, tmp_path: Path
) -> None:
    """Regression: strangeDate.msg has no clientSubmitTime / receivedTime /
    Date header. Before the fallback fix the Date line was empty. It should
    now contain the PR_CREATION_TIME (2016-02-23) instead."""
    from msg_handling.unpack import ensure_unpacked

    dst = tmp_path / "dst"
    ensure_unpacked(scratch_src / "strangeDate.msg", target=dst)

    txt = (dst / "strangeDate.txt").read_text(encoding="utf-8")
    date_line = next(line for line in txt.splitlines() if line.startswith("Date:"))
    assert "2016" in date_line, date_line
    assert "Feb" in date_line, date_line
    # Body content sanity check — the test file's body is a Wikipedia-style
    # description of the "John Doe" placeholder convention.
    assert "John Doe" in txt
    assert "Subject: MSG Test File" in txt
