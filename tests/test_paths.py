import os
from unittest.mock import patch

import pytest

from server.paths import to_relative, to_absolute


def test_roundtrip_sibling_corpus(tmp_path):
    """A corpus file sibling to the index round-trips through ../ paths."""
    db = tmp_path / "drive" / "lancedb"
    db.mkdir(parents=True)
    corpus = tmp_path / "drive" / "corpus"
    corpus.mkdir()
    f = corpus / "doc.md"
    f.write_text("hello")

    rel = to_relative(str(f), str(db))
    assert rel == os.path.join("..", "corpus", "doc.md")

    back = to_absolute(rel, str(db))
    assert os.path.normpath(back) == os.path.normpath(str(f))


def test_file_at_db_dir_level(tmp_path):
    """A file directly inside the index directory has no ../ segments."""
    db = tmp_path / "drive" / "lancedb"
    db.mkdir(parents=True)
    f = db / "doc.md"
    f.write_text("hello")

    rel = to_relative(str(f), str(db))
    assert rel == "doc.md"
    assert ".." not in rel


def test_trailing_slash_normalized(tmp_path):
    """Trailing slashes do not change the relative path."""
    db = tmp_path / "lancedb"
    db.mkdir()
    sub = tmp_path / "corpus" / "sub"
    sub.mkdir(parents=True)

    assert to_relative(str(sub) + "/", str(db)) == to_relative(str(sub), str(db))


def test_to_absolute_resolves_against_current_db_location(tmp_path):
    """The same relative path resolves differently per index location."""
    rel = os.path.join("..", "corpus", "doc.md")
    abs_a = to_absolute(rel, str(tmp_path / "mount_a" / "lancedb"))
    abs_b = to_absolute(rel, str(tmp_path / "mount_b" / "lancedb"))
    assert abs_a.endswith(os.path.join("mount_a", "corpus", "doc.md"))
    assert abs_b.endswith(os.path.join("mount_b", "corpus", "doc.md"))


def _fake_stat(path, *args, **kwargs):
    dev = 1 if str(path).rstrip("/").endswith("lancedb") else 2
    return type("S", (), {"st_dev": dev})()


def test_volume_mismatch_raises(tmp_path):
    """A file on a different volume than the index is rejected."""
    db = tmp_path / "lancedb"
    db.mkdir()
    f = tmp_path / "doc.md"
    f.write_text("hello")

    with patch("server.paths.os.stat", side_effect=_fake_stat):
        with pytest.raises(ValueError, match="different volume"):
            to_relative(str(f), str(db))


def test_volume_check_skipped_when_disabled(tmp_path):
    """check_volume=False suppresses the cross-volume guard."""
    db = tmp_path / "lancedb"
    db.mkdir()
    f = tmp_path / "doc.md"
    f.write_text("hello")

    with patch("server.paths.os.stat", side_effect=_fake_stat):
        rel = to_relative(str(f), str(db), check_volume=False)
    assert isinstance(rel, str)


def test_volume_check_skipped_for_missing_path(tmp_path):
    """A path that does not exist on disk skips the volume guard."""
    db = tmp_path / "lancedb"
    db.mkdir()
    rel = to_relative(str(tmp_path / "ghost.md"), str(db))
    assert isinstance(rel, str)
    assert "ghost.md" in rel
