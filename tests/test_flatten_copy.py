"""Tests for scripts/flatten_copy.py — recursive flat-copy utility."""

from __future__ import annotations

import os
from pathlib import Path


def _write(p: Path, content: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_flat_copy_of_top_level_files(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "b.txt", "B")

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / "a.txt").read_text() == "A"
    assert (dst / "b.txt").read_text() == "B"


def test_recursive_walk_flattens_subdirs(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "top.txt", "T")
    _write(src / "sub" / "deep.txt", "D")
    _write(src / "sub" / "nested" / "deeper.txt", "DD")

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / "top.txt").read_text() == "T"
    assert (dst / "deep.txt").read_text() == "D"
    assert (dst / "deeper.txt").read_text() == "DD"
    # No subdirs should exist in the destination.
    assert not any(p.is_dir() for p in dst.iterdir())


def test_name_collision_uses_underscore_suffix(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a" / "file.pdf", "first")
    _write(src / "b" / "file.pdf", "second")
    _write(src / "c" / "file.pdf", "third")

    rc = main([str(src), str(dst)])
    assert rc == 0
    contents = sorted((p.name, p.read_text()) for p in dst.iterdir())
    names = [n for n, _ in contents]
    assert "file.pdf" in names
    assert "file_1.pdf" in names
    assert "file_2.pdf" in names
    # No file was overwritten — three distinct payloads survived.
    assert sorted(c for _, c in contents) == ["first", "second", "third"]


def test_collision_on_file_without_extension(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a" / "README", "first")
    _write(src / "b" / "README", "second")

    rc = main([str(src), str(dst)])
    assert rc == 0
    names = sorted(p.name for p in dst.iterdir())
    assert names == ["README", "README_1"]


def test_creates_target_if_missing(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    _write(src / "x.txt", "X")
    dst = tmp_path / "deep" / "new" / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / "x.txt").is_file()


def test_dotfiles_included(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / ".env", "secret")
    _write(src / "sub" / ".hidden", "h")

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / ".env").read_text() == "secret"
    assert (dst / ".hidden").read_text() == "h"


def test_symlinks_skipped(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    src.mkdir()
    real = _write(src / "real.txt", "R")
    link = src / "link.txt"
    os.symlink(real, link)
    dst = tmp_path / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / "real.txt").read_text() == "R"
    assert not (dst / "link.txt").exists()


def test_missing_source_exits_nonzero(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    rc = main([str(tmp_path / "no-such"), str(tmp_path / "dst")])
    assert rc != 0
    assert "not found" in capsys.readouterr().err.lower()


def test_source_is_file_exits_nonzero(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    f = tmp_path / "single.txt"
    f.write_text("hi")
    rc = main([str(f), str(tmp_path / "dst")])
    assert rc != 0


def test_target_inside_source_refused(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    _write(src / "a.txt", "A")
    dst = src / "out"

    rc = main([str(src), str(dst)])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "inside" in err or "within" in err


def test_empty_source_exits_zero(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "0" in captured.err or "0" in captured.out


def test_preserves_mtime(tmp_path: Path) -> None:
    """shutil.copy2 should preserve mtime — verify via a manually-set value."""
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    src.mkdir()
    f = _write(src / "x.txt", "X")
    target_ts = 1_700_000_000
    os.utime(f, (target_ts, target_ts))
    dst = tmp_path / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert int((dst / "x.txt").stat().st_mtime) == target_ts
