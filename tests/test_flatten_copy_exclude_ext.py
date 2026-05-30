"""Tests for the --exclude-ext filter in scripts/flatten_copy.py."""

from __future__ import annotations

from pathlib import Path


def _write(p: Path, content: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _files_only(d: Path) -> set[str]:
    """Names of regular files directly under ``d`` (ignores _runlogs/)."""
    return {p.name for p in d.iterdir() if p.is_file()}


def test_excludes_named_extension(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "b.ics", "B")
    _write(src / "sub" / "c.pdf", "C")

    rc = main([str(src), str(dst), "--exclude-ext", ".ics"])
    assert rc == 0
    assert _files_only(dst) == {"a.txt", "c.pdf"}
    assert not (dst / "b.ics").exists()


def test_exclude_is_case_insensitive(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "keep.txt", "K")
    _write(src / "upper.ICS", "U")
    _write(src / "sub" / "mixed.Ics", "M")

    rc = main([str(src), str(dst), "--exclude-ext", ".ics"])
    assert rc == 0
    assert _files_only(dst) == {"keep.txt"}


def test_exclude_ext_without_leading_dot(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "b.ics", "B")

    rc = main([str(src), str(dst), "--exclude-ext", "ics"])
    assert rc == 0
    assert _files_only(dst) == {"a.txt"}


def test_comma_separated_excludes_multiple(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "b.ics", "B")
    _write(src / "c.tmp", "C")

    rc = main([str(src), str(dst), "--exclude-ext", ".ics,.tmp"])
    assert rc == 0
    assert _files_only(dst) == {"a.txt"}


def test_repeated_flag_excludes_multiple(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "b.ics", "B")
    _write(src / "c.tmp", "C")

    rc = main(
        [str(src), str(dst), "--exclude-ext", ".ics", "--exclude-ext", ".tmp"]
    )
    assert rc == 0
    assert _files_only(dst) == {"a.txt"}


def test_no_exclude_copies_everything(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "b.ics", "B")

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert _files_only(dst) == {"a.txt", "b.ics"}


def test_excluded_files_not_counted_in_header(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "b.ics", "B")
    _write(src / "c.ics", "C")

    rc = main([str(src), str(dst), "--exclude-ext", ".ics"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "Found 1 regular file(s)" in err, err
    assert "Excluding extensions: .ics" in err, err


def test_normalize_exts_helper() -> None:
    from scripts.flatten_copy import _normalize_exts

    assert _normalize_exts(None) == set()
    assert _normalize_exts([".ics"]) == {".ics"}
    assert _normalize_exts(["ICS"]) == {".ics"}
    assert _normalize_exts([".ics,.tmp"]) == {".ics", ".tmp"}
    assert _normalize_exts([".ics", "tmp"]) == {".ics", ".tmp"}
    assert _normalize_exts([" , .ICS , "]) == {".ics"}
