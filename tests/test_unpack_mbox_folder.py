"""Tests for scripts/unpack_mbox_folder.py — batch mbox unpacker."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_mbox_messages import _make_entry, _make_mbox


def _simple_mbox(path: Path) -> Path:
    """Write a minimal one-message mbox at ``path``."""
    return _make_mbox(
        path.parent,
        _make_entry(
            {"From": "a@x", "Subject": "one", "Message-ID": f"<{path.stem}@x>"},
            "body",
        ),
        name=path.name,
    )


def test_unpacks_every_mbox_in_input_folder(tmp_path: Path) -> None:
    from scripts.unpack_mbox_folder import main

    src = tmp_path / "in"
    src.mkdir()
    _simple_mbox(src / "alpha.mbox")
    _simple_mbox(src / "beta.mbox")
    out = tmp_path / "out"

    rc = main([str(src), str(out)])
    assert rc == 0
    assert (out / "alpha").is_dir()
    assert (out / "beta").is_dir()
    assert any((out / "alpha").rglob("msg-*.txt"))
    assert any((out / "beta").rglob("msg-*.txt"))


def test_recursive_discovery_into_subdirs(tmp_path: Path) -> None:
    from scripts.unpack_mbox_folder import main

    src = tmp_path / "in"
    nested = src / "2024" / "q1"
    nested.mkdir(parents=True)
    _simple_mbox(nested / "inbox.mbox")
    out = tmp_path / "out"

    rc = main([str(src), str(out)])
    assert rc == 0
    assert (out / "inbox").is_dir()
    assert any((out / "inbox").rglob("msg-*.txt"))


def test_collision_on_same_stem_uses_suffix(tmp_path: Path) -> None:
    from scripts.unpack_mbox_folder import main

    src = tmp_path / "in"
    a_dir = src / "2024"
    b_dir = src / "2025"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    _simple_mbox(a_dir / "inbox.mbox")
    _simple_mbox(b_dir / "inbox.mbox")
    out = tmp_path / "out"

    rc = main([str(src), str(out)])
    assert rc == 0
    # First mbox wins the plain name; second gets -1.
    assert (out / "inbox").is_dir()
    assert (out / "inbox-1").is_dir()
    assert any((out / "inbox").rglob("msg-*.txt"))
    assert any((out / "inbox-1").rglob("msg-*.txt"))


def test_creates_output_folder_if_missing(tmp_path: Path) -> None:
    from scripts.unpack_mbox_folder import main

    src = tmp_path / "in"
    src.mkdir()
    _simple_mbox(src / "a.mbox")
    out = tmp_path / "does" / "not" / "exist"

    rc = main([str(src), str(out)])
    assert rc == 0
    assert out.is_dir()
    assert (out / "a").is_dir()


def test_missing_input_folder_exits_nonzero(tmp_path: Path, capsys) -> None:
    from scripts.unpack_mbox_folder import main

    rc = main([str(tmp_path / "no-such"), str(tmp_path / "out")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "does not exist" in err.lower()


def test_input_path_is_file_exits_nonzero(tmp_path: Path) -> None:
    from scripts.unpack_mbox_folder import main

    f = tmp_path / "input.txt"
    f.write_text("hi")
    rc = main([str(f), str(tmp_path / "out")])
    assert rc != 0


def test_target_inside_source_refused(tmp_path: Path, capsys) -> None:
    from scripts.unpack_mbox_folder import main

    src = tmp_path / "in"
    src.mkdir()
    _simple_mbox(src / "a.mbox")
    out = src / "out"

    rc = main([str(src), str(out)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "inside" in err.lower() or "within" in err.lower()


def test_empty_input_exits_zero(tmp_path: Path, capsys) -> None:
    from scripts.unpack_mbox_folder import main

    src = tmp_path / "in"
    src.mkdir()
    out = tmp_path / "out"

    rc = main([str(src), str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "0" in captured.err or "0" in captured.out


def test_case_insensitive_extension(tmp_path: Path) -> None:
    from scripts.unpack_mbox_folder import main

    src = tmp_path / "in"
    src.mkdir()
    _simple_mbox(src / "MIXED.MBOX")
    out = tmp_path / "out"

    rc = main([str(src), str(out)])
    assert rc == 0
    assert (out / "MIXED").is_dir()


def test_skips_directories_named_dot_mbox(tmp_path: Path) -> None:
    from scripts.unpack_mbox_folder import main

    src = tmp_path / "in"
    (src / "fake.mbox").mkdir(parents=True)  # dir, not file
    _simple_mbox(src / "real.mbox")
    out = tmp_path / "out"

    rc = main([str(src), str(out)])
    assert rc == 0
    assert (out / "real").is_dir()
    assert not (out / "fake").exists()


def test_per_file_failure_does_not_abort_batch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """If one mbox raises during unpack, the rest still get processed."""
    from scripts import unpack_mbox_folder

    src = tmp_path / "in"
    src.mkdir()
    _simple_mbox(src / "good1.mbox")
    _simple_mbox(src / "bad.mbox")
    _simple_mbox(src / "good2.mbox")
    out = tmp_path / "out"

    real = unpack_mbox_folder.ensure_unpacked

    def flaky(mbox_path, target=None):
        if Path(mbox_path).name == "bad.mbox":
            raise RuntimeError("kaboom")
        return real(mbox_path, target=target)

    monkeypatch.setattr(unpack_mbox_folder, "ensure_unpacked", flaky)

    rc = unpack_mbox_folder.main([str(src), str(out)])
    # Non-zero exit because at least one failed.
    assert rc != 0
    # But the two healthy mboxes were still processed.
    assert (out / "good1").is_dir()
    assert (out / "good2").is_dir()
    assert not (out / "bad").exists() or not any((out / "bad").rglob("msg-*.txt"))
    err = capsys.readouterr().err
    assert "bad.mbox" in err
    assert "kaboom" in err
