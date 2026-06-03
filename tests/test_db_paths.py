import os

from server.db_paths import resolve_db_dirs, resolve_write_dir


def test_explicit_wins_single(monkeypatch, tmp_path):
    monkeypatch.setenv("LANCEDB_PATHS", str(tmp_path / "a") + os.pathsep + str(tmp_path / "b"))
    assert resolve_db_dirs(str(tmp_path / "x")) == [os.path.abspath(str(tmp_path / "x"))]


def test_paths_list_split_and_dedup(monkeypatch, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    monkeypatch.setenv("LANCEDB_PATHS", os.pathsep.join([a, b, a, ""]))
    got = resolve_db_dirs(None)
    assert got == [os.path.abspath(a), os.path.abspath(b)]


def test_singular_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("LANCEDB_PATHS", raising=False)
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "solo"))
    assert resolve_db_dirs(None) == [os.path.abspath(str(tmp_path / "solo"))]


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv("LANCEDB_PATHS", raising=False)
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    assert resolve_db_dirs(None) == [os.path.abspath("./lancedb")]


def test_write_dir_first_of_paths(monkeypatch, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    monkeypatch.setenv("LANCEDB_PATHS", os.pathsep.join([a, b]))
    assert resolve_write_dir(None) == os.path.abspath(a)


def test_write_dir_explicit_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("LANCEDB_PATHS", str(tmp_path / "a"))
    assert resolve_write_dir(str(tmp_path / "z")) == os.path.abspath(str(tmp_path / "z"))
