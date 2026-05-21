from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from server.indexer import discover_files, compute_file_hash, index_folder, EXCLUDE_PATTERNS
from server.store import VectorStore


@pytest.fixture
def docs_dir(tmp_path):
    (tmp_path / "readme.md").write_text("# Project Alpha\n\nRevenue reporting.")
    (tmp_path / "notes.txt").write_text("Meeting notes: Q3 revenue grew 23%")
    (tmp_path / "bericht.md").write_text("# Quartalsbericht\n\nDer Umsatz stieg um 23%.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("Nested file content.")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    return tmp_path


def test_discover_files_finds_supported(docs_dir):
    files = discover_files(docs_dir, file_types=None, recursive=True)
    names = {f.name for f in files}
    assert "readme.md" in names
    assert "notes.txt" in names
    assert "deep.txt" in names
    assert "image.png" not in names


def test_discover_files_excludes_patterns(docs_dir):
    files = discover_files(docs_dir, file_types=None, recursive=True)
    names = {f.name for f in files}
    assert ".DS_Store" not in names


def test_discover_files_non_recursive(docs_dir):
    files = discover_files(docs_dir, file_types=None, recursive=False)
    names = {f.name for f in files}
    assert "readme.md" in names
    assert "deep.txt" not in names


def test_discover_files_filter_by_type(docs_dir):
    files = discover_files(docs_dir, file_types={".md"}, recursive=True)
    assert all(f.suffix == ".md" for f in files)
    assert len(files) == 2  # readme.md + bericht.md


def test_compute_file_hash_deterministic(docs_dir):
    h1 = compute_file_hash(docs_dir / "notes.txt")
    h2 = compute_file_hash(docs_dir / "notes.txt")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex length


def test_compute_file_hash_changes_on_modification(docs_dir):
    f = docs_dir / "notes.txt"
    h1 = compute_file_hash(f)
    f.write_text("Modified content")
    h2 = compute_file_hash(f)
    assert h1 != h2


def _fake_embed(texts):
    """Deterministic fake embeddings for testing."""
    results = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2**32)
        vec = rng.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        results.append(vec)
    return np.array(results)


@patch("server.indexer.get_model")
def test_index_folder_full_pipeline(mock_get_model, docs_dir, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    result = index_folder(str(docs_dir), db_path=db_path)

    assert result["status"] == "completed"
    assert result["files_indexed"] == 4  # readme.md, notes.txt, bericht.md, sub/deep.txt
    assert result["files_skipped"] == 0
    assert result["files_failed"] == 0
    assert result["total_chunks"] > 0


@patch("server.indexer.get_model")
def test_index_folder_incremental(mock_get_model, docs_dir, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")

    # First run indexes everything
    r1 = index_folder(str(docs_dir), db_path=db_path)
    assert r1["files_indexed"] == 4

    # Second run skips everything (no changes)
    r2 = index_folder(str(docs_dir), db_path=db_path)
    assert r2["files_indexed"] == 0
    assert r2["files_skipped"] == 4


@patch("server.indexer.get_model")
def test_index_folder_detects_deleted_files(mock_get_model, docs_dir, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    index_folder(str(docs_dir), db_path=db_path)

    # Delete a file
    (docs_dir / "notes.txt").unlink()
    r2 = index_folder(str(docs_dir), db_path=db_path)
    assert r2["files_deleted"] == 1


@patch("server.indexer.get_model")
def test_indexing_second_folder_preserves_first(mock_get_model, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    folder_a = tmp_path / "folder_a"
    folder_a.mkdir()
    folder_b = tmp_path / "folder_b"
    folder_b.mkdir()
    (folder_a / "a.txt").write_text("Alpha content about revenue.")
    (folder_b / "b.txt").write_text("Beta content about expenses.")

    index_folder(str(folder_a), db_path=db_path)
    index_folder(str(folder_b), db_path=db_path)  # must NOT wipe folder A

    files = VectorStore(db_path).get_all_files()
    assert any("a.txt" in f for f in files)
    assert any("b.txt" in f for f in files)


@patch("server.indexer.get_model")
def test_index_folder_nonexistent_raises(mock_get_model, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    with pytest.raises(FileNotFoundError):
        index_folder("/nonexistent/folder", db_path=str(tmp_path / "testdb"))
