import os
from unittest.mock import patch

import numpy as np
import pytest

from server.search import semantic_search
from server.store import VectorStore, EMBEDDING_DIM
from server.paths import to_absolute


def _fake_embed(texts):
    results = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2**32)
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        results.append(vec)
    return np.array(results)


def _seed_store(store: VectorStore, texts: list[str], source_file: str = "corpus/doc.txt"):
    """Seed the store directly. source_file is a path relative to the index
    directory, matching what the indexer now writes."""
    file_name = os.path.basename(source_file)
    folder_path = os.path.dirname(source_file) or "."
    chunks = []
    for i, text in enumerate(texts):
        vec = _fake_embed([text])[0]
        chunks.append({
            "id": f"chunk_{hash(text) % 10000}_{i}",
            "text": text,
            "source_file": source_file,
            "file_name": file_name,
            "file_type": ".txt",
            "folder_path": folder_path,
            "chunk_index": i,
            "content_hash": "fakehash",
            "mtime_ns": 0,
            "file_size": 0,
            "vector": vec.tolist(),
        })
    store.add_chunks(chunks)


@patch("server.search.get_model")
def test_semantic_search_returns_results(mock_get_model, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    store = VectorStore(db_path)
    _seed_store(store, [
        "revenue grew 23% in Q3",
        "python is a programming language",
        "the weather is nice today",
    ])

    result = semantic_search("Q3 revenue growth", db_path=db_path, top_k=2)
    assert result["total_results"] <= 2
    assert len(result["results"]) > 0
    assert "text" in result["results"][0]
    assert "source_file" in result["results"][0]
    assert "score" in result["results"][0]


@patch("server.search.get_model")
def test_semantic_search_empty_store(mock_get_model, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    result = semantic_search("anything", db_path=db_path)
    assert result["results"] == []
    assert result["total_results"] == 0


@patch("server.search.get_model")
def test_semantic_search_with_folder_filter(mock_get_model, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    store = VectorStore(db_path)
    _seed_store(store, ["doc in folder a"], source_file="folder_a/doc.txt")
    _seed_store(store, ["doc in folder b"], source_file="folder_b/doc.txt")

    # The caller passes an absolute folder path; search converts it to the
    # stored relative form internally.
    result = semantic_search(
        "doc", db_path=db_path, folder_path=to_absolute("folder_a", db_path)
    )
    assert len(result["results"]) >= 1
    for r in result["results"]:
        assert "folder_a" in r["source_file"]
        assert "folder_b" not in r["source_file"]


@patch("server.search.get_model")
def test_semantic_search_folder_filter_cross_volume(mock_get_model, tmp_path):
    """Folder filter works when corpus and index are on different volumes.
    Cross-volume indexing stores absolute paths in the folder_path column, so
    the search-time to_relative() must produce the matching absolute form
    (not a relpath) for the exact-match WHERE clause to hit any rows."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "lancedb")
    store = VectorStore(db_path)

    # Simulate a cross-volume corpus: real on-disk paths whose st_dev will be
    # mocked to differ from the index's volume.
    abs_folder = str(tmp_path / "corpus" / "folder_a")
    os.makedirs(abs_folder)
    abs_file = os.path.join(abs_folder, "doc.txt")
    with open(abs_file, "w") as fh:
        fh.write("placeholder")

    # Seed the store with the absolute paths a cross-volume indexer would write.
    _seed_store(store, ["doc in folder a"], source_file=abs_file)

    # Make the corpus and the index appear on different volumes. The patch
    # targets os.stat globally (pathlib + lancedb both call it), so the fake
    # delegates to the real stat and only overrides st_dev.
    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        s = real_stat(path, *args, **kwargs)
        new_dev = 1 if "lancedb" in str(path) else 2
        return os.stat_result((
            s.st_mode, s.st_ino, new_dev, s.st_nlink, s.st_uid, s.st_gid,
            s.st_size, s.st_atime, s.st_mtime, s.st_ctime,
        ))

    with patch("server.paths.os.stat", side_effect=fake_stat):
        result = semantic_search("doc", db_path=db_path, folder_path=abs_folder)

    assert len(result["results"]) >= 1
    assert result["results"][0]["source_file"] == abs_file


@patch("server.search.get_model")
def test_semantic_search_hybrid_mode(mock_get_model, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    store = VectorStore(db_path)
    _seed_store(store, [
        "revenue grew 23% in Q3",
        "python is a programming language",
        "the weather is nice today",
    ])
    store.create_fts_index()  # built at index time, mirroring the indexer

    result = semantic_search("revenue Q3", db_path=db_path, top_k=2, mode="hybrid")
    assert result["total_results"] >= 1
    assert result["mode"] == "hybrid"
    assert "score" in result["results"][0]


@patch("server.search.get_model")
def test_hybrid_search_does_not_rebuild_fts_index(mock_get_model, tmp_path):
    """Hybrid search no longer rebuilds the FTS index per query — it is built
    once at index time (4.7)."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    store = VectorStore(db_path)
    _seed_store(store, ["revenue grew 23% in Q3", "python", "the weather"])
    store.create_fts_index()

    with patch.object(VectorStore, "create_fts_index") as fts_spy:
        semantic_search("revenue", db_path=db_path, mode="hybrid")
    fts_spy.assert_not_called()
