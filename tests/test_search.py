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


# --- index records its model; search loads it (env ignored) ---------------

from server.index_meta import write_meta as _write_meta
from server.embedding_models import resolve_profile as _resolve_profile


def _fake_embed_384(texts):
    out = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2**32)
        v = rng.randn(384).astype(np.float32)
        v = v / np.linalg.norm(v)
        out.append(v)
    return np.array(out)


def test_search_loads_model_from_meta_ignoring_env(tmp_path, monkeypatch):
    db = str(tmp_path / "idx")
    profile, dim = _resolve_profile("minilm", None)  # 384-dim index
    _write_meta(db, profile, dim)

    store = VectorStore(db, dim=384)
    # Seed 384-dim rows so they match the recorded index width.
    chunks = []
    for i, text in enumerate(["alpha doc", "beta doc"]):
        vec = _fake_embed_384([text])[0]
        chunks.append({
            "id": f"c_{i}", "text": text, "source_file": "corpus/d.txt",
            "file_name": "d.txt", "file_type": ".txt", "folder_path": "corpus",
            "chunk_index": i, "content_hash": "h", "mtime_ns": 0, "file_size": 0,
            "vector": vec.tolist(),
        })
    store.add_chunks(chunks)

    captured = {}

    def fake_get_model(p, d):
        captured["alias"] = p.alias
        captured["dim"] = d
        return type("M", (), {"encode": lambda self, texts, **kw: _fake_embed_384(texts)})()

    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-0.6b")  # must be ignored
    with patch("server.search.get_model", fake_get_model):
        out = semantic_search("alpha", db_path=db)

    assert captured["alias"] == "minilm"
    assert captured["dim"] == 384
    assert out["total_results"] >= 1


# --- multi-index: search_one provenance tagging ---------------------------

from server.search import search_one


def test_search_one_tags_results_with_origin(tmp_path, monkeypatch):
    from server.index_meta import write_meta as _wm
    from server.embedding_models import resolve_profile as _rp
    db = str(tmp_path / "idx")
    profile, dim = _rp("minilm", None)  # 384
    _wm(db, profile, dim)
    store = VectorStore(db, dim=384)
    chunks = []
    for i, text in enumerate(["alpha", "beta"]):
        vec = _fake_embed_384([text])[0]
        chunks.append({
            "id": f"c_{i}", "text": text, "source_file": "corpus/d.txt",
            "file_name": "d.txt", "file_type": ".txt", "folder_path": "corpus",
            "chunk_index": i, "content_hash": "h", "mtime_ns": 0, "file_size": 0,
            "vector": vec.tolist(),
        })
    store.add_chunks(chunks)

    def fake_get_model(p, d):
        return type("M", (), {"encode": lambda self, t, **k: _fake_embed_384(t)})()

    with patch("server.search.get_model", fake_get_model):
        hits = search_one(db, "alpha", top_k=10, folder_path=None,
                          file_type=None, mode="vector")
    assert hits, "expected at least one hit"
    assert all(h["index_path"] == os.path.abspath(db) for h in hits)
    assert all(h["model_alias"] == "minilm" for h in hits)


# --- multi-index: semantic_search fan-out ---------------------------------

def test_semantic_search_fans_out_over_two_indexes(tmp_path, monkeypatch):
    from server.index_meta import write_meta as _wm
    from server.embedding_models import resolve_profile as _rp

    def seed(db, texts):
        profile, dim = _rp("minilm", None)
        _wm(db, profile, dim)
        s = VectorStore(db, dim=384)
        rows = []
        for i, t in enumerate(texts):
            v = _fake_embed_384([t])[0]
            rows.append({
                "id": f"{db}_{i}", "text": t, "source_file": f"corpus/{i}.txt",
                "file_name": f"{i}.txt", "file_type": ".txt", "folder_path": "corpus",
                "chunk_index": 0, "content_hash": "h", "mtime_ns": 0, "file_size": 0,
                "vector": v.tolist(),
            })
        s.add_chunks(rows)

    a, b = str(tmp_path / "A"), str(tmp_path / "B")
    seed(a, ["alpha one"])
    seed(b, ["alpha two"])
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    monkeypatch.setenv("LANCEDB_PATHS", f"{a}{os.pathsep}{b}")

    def fake_get_model(p, d):
        return type("M", (), {"encode": lambda self, t, **k: _fake_embed_384(t)})()

    with patch("server.search.get_model", fake_get_model):
        out = semantic_search("alpha", db_path=None)

    assert set(out["indexes_searched"]) == {os.path.abspath(a), os.path.abspath(b)}
    origins = {r["index_path"] for r in out["results"]}
    assert origins == {os.path.abspath(a), os.path.abspath(b)}


def test_semantic_search_skips_missing_index(tmp_path, monkeypatch):
    from server.index_meta import write_meta as _wm
    from server.embedding_models import resolve_profile as _rp
    good = str(tmp_path / "good")
    profile, dim = _rp("minilm", None)
    _wm(good, profile, dim)
    s = VectorStore(good, dim=384)
    v = _fake_embed_384(["hello"])[0]
    s.add_chunks([{
        "id": "g0", "text": "hello", "source_file": "c/h.txt", "file_name": "h.txt",
        "file_type": ".txt", "folder_path": "c", "chunk_index": 0, "content_hash": "h",
        "mtime_ns": 0, "file_size": 0, "vector": v.tolist(),
    }])
    # A regular file (not a dir) makes lancedb.connect raise -> the index is
    # skipped, not fatal.
    broken = str(tmp_path / "broken")
    (tmp_path / "broken").write_text("not a lancedb")
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    monkeypatch.setenv("LANCEDB_PATHS", f"{good}{os.pathsep}{broken}")

    def fake_get_model(p, d):
        return type("M", (), {"encode": lambda self, t, **k: _fake_embed_384(t)})()

    with patch("server.search.get_model", fake_get_model):
        out = semantic_search("hello", db_path=None)
    assert out["total_results"] >= 1
    assert any(s_["index"] == os.path.abspath(broken) for s_ in out["skipped"])
