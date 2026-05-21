import lancedb
import numpy as np
import pyarrow as pa
import pytest

from server.store import VectorStore, TABLE_NAME

# Pre-Tier-2 schema (no mtime_ns / file_size) — used to test in-place migration.
OLD_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("file_name", pa.string()),
    pa.field("file_type", pa.string()),
    pa.field("folder_path", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("content_hash", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384)),
])


@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "testdb"))


def _make_chunks(
    texts: list[str],
    source_file: str = "/fake/doc.txt",
    file_hash: str = "abc123",
    mtime_ns: int = 1000,
    file_size: int = 500,
):
    """Helper to create chunk dicts with random embeddings."""
    chunks = []
    for i, text in enumerate(texts):
        rng = np.random.RandomState(hash(text) % 2**32)
        vec = rng.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        chunks.append({
            "id": f"chunk_{i}",
            "text": text,
            "source_file": source_file,
            "file_name": source_file.split("/")[-1],
            "file_type": ".txt",
            "folder_path": "/fake",
            "chunk_index": i,
            "content_hash": file_hash,
            "mtime_ns": mtime_ns,
            "file_size": file_size,
            "vector": vec.tolist(),
        })
    return chunks


def test_add_and_count(store):
    chunks = _make_chunks(["hello world", "foo bar"])
    store.add_chunks(chunks)
    assert store.count_chunks() == 2


def test_get_file_hash(store):
    chunks = _make_chunks(["text"], source_file="/fake/a.txt", file_hash="hash_abc")
    store.add_chunks(chunks)
    assert store.get_file_hash("/fake/a.txt") == "hash_abc"


def test_get_file_hash_missing(store):
    assert store.get_file_hash("/nonexistent.txt") is None


def test_delete_by_file(store):
    chunks_a = _make_chunks(["chunk a"], source_file="/fake/a.txt")
    chunks_b = _make_chunks(["chunk b"], source_file="/fake/b.txt")
    # Give unique IDs
    chunks_b[0]["id"] = "chunk_b_0"
    store.add_chunks(chunks_a)
    store.add_chunks(chunks_b)
    assert store.count_chunks() == 2

    store.delete_by_file("/fake/a.txt")
    assert store.count_chunks() == 1
    assert store.get_file_hash("/fake/a.txt") is None
    assert store.get_file_hash("/fake/b.txt") is not None


def test_get_all_files(store):
    chunks_a = _make_chunks(["a"], source_file="/fake/a.txt")
    chunks_b = _make_chunks(["b"], source_file="/fake/b.txt")
    chunks_b[0]["id"] = "chunk_b_0"
    store.add_chunks(chunks_a)
    store.add_chunks(chunks_b)
    files = store.get_all_files()
    assert set(files) == {"/fake/a.txt", "/fake/b.txt"}


def test_vector_search(store):
    chunks = _make_chunks(
        ["the cat sat on the mat", "revenue grew 23%", "python programming"],
        source_file="/fake/doc.txt",
    )
    store.add_chunks(chunks)

    # Search with the vector of the first chunk — should return it as top result
    results = store.vector_search(chunks[0]["vector"], top_k=2)
    assert len(results) <= 2
    assert results[0]["text"] == "the cat sat on the mat"


def test_vector_search_empty_store(store):
    results = store.vector_search([0.0] * 384, top_k=5)
    assert results == []


def test_vector_search_with_folder_filter(store):
    chunks_a = _make_chunks(["hello from folder a"], source_file="/folder_a/doc.txt")
    chunks_a[0]["folder_path"] = "/folder_a"
    chunks_b = _make_chunks(["hello from folder b"], source_file="/folder_b/doc.txt")
    chunks_b[0]["id"] = "chunk_b_0"
    chunks_b[0]["folder_path"] = "/folder_b"

    store.add_chunks(chunks_a)
    store.add_chunks(chunks_b)

    results = store.vector_search(
        chunks_a[0]["vector"], top_k=10, folder_path="/folder_a"
    )
    assert len(results) == 1
    assert results[0]["folder_path"] == "/folder_a"


def test_apostrophe_in_path_is_query_safe(store):
    """Single quotes in paths must not break where/delete clauses."""
    tricky = "/fake/Bob's notes.txt"
    chunks = _make_chunks(["report content"], source_file=tricky, file_hash="hash_q")
    chunks[0]["folder_path"] = "/fake/Bob's docs"
    store.add_chunks(chunks)

    # get_file_hash builds a where-clause from the path
    assert store.get_file_hash(tricky) == "hash_q"

    # vector_search folder filter builds a where-clause from folder_path
    results = store.vector_search(
        chunks[0]["vector"], top_k=10, folder_path="/fake/Bob's docs"
    )
    assert len(results) == 1
    assert results[0]["source_file"] == tricky

    # delete_by_file builds a delete-clause from the path
    store.delete_by_file(tricky)
    assert store.count_chunks() == 0


def test_fts_search(store):
    chunks = _make_chunks(
        ["revenue grew 23% in Q3", "the cat sat on the mat", "python programming"],
    )
    store.add_chunks(chunks)
    store.create_fts_index()
    results = store.fts_search("revenue Q3", top_k=2)
    assert len(results) >= 1
    assert "revenue" in results[0]["text"]


def test_fts_search_empty_store(store):
    results = store.fts_search("anything", top_k=5)
    assert results == []


def test_hybrid_search(store):
    chunks = _make_chunks(
        ["revenue grew 23% in Q3", "the cat sat on the mat", "python programming"],
    )
    store.add_chunks(chunks)
    store.create_fts_index()
    # Use the vector of the first chunk and a matching text query
    results = store.hybrid_search(
        query_text="revenue Q3",
        query_vector=chunks[0]["vector"],
        top_k=2,
    )
    assert len(results) >= 1
    assert "revenue" in results[0]["text"]
    assert "rrf_score" in results[0]


def _random_chunks(n: int) -> list[dict]:
    """n chunk dicts with unique ids and random unit vectors — no embedding model."""
    chunks = []
    for i in range(n):
        rng = np.random.RandomState(i)
        vec = rng.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        chunks.append({
            "id": f"c_{i}",
            "text": f"document number {i}",
            "source_file": f"/fake/doc_{i}.txt",
            "file_name": f"doc_{i}.txt",
            "file_type": ".txt",
            "folder_path": "/fake",
            "chunk_index": 0,
            "content_hash": "h",
            "mtime_ns": i,
            "file_size": i,
            "vector": vec.tolist(),
        })
    return chunks


def test_create_vector_index_noop_on_empty_store(store):
    """No table yet — create_vector_index must not raise."""
    store.create_vector_index()
    assert store.count_chunks() == 0


def test_create_vector_index_noop_below_threshold(store, monkeypatch):
    """Below VECTOR_INDEX_MIN_ROWS rows, no ANN index is built (flat scan is fine)."""
    monkeypatch.setattr("server.store.VECTOR_INDEX_MIN_ROWS", 1000)
    store.add_chunks(_random_chunks(20))
    store.create_vector_index()
    assert store._get_table().list_indices() == []


def test_create_vector_index_builds_above_threshold(store, monkeypatch):
    """At or above the threshold, an IVF_PQ index is built on the vector column."""
    monkeypatch.setattr("server.store.VECTOR_INDEX_MIN_ROWS", 256)
    store.add_chunks(_random_chunks(600))
    store.create_vector_index()
    indices = store._get_table().list_indices()
    assert len(indices) == 1


def test_create_vector_index_refresh_is_idempotent(store, monkeypatch):
    """Re-running create_vector_index replaces the index rather than erroring."""
    monkeypatch.setattr("server.store.VECTOR_INDEX_MIN_ROWS", 256)
    store.add_chunks(_random_chunks(600))
    store.create_vector_index()
    store.create_vector_index()
    assert len(store._get_table().list_indices()) == 1


# --- Tier 2: schema, bulk reads, stat refresh, compaction ---


def test_schema_has_stat_fields():
    """The schema carries mtime_ns and file_size for the cheap change-check."""
    from server.store import SCHEMA
    assert "mtime_ns" in SCHEMA.names
    assert "file_size" in SCHEMA.names


def test_ensure_schema_migrates_pre_tier2_table(tmp_path):
    """ensure_schema adds the new columns to a table created without them."""
    db_path = str(tmp_path / "db")
    db = lancedb.connect(db_path)
    db.create_table(TABLE_NAME, schema=OLD_SCHEMA)  # pre-Tier-2 index

    store = VectorStore(db_path)
    store.ensure_schema()

    names = store._get_table().schema.names
    assert "mtime_ns" in names
    assert "file_size" in names


def test_ensure_schema_noop_when_columns_present(store):
    """ensure_schema is a harmless no-op on an already-current table."""
    store.add_chunks(_make_chunks(["x"]))
    store.ensure_schema()
    assert "mtime_ns" in store._get_table().schema.names


def test_ensure_schema_no_table_is_safe(store):
    """ensure_schema on a fresh DB with no table must not raise."""
    store.ensure_schema()
    assert store.count_chunks() == 0


def test_get_file_index_returns_stat_and_hash(store):
    store.add_chunks(_make_chunks(
        ["a", "b"], source_file="/f/x.txt",
        file_hash="h1", mtime_ns=111, file_size=222,
    ))
    idx = store.get_file_index()
    assert idx == {
        "/f/x.txt": {"content_hash": "h1", "mtime_ns": 111, "file_size": 222},
    }


def test_get_file_index_empty_store(store):
    assert store.get_file_index() == {}


def test_get_file_index_multiple_files(store):
    a = _make_chunks(["a"], source_file="/f/a.txt", file_hash="ha", mtime_ns=1, file_size=10)
    b = _make_chunks(["b"], source_file="/f/b.txt", file_hash="hb", mtime_ns=2, file_size=20)
    b[0]["id"] = "chunk_b_0"
    store.add_chunks(a)
    store.add_chunks(b)
    idx = store.get_file_index()
    assert set(idx) == {"/f/a.txt", "/f/b.txt"}
    assert idx["/f/b.txt"] == {"content_hash": "hb", "mtime_ns": 2, "file_size": 20}


def test_update_file_stat(store):
    store.add_chunks(_make_chunks(
        ["a", "b"], source_file="/f/x.txt", mtime_ns=1, file_size=1,
    ))
    store.update_file_stat("/f/x.txt", mtime_ns=999, file_size=888)
    idx = store.get_file_index()
    assert idx["/f/x.txt"]["mtime_ns"] == 999
    assert idx["/f/x.txt"]["file_size"] == 888


def test_optimize_table_runs(store):
    store.add_chunks(_make_chunks(["a", "b"]))
    store.optimize_table()
    assert store.count_chunks() == 2


def test_optimize_table_empty_store_is_safe(store):
    store.optimize_table()
    assert store.count_chunks() == 0


def test_add_chunks_persists_stat_fields(store):
    """add_chunks writes mtime_ns/file_size through to the table."""
    store.add_chunks(_make_chunks(["a"], mtime_ns=12345, file_size=678))
    rows = (
        store._get_table()
        .search()
        .select(["mtime_ns", "file_size"])
        .limit(1)
        .to_list()
    )
    assert rows[0]["mtime_ns"] == 12345
    assert rows[0]["file_size"] == 678
