import lancedb
import numpy as np
import pyarrow as pa
import pytest

from server.store import VectorStore, TABLE_NAME, EMBEDDING_DIM

# Pre-Tier-2 schema (no mtime_ns / file_size) — used to test in-place migration.
# Uses the current EMBEDDING_DIM, not a historical one, so the migration test
# stays focused on the column-add path instead of also tripping the dim guard.
OLD_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("file_name", pa.string()),
    pa.field("file_type", pa.string()),
    pa.field("folder_path", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("content_hash", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
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
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
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
    results = store.vector_search([0.0] * EMBEDDING_DIM, top_k=5)
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
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
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


def test_open_index_with_mismatched_vector_dim_raises(tmp_path):
    """The dim guard exists so a user upgrading the embedding model gets a
    clear 'delete and re-index' message instead of a cryptic Arrow error.
    Create a table whose vector field is intentionally a non-current dim and
    confirm opening it via VectorStore raises with both dims in the message."""
    db_path = str(tmp_path / "db")
    wrong_dim = 999
    assert wrong_dim != EMBEDDING_DIM
    wrong_schema = pa.schema(
        [f for f in OLD_SCHEMA if f.name != "vector"]
        + [pa.field("vector", pa.list_(pa.float32(), wrong_dim))]
    )
    lancedb.connect(db_path).create_table(TABLE_NAME, schema=wrong_schema)

    store = VectorStore(db_path)
    with pytest.raises(RuntimeError, match=f"{wrong_dim}-dim"):
        store._get_table()


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


# --- Spreadsheet description support ---


# Pre-spreadsheet-descriptions schema (has stat fields, lacks chunk_kind/sheet_name).
PRE_DESCRIPTIONS_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("file_name", pa.string()),
    pa.field("file_type", pa.string()),
    pa.field("folder_path", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("content_hash", pa.string()),
    pa.field("mtime_ns", pa.int64()),
    pa.field("file_size", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
])


def test_schema_has_description_fields():
    from server.store import SCHEMA
    assert "chunk_kind" in SCHEMA.names
    assert "sheet_name" in SCHEMA.names


def test_ensure_schema_adds_chunk_kind_and_sheet_name(tmp_path):
    """A pre-existing chunks table without the new columns gets them added;
    existing rows are preserved."""
    db_path = str(tmp_path / "db")
    db = lancedb.connect(db_path)
    table = db.create_table(TABLE_NAME, schema=PRE_DESCRIPTIONS_SCHEMA)
    table.add([{
        "id": "x_0", "text": "hi", "source_file": "a.txt",
        "file_name": "a.txt", "file_type": ".txt", "folder_path": ".",
        "chunk_index": 0, "content_hash": "deadbeef",
        "mtime_ns": 0, "file_size": 2,
        "vector": [0.0] * EMBEDDING_DIM,
    }])

    store = VectorStore(db_path)
    store.ensure_schema()

    names = set(store._get_table().schema.names)
    assert "chunk_kind" in names
    assert "sheet_name" in names
    rows = store._get_table().search().limit(10).to_list()
    assert any(r["id"] == "x_0" for r in rows)


def test_add_chunks_persists_description_fields(store):
    """add_chunks writes chunk_kind/sheet_name through to the table; missing
    values default to chunk_kind='text' and sheet_name=null."""
    chunks = _make_chunks(["plain text"])
    desc_chunk = _make_chunks(["a sales workbook"], source_file="/f/biz.xlsx")[0]
    desc_chunk["id"] = "desc_0"
    desc_chunk["chunk_kind"] = "sheet_description"
    desc_chunk["sheet_name"] = "Sales"
    store.add_chunks(chunks + [desc_chunk])
    rows = (
        store._get_table().search()
        .select(["id", "chunk_kind", "sheet_name"])
        .limit(10).to_list()
    )
    rows_by_id = {r["id"]: r for r in rows}
    assert rows_by_id["chunk_0"]["chunk_kind"] == "text"
    assert rows_by_id["chunk_0"]["sheet_name"] is None
    assert rows_by_id["desc_0"]["chunk_kind"] == "sheet_description"
    assert rows_by_id["desc_0"]["sheet_name"] == "Sales"


# --- pending_descriptions table ---


def test_pending_descriptions_enqueue_list_remove(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.enqueue_pending(
        file_path="folder/a.xlsx",
        needs=["sheet:Sales", "sheet:Costs", "file"],
        preview_json='{"type":"xlsx"}',
        content_hash="hashA",
    )
    store.enqueue_pending(
        file_path="folder/b.csv",
        needs=["file"],
        preview_json='{"type":"csv"}',
        content_hash="hashB",
    )
    pending = store.list_pending(limit=10)
    assert len(pending) == 2
    paths = {p["file_path"] for p in pending}
    assert paths == {"folder/a.xlsx", "folder/b.csv"}

    store.remove_pending("folder/a.xlsx")
    after = store.list_pending(limit=10)
    assert {p["file_path"] for p in after} == {"folder/b.csv"}
    assert store.pending_count() == 1


def test_pending_descriptions_update_needs(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.enqueue_pending(
        "x.xlsx", ["sheet:A", "sheet:B", "file"], "{}", "h",
    )
    store.update_pending_needs("x.xlsx", ["sheet:B", "file"])
    [entry] = store.list_pending()
    assert entry["needs"] == ["sheet:B", "file"]


def test_pending_descriptions_get_entry(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    assert store.get_pending_entry("nope") is None
    store.enqueue_pending(
        "y.csv", ["file"], '{"type":"csv"}', "h",
    )
    entry = store.get_pending_entry("y.csv")
    assert entry["needs"] == ["file"]
    assert entry["content_hash"] == "h"


def test_pending_descriptions_enqueue_is_idempotent(tmp_path):
    """Re-enqueueing the same path replaces the existing entry."""
    store = VectorStore(str(tmp_path / "db"))
    store.enqueue_pending("x.csv", ["file"], '{}', "h1")
    store.enqueue_pending("x.csv", ["file"], '{}', "h2")
    assert store.pending_count() == 1
    assert store.get_pending_entry("x.csv")["content_hash"] == "h2"


def test_list_pending_filter_by_folder(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.enqueue_pending("a/x.csv", ["file"], "{}", "h")
    store.enqueue_pending("b/y.csv", ["file"], "{}", "h")
    res = store.list_pending(folder_path="a/", limit=10)
    assert {r["file_path"] for r in res} == {"a/x.csv"}


# --- dismissed_files + legacy eviction ---


def test_dismissal_lifecycle(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    assert not store.is_dismissed("x.csv", "hashA")
    store.dismiss("x.csv", "hashA")
    assert store.is_dismissed("x.csv", "hashA")
    # different hash = not dismissed (file content changed since dismissal)
    assert not store.is_dismissed("x.csv", "hashB")
    store.clear_dismissal("x.csv")
    assert not store.is_dismissed("x.csv", "hashA")


def test_dismissal_replaces_existing(tmp_path):
    """Re-dismissing the same path replaces the prior hash."""
    store = VectorStore(str(tmp_path / "db"))
    store.dismiss("x.csv", "h1")
    store.dismiss("x.csv", "h2")
    assert not store.is_dismissed("x.csv", "h1")
    assert store.is_dismissed("x.csv", "h2")


def test_evict_legacy_spreadsheet_chunks(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    # Seed: 2 CSV legacy text chunks, 1 PDF text chunk, 1 CSV description chunk.
    base = _make_chunks(
        ["row data"], source_file="a.csv", file_hash="h", mtime_ns=0, file_size=1,
    )[0]
    base["file_type"] = ".csv"
    base["file_name"] = "a.csv"

    second = _make_chunks(
        ["more rows"], source_file="b.csv", file_hash="h", mtime_ns=0, file_size=1,
    )[0]
    second["id"] = "csv_1"
    second["file_type"] = ".csv"
    second["file_name"] = "b.csv"

    pdf = _make_chunks(
        ["pdf content"], source_file="c.pdf", file_hash="h", mtime_ns=0, file_size=1,
    )[0]
    pdf["id"] = "pdf_0"
    pdf["file_type"] = ".pdf"
    pdf["file_name"] = "c.pdf"

    csv_desc = _make_chunks(
        ["a sales export"], source_file="d.csv", file_hash="h",
        mtime_ns=0, file_size=1,
    )[0]
    csv_desc["id"] = "csv_desc"
    csv_desc["file_type"] = ".csv"
    csv_desc["file_name"] = "d.csv"
    csv_desc["chunk_kind"] = "file_description"

    store.add_chunks([base, second, pdf, csv_desc])
    evicted = store.evict_legacy_spreadsheet_chunks()
    assert evicted == {"a.csv", "b.csv"}
    remaining_files = set(store.get_all_files())
    assert remaining_files == {"c.pdf", "d.csv"}


def test_evict_legacy_spreadsheet_chunks_empty_store(tmp_path):
    """Eviction on a fresh DB without the chunks table is a no-op."""
    store = VectorStore(str(tmp_path / "db"))
    assert store.evict_legacy_spreadsheet_chunks() == set()
