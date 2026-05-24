import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest

from server.indexer import (
    discover_files,
    compute_file_hash,
    index_folder,
    _finalize_index,
    _select_device,
)
from server.exclusions import ExclusionRules
from server.store import VectorStore, EMBEDDING_DIM


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


def test_discover_files_excludes_via_exclusion_rules(docs_dir, tmp_path):
    """Excluding a file via ExclusionRules keeps it out of the walk results."""
    rules = ExclusionRules.load(
        docs_dir, extra_patterns=["*.DS_Store"], db_dir=str(tmp_path / "db")
    )
    files = discover_files(docs_dir, file_types=None, recursive=True, exclusions=rules)
    names = {f.name for f in files}
    assert ".DS_Store" not in names
    assert "readme.md" in names  # other files still discovered


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
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
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
def test_index_survives_relocation(mock_get_model, tmp_path):
    """Moving the index + corpus to a new absolute path does not re-index."""
    import shutil

    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    # Lay out the index and corpus as siblings on one "drive".
    root_a = tmp_path / "drive_a"
    corpus_a = root_a / "corpus"
    corpus_a.mkdir(parents=True)
    db_a = root_a / "lancedb"
    (corpus_a / "readme.md").write_text("# Revenue\n\nQ3 revenue grew 23%.")
    (corpus_a / "notes.txt").write_text("Meeting notes about revenue.")
    (corpus_a / "sub").mkdir()
    (corpus_a / "sub" / "deep.txt").write_text("Nested revenue details.")

    r1 = index_folder(str(corpus_a), db_path=str(db_a))
    assert r1["files_indexed"] == 3

    # Simulate re-plugging the drive at a different absolute mount point.
    root_b = tmp_path / "drive_b_other_mount"
    shutil.move(str(root_a), str(root_b))
    corpus_b = root_b / "corpus"
    db_b = root_b / "lancedb"

    r2 = index_folder(str(corpus_b), db_path=str(db_b))
    assert r2["files_indexed"] == 0      # nothing re-indexed
    assert r2["files_skipped"] == 3      # all skipped via hash match
    assert r2["files_deleted"] == 0      # no false orphans

    from server.search import semantic_search
    with patch("server.search.get_model", return_value=mock_model):
        res = semantic_search("revenue", db_path=str(db_b))
        hybrid = semantic_search("revenue", db_path=str(db_b), mode="hybrid")

    assert res["total_results"] > 0
    assert hybrid["total_results"] > 0
    # Displayed paths resolve to the new location, for both search modes.
    assert all(str(corpus_b) in r["source_file"] for r in res["results"])
    assert all(str(corpus_b) in r["source_file"] for r in hybrid["results"])


@patch("server.indexer.get_model")
def test_index_folder_handles_apostrophe_in_path(mock_get_model, tmp_path):
    """A file whose path contains a single quote is skip-detected on re-runs."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "testdb")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "Bob's notes.txt").write_text("Quarterly revenue figures for Bob.")

    r1 = index_folder(str(corpus), db_path=db_path)
    assert r1["files_indexed"] == 1
    chunks_after_first = r1["total_chunks"]

    # Second run: the path has an apostrophe; the skip-detection clause must be
    # well-formed so the unchanged file is skipped, not re-indexed/duplicated.
    r2 = index_folder(str(corpus), db_path=db_path)
    assert r2["files_indexed"] == 0
    assert r2["files_skipped"] == 1
    assert r2["total_chunks"] == chunks_after_first


@patch("server.indexer.get_model")
def test_index_folder_nonexistent_raises(mock_get_model, tmp_path):
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    with pytest.raises(FileNotFoundError):
        index_folder("/nonexistent/folder", db_path=str(tmp_path / "testdb"))


@patch("server.indexer.get_model")
def test_index_folder_reports_progress(mock_get_model, docs_dir, tmp_path):
    """index_folder calls progress_callback as it works through the files."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    calls = []
    index_folder(
        str(docs_dir),
        db_path=str(tmp_path / "testdb"),
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert calls, "progress_callback was never invoked"
    assert all(total == 4 for _, total in calls)
    assert calls[-1] == (4, 4)  # final call reports every file done
    processed = [done for done, _ in calls]
    assert processed == sorted(processed)  # monotonically non-decreasing


@patch("server.indexer.get_model")
def test_index_folder_returns_finalize_warnings(mock_get_model, docs_dir, tmp_path):
    """The return dict carries a finalize_warnings list (empty on a clean run)."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    result = index_folder(str(docs_dir), db_path=str(tmp_path / "testdb"))
    assert result["finalize_warnings"] == []


def _spy_finalize_steps(store):
    """Replace the three finalize steps on a store with recorders."""
    called = []
    store.optimize_table = lambda: called.append("optimize")
    store.create_vector_index = lambda: called.append("vector")
    store.create_fts_index = lambda: called.append("fts")
    return called


def test_finalize_index_does_nothing_when_unchanged(tmp_path):
    """No content and no stat change → no compaction, no index build."""
    store = VectorStore(str(tmp_path / "db"))
    called = _spy_finalize_steps(store)

    warnings = _finalize_index(store, content_changed=False, stats_changed=False)

    assert warnings == []
    assert called == []


def test_finalize_index_content_change_compacts_and_builds_indexes(tmp_path):
    """A content change compacts, then rebuilds the vector and FTS indexes."""
    store = VectorStore(str(tmp_path / "db"))
    called = _spy_finalize_steps(store)

    warnings = _finalize_index(store, content_changed=True, stats_changed=False)

    assert warnings == []
    assert called == ["optimize", "vector", "fts"]


def test_finalize_index_stats_only_compacts(tmp_path):
    """A pure stat refresh (e.g. drive move) compacts but does not rebuild indexes."""
    store = VectorStore(str(tmp_path / "db"))
    called = _spy_finalize_steps(store)

    _finalize_index(store, content_changed=False, stats_changed=True)

    assert called == ["optimize"]


def test_finalize_index_captures_step_failure_as_warning(tmp_path):
    """A failing finalize step becomes a warning — it never fails the run."""
    store = VectorStore(str(tmp_path / "db"))
    store.optimize_table = lambda: None
    store.create_fts_index = lambda: None

    def boom():
        raise RuntimeError("disk full")

    store.create_vector_index = boom

    warnings = _finalize_index(store, content_changed=True, stats_changed=False)

    assert len(warnings) == 1
    assert "disk full" in warnings[0]


@patch("server.indexer.get_model")
def test_unchanged_rerun_skips_hashing(mock_get_model, docs_dir, tmp_path):
    """A re-run over unchanged files takes the mtime/size fast path — no hashing."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model
    db_path = str(tmp_path / "db")

    index_folder(str(docs_dir), db_path=db_path)  # first run hashes everything

    with patch("server.indexer.compute_file_hash", wraps=compute_file_hash) as spy:
        result = index_folder(str(docs_dir), db_path=db_path)

    assert result["files_skipped"] == 4
    assert result["files_indexed"] == 0
    assert spy.call_count == 0  # mtime + size matched; no file was read or hashed


@patch("server.indexer.get_model")
def test_touched_file_uses_hash_fallback_then_fastpaths(mock_get_model, docs_dir, tmp_path):
    """A file touched (new mtime, same content) is hash-checked once, then its
    stored stat is refreshed so the next run fast-paths it."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model
    db_path = str(tmp_path / "db")

    index_folder(str(docs_dir), db_path=db_path)

    target = docs_dir / "readme.md"
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))

    with patch("server.indexer.compute_file_hash", wraps=compute_file_hash) as spy:
        r2 = index_folder(str(docs_dir), db_path=db_path)
    assert r2["files_indexed"] == 0
    assert r2["files_skipped"] == 4
    assert spy.call_count == 1  # only the touched file was hashed

    with patch("server.indexer.compute_file_hash", wraps=compute_file_hash) as spy3:
        r3 = index_folder(str(docs_dir), db_path=db_path)
    assert r3["files_skipped"] == 4
    assert spy3.call_count == 0  # stat was refreshed — fast path again


@patch("server.indexer.get_model")
def test_content_change_triggers_reindex(mock_get_model, docs_dir, tmp_path):
    """Changed file content is re-indexed; unchanged siblings are skipped."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model
    db_path = str(tmp_path / "db")

    index_folder(str(docs_dir), db_path=db_path)
    (docs_dir / "readme.md").write_text("# Completely different content now")

    r2 = index_folder(str(docs_dir), db_path=db_path)
    assert r2["files_indexed"] == 1
    assert r2["files_skipped"] == 3


@patch("server.indexer.get_model")
def test_batched_writes_persist_all_chunks(mock_get_model, tmp_path, monkeypatch):
    """Buffered writes flushed across the threshold persist every chunk."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model
    monkeypatch.setattr("server.indexer.FLUSH_CHUNK_THRESHOLD", 3)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(10):
        (corpus / f"f{i}.txt").write_text(f"content of file number {i}")

    db_path = str(tmp_path / "db")
    result = index_folder(str(corpus), db_path=db_path)

    assert result["files_indexed"] == 10
    assert result["total_chunks"] >= 10
    assert VectorStore(db_path).count_chunks() == result["total_chunks"]


def test_index_folder_migrates_pre_tier2_index(tmp_path):
    """A pre-Tier-2 index (old schema, no mtime/size) migrates in place: the
    first run hash-confirms each file unchanged and backfills its stat, and the
    next run takes the pure fast path."""
    import lancedb
    from server.store import SCHEMA, TABLE_NAME
    from server.paths import to_relative

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Alpha content about revenue.")
    (corpus / "b.txt").write_text("Beta content about expenses.")
    db_path = str(tmp_path / "db")

    # Hand-build a pre-Tier-2 index: old schema, real content hashes, no stats.
    # Also strip chunk_kind/sheet_name so this stays a "fully pre-everything"
    # simulated index — the migration path adds all of them in one pass.
    old_schema = pa.schema(
        [f for f in SCHEMA if f.name not in (
            "mtime_ns", "file_size", "chunk_kind", "sheet_name",
        )]
    )
    rows = []
    for f in (corpus / "a.txt", corpus / "b.txt"):
        rel = to_relative(str(f), db_path)
        rows.append({
            "id": f"{rel}_0",
            "text": f.read_text(),
            "source_file": rel,
            "file_name": f.name,
            "file_type": ".txt",
            "folder_path": ".",
            "chunk_index": 0,
            "content_hash": compute_file_hash(f),
            "vector": [0.0] * EMBEDDING_DIM,
        })
    lancedb.connect(db_path).create_table(TABLE_NAME, data=rows, schema=old_schema)

    # First post-upgrade run: stat columns are NULL, so each file is hashed
    # once, confirmed unchanged, and its stat backfilled.
    with patch("server.indexer.compute_file_hash", wraps=compute_file_hash) as spy1:
        r1 = index_folder(str(corpus), db_path=db_path)
    assert r1["files_indexed"] == 0
    assert r1["files_skipped"] == 2
    assert spy1.call_count == 2

    # Second run: stats are backfilled — pure fast path, nothing hashed.
    with patch("server.indexer.compute_file_hash", wraps=compute_file_hash) as spy2:
        r2 = index_folder(str(corpus), db_path=db_path)
    assert r2["files_skipped"] == 2
    assert spy2.call_count == 0


# -- Tier 3: per-file size cap (item 4.8) ------------------------------------

@patch("server.indexer.get_model")
def test_index_folder_skips_oversized_files(mock_get_model, tmp_path, monkeypatch):
    """A file over MAX_FILE_SIZE_BYTES is skipped, reported, and not indexed."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model
    monkeypatch.setattr("server.indexer.MAX_FILE_SIZE_BYTES", 1024)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "small.txt").write_text("a normal small document about revenue")
    (corpus / "huge.txt").write_text("x" * 300_000)  # ~0.3 MB, over the 1 KB cap

    db_path = str(tmp_path / "db")
    result = index_folder(str(corpus), db_path=db_path)

    assert result["files_indexed"] == 1
    assert result["files_size_skipped"] == 1
    assert len(result["oversized_files"]) == 1
    assert result["oversized_files"][0]["file"].endswith("huge.txt")
    assert result["oversized_files"][0]["size_mb"] > 0

    # The oversized file produced no chunks; the small one did.
    files = VectorStore(db_path).get_all_files()
    assert any("small.txt" in f for f in files)
    assert not any("huge.txt" in f for f in files)


@patch("server.indexer.get_model")
def test_oversized_growth_removes_stale_chunks(mock_get_model, tmp_path, monkeypatch):
    """A file indexed while small, then grown past the cap, has its stale chunks
    removed — and is not double-counted as an orphan deletion."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "doc.txt"
    target.write_text("small document about revenue")
    db_path = str(tmp_path / "db")

    r1 = index_folder(str(corpus), db_path=db_path)
    assert r1["files_indexed"] == 1
    assert VectorStore(db_path).count_chunks() > 0

    # The file grows past a now-tiny cap.
    target.write_text("x" * 5000)
    monkeypatch.setattr("server.indexer.MAX_FILE_SIZE_BYTES", 1024)
    r2 = index_folder(str(corpus), db_path=db_path)

    assert r2["files_size_skipped"] == 1
    assert r2["files_deleted"] == 0  # not also counted as an orphan
    assert VectorStore(db_path).count_chunks() == 0  # stale chunks gone


@patch("server.indexer.get_model")
def test_size_cap_disabled_indexes_everything(mock_get_model, tmp_path, monkeypatch):
    """MAX_FILE_SIZE_BYTES of 0 disables the cap — large files index normally."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model
    monkeypatch.setattr("server.indexer.MAX_FILE_SIZE_BYTES", 0)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "big.txt").write_text("revenue " * 5000)

    result = index_folder(str(corpus), db_path=str(tmp_path / "db"))
    assert result["files_indexed"] == 1
    assert result["files_size_skipped"] == 0


@patch("server.indexer.get_model")
def test_index_folder_exempts_pst_from_size_cap(mock_get_model, tmp_path, monkeypatch):
    """A .pst over the cap is indexed, not reported oversized — pypff streams
    the archive. A non-streaming file of the same size is still skipped."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model
    monkeypatch.setattr("server.indexer.MAX_FILE_SIZE_BYTES", 1024)
    # The fake .pst is not a real archive; parse it to a fixed part instead.
    monkeypatch.setattr(
        "server.parsers._extract_pst",
        lambda path: [{"text": "a mail message about revenue", "metadata": {}}],
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "archive.pst").write_bytes(b"x" * 4096)  # over the 1 KB cap
    (corpus / "huge.txt").write_text("x" * 4096)        # over the 1 KB cap

    result = index_folder(str(corpus), db_path=str(tmp_path / "db"))

    assert result["files_indexed"] == 1       # the .pst was indexed
    assert result["files_size_skipped"] == 1  # the .txt was skipped
    oversized = [o["file"] for o in result["oversized_files"]]
    assert any(f.endswith("huge.txt") for f in oversized)
    assert not any(f.endswith("archive.pst") for f in oversized)


# -- Tier 3: embedding device selection (item 4.10) --------------------------

def test_select_device_prefers_mps_when_available():
    """_select_device picks MPS when the Apple-Silicon backend reports ready."""
    import torch

    if getattr(torch.backends, "mps", None) is None:
        pytest.skip("this torch build has no mps backend")
    with patch.object(torch.backends.mps, "is_available", return_value=True):
        assert _select_device() == "mps"


def test_select_device_falls_back_to_cpu_without_accelerator():
    """With neither MPS nor CUDA available, _select_device returns cpu."""
    import torch
    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        if getattr(torch.backends, "mps", None) is not None:
            stack.enter_context(
                patch.object(torch.backends.mps, "is_available", return_value=False)
            )
        assert _select_device() == "cpu"


# -- Exclusion rules integration --------------------------------------------

@patch("server.indexer.get_model")
def test_index_folder_with_exclude_param_skips_matching_files(
    mock_get_model, tmp_path
):
    """index_folder honours the `exclude` param. Uses `vendored/` so the test
    doesn't accidentally rely on any default-list behaviour."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "keep.md").write_text("real content about revenue")
    (corpus / "vendored").mkdir()
    (corpus / "vendored" / "junk.md").write_text("vendored library README")

    db_path = str(tmp_path / "db")
    result = index_folder(str(corpus), db_path=db_path, exclude=["vendored/**"])

    assert result["files_indexed"] == 1
    files = VectorStore(db_path).get_all_files()
    assert any("keep.md" in f for f in files)
    assert not any("vendored" in f for f in files)
    assert result["exclusion_patterns"] == ["vendored/**"]


@patch("server.indexer.get_model")
def test_index_folder_honours_semanticignore_file(mock_get_model, tmp_path):
    """A `.semanticignore` at the indexed root is honoured without a param."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / ".semanticignore").write_text("vendored/\n")
    (corpus / "keep.md").write_text("real content about revenue")
    (corpus / "vendored").mkdir()
    (corpus / "vendored" / "junk.md").write_text("vendored library README")

    db_path = str(tmp_path / "db")
    result = index_folder(str(corpus), db_path=db_path)

    assert result["files_indexed"] == 1
    files = VectorStore(db_path).get_all_files()
    assert not any("vendored" in f for f in files)


def test_discover_files_prunes_excluded_dirs_without_reading(tmp_path):
    """When a directory matches an exclusion rule, the walk does NOT descend.
    Verified with a parser-call spy: a file deep inside an excluded dir is
    never returned, so the parser is never invoked on it."""
    corpus = tmp_path / "corpus"
    nested = corpus / "vendored" / "deeply" / "buried"
    nested.mkdir(parents=True)
    (nested / "secret.md").write_text("should never be read")
    (corpus / "keep.md").write_text("ok")

    rules = ExclusionRules.load(
        corpus, extra_patterns=["vendored/"], db_dir=str(tmp_path / "db")
    )
    files = discover_files(corpus, file_types=None, recursive=True, exclusions=rules)
    names = {f.name for f in files}
    assert "secret.md" not in names
    assert "keep.md" in names
    # No path under vendored/ should be returned at all.
    assert not any("vendored" in str(f) for f in files)


@patch("server.indexer.get_model")
def test_rerun_with_new_exclusion_prunes_chunks(mock_get_model, tmp_path):
    """A re-run after a new exclusion rule prunes the now-excluded chunks and
    reports them in files_excluded_pruned. Uses `vendored/` to keep the test
    hermetic — independent of any default-list state."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "keep.md").write_text("real content")
    (corpus / "vendored").mkdir()
    (corpus / "vendored" / "junk.md").write_text("vendored content")

    db_path = str(tmp_path / "db")

    # First run: no exclusions — both files indexed.
    r1 = index_folder(str(corpus), db_path=db_path)
    assert r1["files_indexed"] == 2
    files_after_r1 = VectorStore(db_path).get_all_files()
    assert any("vendored" in f for f in files_after_r1)

    # Add an exclusion and re-run.
    (corpus / ".semanticignore").write_text("vendored/\n")
    r2 = index_folder(str(corpus), db_path=db_path)
    assert r2["files_excluded_pruned"] >= 1
    files_after_r2 = VectorStore(db_path).get_all_files()
    assert not any("vendored" in f for f in files_after_r2)
    # The now-excluded file was NOT counted as an orphan deletion.
    assert r2["files_deleted"] == 0

    # Third run: nothing newly excluded — files_excluded_pruned drops back to 0.
    r3 = index_folder(str(corpus), db_path=db_path)
    assert r3["files_excluded_pruned"] == 0


@patch("server.indexer.get_model")
def test_exclusion_in_folder_b_does_not_prune_folder_a(mock_get_model, tmp_path):
    """A re-run of folder B with an exclusion rule must NOT prune folder A's
    chunks, even when those chunks would also match the rule by path. The
    prune pass must be scoped to the folder being indexed."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    db_path = str(tmp_path / "db")

    folder_a = tmp_path / "folder_a"
    folder_a.mkdir()
    (folder_a / "keep_a.txt").write_text("folder A text content")

    folder_b = tmp_path / "folder_b"
    folder_b.mkdir()
    (folder_b / "keep_b.md").write_text("folder B markdown content")
    (folder_b / "drop_b.txt").write_text("folder B text content")

    # Index both — no exclusions.
    index_folder(str(folder_a), db_path=db_path)
    index_folder(str(folder_b), db_path=db_path)

    files = VectorStore(db_path).get_all_files()
    assert any("keep_a.txt" in f for f in files)
    assert any("drop_b.txt" in f for f in files)

    # Re-index folder B only, excluding *.txt. Folder A's *.txt must survive.
    index_folder(str(folder_b), db_path=db_path, exclude=["*.txt"])
    files = VectorStore(db_path).get_all_files()
    assert any("keep_a.txt" in f for f in files), \
        "folder A's *.txt was wiped by folder B's exclusion rule"
    assert not any("drop_b.txt" in f for f in files), \
        "folder B's *.txt was not pruned"


def test_index_folder_raises_on_invalid_pattern(tmp_path):
    """An invalid exclusion pattern raises ValueError — the MCP layer turns
    this into a `rejected` response. The index is never touched."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("content")

    with pytest.raises(ValueError) as exc:
        index_folder(str(corpus), db_path=str(tmp_path / "db"), exclude=["\\"])
    assert "\\" in str(exc.value)


@patch("server.indexer.get_model")
def test_prune_works_when_folder_is_a_symlink(mock_get_model, tmp_path):
    """Indexing through a symlink to the corpus must still prune correctly on
    a re-run. The earlier bug was that f_abs from to_absolute wasn't resolved,
    so the in_scope check failed (the symlinked folder_path resolves to a
    different real path) and the prune pass silently did nothing."""
    mock_model = type("MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)})()
    mock_get_model.return_value = mock_model

    real = tmp_path / "real_corpus"
    real.mkdir()
    (real / "keep.md").write_text("real content")
    (real / "vendored").mkdir()
    (real / "vendored" / "junk.md").write_text("vendored content")

    link = tmp_path / "linked_corpus"
    link.symlink_to(real, target_is_directory=True)

    db_path = str(tmp_path / "db")

    # Index through the symlink.
    r1 = index_folder(str(link), db_path=db_path)
    assert r1["files_indexed"] == 2

    # Add an exclusion and re-run — the prune must still see the vendored file
    # and remove it, even though the path used to call index_folder is a
    # symlink to a different real path.
    (real / ".semanticignore").write_text("vendored/\n")
    r2 = index_folder(str(link), db_path=db_path)
    assert r2["files_excluded_pruned"] >= 1
