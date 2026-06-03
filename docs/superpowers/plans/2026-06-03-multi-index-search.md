# Multi-index Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search all configured LanceDB indexes at once, fusing per-index ranked results via RRF, while writes stay single-index and status aggregates across indexes.

**Architecture:** A new `server/db_paths.py` resolves `LANCEDB_PATHS` (and the legacy `LANCEDB_PATH`) to a list of index dirs. `search.py`'s single-index body is extracted into `search_one(db_dir, …)` that tags each hit with its origin; `server/multi_search.py` fans out over the list and RRF-merges. `get_index_status` aggregates per-index. Write paths resolve to one dir.

**Tech Stack:** Python, LanceDB, sentence-transformers, pytest.

---

## File Structure

- `server/db_paths.py` *(new)* — `resolve_db_dirs(explicit) -> list[str]`, `resolve_write_dir(explicit) -> str`. Single home for index-dir env resolution.
- `server/search.py` *(modify)* — extract `search_one(db_dir, …) -> list[dict]` (tags results with `index_path`/`model_alias`); `semantic_search` delegates to multi-search.
- `server/multi_search.py` *(new)* — `search_indexes(...)` fan-out + `_rrf_merge(...)`.
- `server/main.py` *(modify)* — `get_index_status` aggregates; write tools use `resolve_write_dir`.
- `server/indexer.py` *(modify)* — `index_folder` / `reindex_one_file` use `resolve_write_dir`.
- `cli/main.py` + `cli/commands.py` *(modify)* — `search`/`status` fan out; `index`/`reindex` use write resolver.

---

### Task 1: Index-dir resolver

**Files:**
- Create: `server/db_paths.py`
- Test: `tests/test_db_paths.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db_paths.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_paths.py -v`
Expected: FAIL — `No module named 'server.db_paths'`

- [ ] **Step 3: Write minimal implementation**

```python
# server/db_paths.py
"""Resolve which LanceDB index directories an operation targets.

Reads are fan-out: `resolve_db_dirs` returns every configured index.
Writes are single-target: `resolve_write_dir` collapses to one.

Precedence (reads): explicit arg → LANCEDB_PATHS (os.pathsep list) →
LANCEDB_PATH (singular) → ./lancedb.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _from_paths_env() -> list[str]:
    raw = os.environ.get("LANCEDB_PATHS")
    if not raw:
        return []
    seen: dict[str, None] = {}
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        seen.setdefault(os.path.abspath(part), None)
    return list(seen)


def resolve_db_dirs(explicit: str | None) -> list[str]:
    """The list of index dirs a read (search/status) should span."""
    if explicit is not None:
        return [os.path.abspath(explicit)]
    paths = _from_paths_env()
    if paths:
        return paths
    singular = os.environ.get("LANCEDB_PATH")
    if singular:
        return [os.path.abspath(singular)]
    return [os.path.abspath("./lancedb")]


def resolve_write_dir(explicit: str | None) -> str:
    """The single index dir a write (index/reindex) should target."""
    if explicit is not None:
        return os.path.abspath(explicit)
    singular = os.environ.get("LANCEDB_PATH")
    if singular:
        return os.path.abspath(singular)
    paths = _from_paths_env()
    if paths:
        if len(paths) > 1:
            logger.info("indexing into %s (first of LANCEDB_PATHS)", paths[0])
        return paths[0]
    return os.path.abspath("./lancedb")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_paths.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add server/db_paths.py tests/test_db_paths.py
git commit -m "feat(search): resolve LANCEDB_PATHS to a list of index dirs"
```

---

### Task 2: Extract `search_one` with provenance tagging

**Files:**
- Modify: `server/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_search.py
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
```

(`_fake_embed_384` already exists in `tests/test_search.py` from the configurable-model work.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py::test_search_one_tags_results_with_origin -v`
Expected: FAIL — `cannot import name 'search_one'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `server/search.py` (keep imports; add `os.path.abspath` of `db_path`). New structure — `search_one` holds the old single-index logic and tags results; `semantic_search` is added in Task 4:

```python
"""Search logic: embed query, vector/hybrid search, format results."""

import os

from server.store import VectorStore
from server.indexer import get_model, encode_query
from server.index_meta import resolve_index_profile
from server.paths import to_relative, to_absolute


def search_one(
    db_dir: str,
    query: str,
    top_k: int,
    folder_path: str | None,
    file_type: str | None,
    mode: str,
) -> list[dict]:
    """Search a single index. Returns hits tagged with their origin index.

    db_dir must be absolute. Each result gains `index_path` and `model_alias`
    so a fan-out caller can show (and fuse by) provenance.
    """
    profile, dim = resolve_index_profile(
        db_dir, env_alias=None, env_dim=None, for_write=False
    )
    store = VectorStore(db_dir, dim=dim)
    model = get_model(profile, dim)
    query_embedding = encode_query(model, profile, query)[0].tolist()

    folder_filter = to_relative(folder_path, db_dir) if folder_path else None

    if mode == "hybrid":
        results = store.hybrid_search(
            query_text=query, query_vector=query_embedding, top_k=top_k,
            folder_path=folder_filter, file_type=file_type,
        )
    else:
        results = store.vector_search(
            query_vector=query_embedding, top_k=top_k,
            folder_path=folder_filter, file_type=file_type,
        )

    for r in results:
        r["source_file"] = to_absolute(r["source_file"], db_dir)
        r["folder_path"] = to_absolute(r["folder_path"], db_dir)
        r["index_path"] = db_dir
        r["model_alias"] = profile.alias
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_search.py::test_search_one_tags_results_with_origin -v`
Expected: PASS

(Existing `semantic_search` tests will FAIL here because `semantic_search` is temporarily gone — Task 4 restores it. Run the full `test_search.py` only after Task 4.)

- [ ] **Step 5: Commit**

```bash
git add server/search.py tests/test_search.py
git commit -m "refactor(search): extract search_one with origin tagging"
```

---

### Task 3: RRF fan-out merge

**Files:**
- Create: `server/multi_search.py`
- Test: `tests/test_multi_search.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_search.py
from server.multi_search import _rrf_merge


def _hit(src, ci, idx):
    return {"source_file": src, "chunk_index": ci, "index_path": idx,
            "model_alias": "m", "text": src}


def test_rrf_merges_and_dedups_same_chunk():
    # Same (source_file, chunk_index) tops both indexes -> summed, returned once.
    a = [_hit("/f.txt", 0, "/idxA"), _hit("/g.txt", 0, "/idxA")]
    b = [_hit("/f.txt", 0, "/idxB"), _hit("/h.txt", 0, "/idxB")]
    merged = _rrf_merge([("/idxA", a), ("/idxB", b)], top_k=10)
    keys = [(m["source_file"], m["chunk_index"]) for m in merged]
    assert keys.count(("/f.txt", 0)) == 1            # deduped
    top = merged[0]
    assert top["source_file"] == "/f.txt"            # boosted by appearing in both
    assert set(top["from_indexes"]) == {"/idxA", "/idxB"}


def test_rrf_respects_top_k():
    a = [_hit(f"/a{i}.txt", 0, "/idxA") for i in range(5)]
    merged = _rrf_merge([("/idxA", a)], top_k=3)
    assert len(merged) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_multi_search.py -v`
Expected: FAIL — `No module named 'server.multi_search'`

- [ ] **Step 3: Write minimal implementation**

```python
# server/multi_search.py
"""Fan-out search across multiple indexes with rank-based fusion.

Each index is searched independently (with its own embedding model), then the
per-index ranked lists are fused by Reciprocal Rank Fusion. RRF is rank-based,
so indexes built with different models — whose raw scores are incomparable —
still merge fairly.
"""

import logging

from server.search import search_one

logger = logging.getLogger(__name__)


def _rrf_merge(per_index: list[tuple[str, list[dict]]], top_k: int, rrf_k: int = 60) -> list[dict]:
    """Fuse per-index ranked lists. Key = (absolute source_file, chunk_index);
    a chunk seen in several indexes sums its contributions and is returned once.
    """
    scores: dict[tuple, float] = {}
    rep: dict[tuple, dict] = {}
    froms: dict[tuple, list[str]] = {}
    for _idx, results in per_index:
        for rank, r in enumerate(results):
            key = (r["source_file"], r["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            froms.setdefault(key, []).append(r["index_path"])
            # Keep the representative from the best (lowest) rank seen so far.
            if key not in rep or rank < rep[key]["_rank"]:
                rep[key] = {**r, "_rank": rank}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for key, score in ranked:
        row = {k: v for k, v in rep[key].items() if k != "_rank"}
        row["score"] = score
        row["rrf_score"] = score
        row["from_indexes"] = froms[key]
        out.append(row)
    return out


def search_indexes(
    query: str,
    db_dirs: list[str],
    top_k: int = 25,
    folder_path: str | None = None,
    file_type: str | None = None,
    mode: str = "vector",
) -> dict:
    """Search every dir in db_dirs, fuse with RRF, return one ranked response."""
    per_index: list[tuple[str, list[dict]]] = []
    skipped: list[dict] = []
    for db_dir in db_dirs:
        try:
            hits = search_one(db_dir, query, top_k * 2, folder_path, file_type, mode)
            per_index.append((db_dir, hits))
        except Exception as exc:  # missing dir, no table, live-lock, dim guard
            logger.warning("skipping index %s: %s", db_dir, exc)
            skipped.append({"index": db_dir, "reason": str(exc)})
    merged = _rrf_merge(per_index, top_k)
    return {
        "query": query,
        "mode": mode,
        "results": merged,
        "total_results": len(merged),
        "indexes_searched": [d for d, _ in per_index],
        "skipped": skipped,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_multi_search.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add server/multi_search.py tests/test_multi_search.py
git commit -m "feat(search): RRF fan-out merge across indexes"
```

---

### Task 4: Wire `semantic_search` to fan out

**Files:**
- Modify: `server/search.py` (add `semantic_search` back, delegating to multi)
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_search.py
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
    missing = str(tmp_path / "nope")
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    monkeypatch.setenv("LANCEDB_PATHS", f"{good}{os.pathsep}{missing}")

    def fake_get_model(p, d):
        return type("M", (), {"encode": lambda self, t, **k: _fake_embed_384(t)})()

    with patch("server.search.get_model", fake_get_model):
        out = semantic_search("hello", db_path=None)
    assert out["total_results"] >= 1
    assert any(s_["index"] == os.path.abspath(missing) for s_ in out["skipped"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py::test_semantic_search_fans_out_over_two_indexes -v`
Expected: FAIL — `semantic_search` not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `server/search.py`:

```python
def semantic_search(
    query: str,
    db_path: str | None = None,
    folder_path: str | None = None,
    top_k: int = 25,
    file_type: str | None = None,
    mode: str = "vector",
) -> dict:
    """Search all configured indexes (or one, if db_path/LANCEDB_PATH names a
    single index) and return one RRF-fused ranked list."""
    from server.db_paths import resolve_db_dirs
    from server.multi_search import search_indexes

    db_dirs = resolve_db_dirs(db_path)
    folder_abs = os.path.abspath(folder_path) if folder_path else None
    return search_indexes(
        query=query, db_dirs=db_dirs, top_k=top_k,
        folder_path=folder_abs, file_type=file_type, mode=mode,
    )
```

Note: `search_indexes` is imported lazily to avoid a circular import (`multi_search` imports `search_one` from this module).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS — new fan-out tests pass; the pre-existing single-index `semantic_search` tests still pass (a single resolved dir returns that index's hits, now also carrying `index_path`/`from_indexes`).

- [ ] **Step 5: Commit**

```bash
git add server/search.py tests/test_search.py
git commit -m "feat(search): semantic_search fans out across configured indexes"
```

---

### Task 5: Aggregate `get_index_status`

**Files:**
- Modify: `server/main.py` (`get_index_status`)
- Test: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_mcp_tools.py
@pytest.mark.anyio
async def test_status_aggregates_across_indexes(tmp_path, monkeypatch):
    import json as _json
    from server.index_meta import write_meta
    from server.embedding_models import resolve_profile

    a, b = str(tmp_path / "A"), str(tmp_path / "B")
    write_meta(a, *resolve_profile("minilm", None))
    write_meta(b, *resolve_profile("qwen3-0.6b", None))
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    monkeypatch.setenv("LANCEDB_PATHS", f"{a}{os.pathsep}{b}")

    async with Client(mcp) as client:
        result = await client.call_tool("get_index_status", {})
        payload = _json.loads(result.content[0].text)
        aliases = {e["model_alias"] for e in payload["indexes"]}
        assert aliases == {"minilm", "qwen3-0.6b"}
        assert payload["total_chunks"] == 0  # nothing indexed, but both listed
        assert len(payload["indexes"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test_status_aggregates_across_indexes -v`
Expected: FAIL — no `indexes` key.

- [ ] **Step 3: Write minimal implementation**

In `server/main.py` `get_index_status`, replace the single-`db_dir` block (the lines from `if db_path is None:` through the `stats = {...}`/`try/except` that builds `stats`) with per-index aggregation. Keep the `jobs` and `exclusions` tail unchanged. Use the first resolved dir for the exclusions/`db_dir` references later in the function.

```python
    from server.store import VectorStore
    from server.paths import to_absolute
    from server.jobs import registry
    from server.exclusions import ExclusionRules, SEMANTICIGNORE_FILENAME
    from server.index_meta import read_meta
    from server.db_paths import resolve_db_dirs
    from pathlib import Path as _Path

    db_dirs = resolve_db_dirs(db_path)
    db_dir = db_dirs[0]  # used by the exclusions report below

    index_entries: list[dict] = []
    all_files: list[str] = []
    for d in db_dirs:
        meta = read_meta(d)
        entry = {
            "index_path": d,
            "model_alias": meta["model_alias"] if meta else None,
            "dim": meta["dim"] if meta else None,
            "total_chunks": 0,
            "total_files": 0,
            "db_size_bytes": 0,
            "db_size": "0 B",
        }
        try:
            store = VectorStore(d)
            files = sorted(to_absolute(f, d) for f in store.get_all_files())
            size_bytes = store.db_size_bytes()
            entry.update(
                total_chunks=store.count_chunks(),
                total_files=len(files),
                db_size_bytes=size_bytes,
                db_size=_human_size(size_bytes),
            )
            all_files.extend(files)
        except Exception:
            pass  # status stays readable even if one index is missing/busy
        index_entries.append(entry)

    total_size = sum(e["db_size_bytes"] for e in index_entries)
    single = len(index_entries) == 1
    stats = {
        "total_chunks": sum(e["total_chunks"] for e in index_entries),
        "total_files": len(set(all_files)),
        "indexed_files": sorted(set(all_files)),
        "db_size_bytes": total_size,
        "db_size": _human_size(total_size),
        "model_alias": index_entries[0]["model_alias"] if single else None,
        "dim": index_entries[0]["dim"] if single else None,
        "indexes": index_entries,
    }
```

(The existing `out = {**stats, "jobs": {...}}` and `folder_path` exclusions block stay as-is; they already reference `db_dir`, now the first index.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcp_tools.py -v`
Expected: PASS — new aggregation test passes; all existing `get_index_status` tests still pass (single index → `indexes` has one entry, top-level keys unchanged, including `model_alias`/`dim`).

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_mcp_tools.py
git commit -m "feat(status): aggregate get_index_status across configured indexes"
```

---

### Task 6: Write paths target a single index

**Files:**
- Modify: `server/indexer.py` (`index_folder` ~402-404, `reindex_one_file` ~261-263), `server/main.py` (write tools that compute `db_dir` from `LANCEDB_PATH`)
- Test: `tests/test_indexer.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_indexer.py
@patch("server.indexer.get_model")
def test_index_folder_writes_to_first_of_paths(mock_get_model, tmp_path, monkeypatch):
    mock_get_model.return_value = type(
        "MockModel", (), {"encode": lambda self, t, **k: _fake_embed(t)}
    )()
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello")
    a, b = str(tmp_path / "A"), str(tmp_path / "B")
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    monkeypatch.setenv("LANCEDB_PATHS", f"{a}{os.pathsep}{b}")

    index_folder(str(corpus), unpack_first=False)  # no db_path → first of paths

    from server.index_meta import read_meta
    assert read_meta(a) is not None       # wrote into the first index
    assert read_meta(b) is None           # not the second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indexer.py::test_index_folder_writes_to_first_of_paths -v`
Expected: FAIL — `index_folder` resolves `LANCEDB_PATH` only (`./lancedb`), ignoring `LANCEDB_PATHS`.

- [ ] **Step 3: Write minimal implementation**

In `server/indexer.py`, replace the db-path resolution in `index_folder`:

```python
    if db_path is None:
        from server.db_paths import resolve_write_dir
        db_dir = resolve_write_dir(None)
    else:
        db_dir = os.path.abspath(db_path)
```

and identically in `reindex_one_file`:

```python
    if db_path is None:
        from server.db_paths import resolve_write_dir
        db_dir = resolve_write_dir(None)
    else:
        db_dir = os.path.abspath(db_path)
```

In `server/main.py`, the write-tool wrappers (`index_folder`, `reindex_file`, `submit_description`, `_auto_drain_descriptions`) currently do `db_path = os.environ.get("LANCEDB_PATH", "./lancedb")` when `db_path is None`. Replace each such line with:

```python
    if db_path is None:
        from server.db_paths import resolve_write_dir
        db_path = resolve_write_dir(None)
    db_dir = os.path.abspath(db_path)
```

Run to find them: `grep -n 'os.environ.get("LANCEDB_PATH", "./lancedb")' server/main.py server/indexer.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_indexer.py -v`
Expected: PASS — new test passes; existing indexer tests unchanged (they pass explicit `db_path`).

- [ ] **Step 5: Commit**

```bash
git add server/indexer.py server/main.py tests/test_indexer.py
git commit -m "feat(index): write paths resolve to a single target index"
```

---

### Task 7: CLI search + status fan out

**Files:**
- Modify: `cli/main.py` (`_resolve_db_path` usage), `cli/commands.py` (`search_cmd`, status printer)
- Test: `tests/cli/test_status.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/cli/test_status.py
import os as _os


def test_cli_status_lists_each_index(tmp_path, monkeypatch, capsys):
    from server.index_meta import write_meta
    from server.embedding_models import resolve_profile
    from cli.commands import status_cmd  # status printer entry

    a, b = str(tmp_path / "A"), str(tmp_path / "B")
    write_meta(a, *resolve_profile("minilm", None))
    write_meta(b, *resolve_profile("qwen3-0.6b", None))
    monkeypatch.delenv("LANCEDB_PATH", raising=False)
    monkeypatch.setenv("LANCEDB_PATHS", f"{a}{_os.pathsep}{b}")

    status_cmd(db_path=None)  # None → fan out
    out = capsys.readouterr().out
    assert "minilm" in out and "qwen3-0.6b" in out
```

(Confirm the status entry point's name/signature in `cli/commands.py` and match it; the printer is the function the `status` verb calls.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_status.py::test_cli_status_lists_each_index -v`
Expected: FAIL — printer takes a single resolved `db_path` and shows one index.

- [ ] **Step 3: Write minimal implementation**

In `cli/main.py`, stop pre-collapsing the path for the `search` and `status` verbs: pass the **explicit** value (`args.db_path`, which is `None` when not given) to those commands instead of `_resolve_db_path(...)`. Keep `_resolve_db_path` for `index`/`reindex`/`unpack` (single-target), or switch those to `resolve_write_dir`. Concretely, where the dispatch calls `search_cmd`/`status_cmd`, pass `db_path=args.db_path`.

In `cli/commands.py`, make `search_cmd` accept `db_path: str | None` and drop the `os.path.isdir(db_path)` pre-check (fan-out handles missing indexes via `skipped`); print a `skipped` note when present:

```python
def search_cmd(query, *, db_path: str | None, mode="vector", top_k=10, folder=None) -> int:
    from server.search import semantic_search
    if mode not in ("vector", "hybrid"):
        logger.error("--mode must be 'vector' or 'hybrid', got %r", mode)
        return 1
    folder_abs = os.path.abspath(folder) if folder else None
    result = semantic_search(query=query, db_path=db_path, folder_path=folder_abs,
                             top_k=top_k, mode=mode)
    for sk in result.get("skipped", []):
        logger.warning("skipped index %s (%s)", sk["index"], sk["reason"])
    results = result.get("results", [])
    if not results:
        print(f"No results for: {query!r} (mode={mode}, top_k={top_k})")
        return 0
    print(f"Searching ({mode}, n={top_k}) across {len(result.get('indexes_searched', []))} index(es)…")
    print()
    for i, row in enumerate(results, start=1):
        score = row.get("score") or 0.0
        path = row.get("source_file", "(unknown)")
        origin = os.path.basename(row.get("index_path", "")) or "?"
        snippet = (row.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:197] + "…"
        print(f"{i:>2}  {float(score):.3f}  [{origin}]  {path}")
        if snippet:
            print(f"      …{snippet}…")
    return 0
```

Make the status printer fan out: resolve `resolve_db_dirs(db_path)` and print one block per index (size, chunks, files, model). Keep the cold-start message when a dir doesn't exist. (Match the existing function's name/signature; reuse `_human_size`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/ -v`
Expected: PASS — new test passes; existing CLI tests still pass (single `--db-path`/`LANCEDB_PATH` resolves to one index).

- [ ] **Step 5: Commit**

```bash
git add cli/main.py cli/commands.py tests/cli/test_status.py
git commit -m "feat(cli): search and status fan out across configured indexes"
```

---

### Task 8: Full suite + docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 2: Document `LANCEDB_PATHS`**

Add a collapsible subsection under `## Advanced Usage` in `README.md`:

```markdown
<details>
<summary><strong>Searching multiple indexes at once</strong></summary>

Set `LANCEDB_PATHS` to several index directories (joined like `PATH`, with `:`
on macOS/Linux) and every search fans out across all of them, fusing the
per-index results into one ranked list via Reciprocal Rank Fusion. Because RRF
is rank-based, indexes built with **different embedding models** merge fairly.
`get_index_status` reports a per-index breakdown plus totals.

```json
{ "env": { "LANCEDB_PATHS": "/data/idx-qwen:/data/idx-bge" } }
```

Indexing still targets one index: pass `--db-path` (or `LANCEDB_PATH`); with
only `LANCEDB_PATHS` set, indexing writes to the **first** path. A missing or
busy index is skipped (search still returns the others' hits).

</details>
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document LANCEDB_PATHS multi-index search"
```

---

## Self-Review Notes

- **Spec coverage:** config resolution (Task 1), `search_one` + provenance (Task 2), RRF fan-out/dedup (Task 3), `semantic_search` fan-out + failure isolation (Task 4), status aggregation (Task 5), single-target writes (Task 6), CLI fan-out (Task 7), docs (Task 8). All spec sections map to a task.
- **Backward compat:** single-index configs (`LANCEDB_PATH`, `./lancedb`, explicit arg) resolve to a one-element list; existing search/status tests pass unchanged; results just gain extra `index_path`/`from_indexes` keys.
- **Type consistency:** `resolve_db_dirs(explicit)->list[str]`, `resolve_write_dir(explicit)->str`; `search_one(db_dir, query, top_k, folder_path, file_type, mode)->list[dict]` tags `index_path`/`model_alias`; `search_indexes(query, db_dirs, top_k, folder_path, file_type, mode)->dict` with keys `results/indexes_searched/skipped`; `_rrf_merge(per_index, top_k, rrf_k)` adds `from_indexes`/`score`. Used consistently across Tasks 2-7.
- **Circular import:** `multi_search` imports `search_one` from `search`; `search.semantic_search` imports `search_indexes` lazily inside the function body to break the cycle.
