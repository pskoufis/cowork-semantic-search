# Multi-index search (fan-out) — design

Let the server search several LanceDB indexes at once. A query is submitted
to every configured index, each index ranks its own hits, and the per-index
ranked lists are fused into a single result list via Reciprocal Rank Fusion.
Reads (search, status) fan out; writes (index, reindex) stay single-index.

Builds on the configurable-embedding-model work: each index records its own
model/dim in `index_meta.json`, so the query is embedded with *that* index's
model before searching it. RRF is rank-based, so indexes built with different
models (and thus incomparable raw scores) still merge fairly.

## Goals

- Search across multiple indexes with one query; return one merged, ranked
  result list.
- Work correctly when indexes use *different* embedding models/dims.
- Aggregate `get_index_status` across all configured indexes.
- Fully backward compatible: single-index configs behave exactly as today.
- Isolate failures: a missing/busy/broken index is skipped, not fatal.

## Non-goals

- Fanning out **writes**. `index_folder` / `reindex_file` always target one
  explicit index.
- A persistent index registry / named indexes / per-index labels beyond the
  path itself. (`LANCEDB_PATHS` is the whole configuration surface.)
- Parallelizing the per-index searches. Sequential fan-out over a handful of
  indexes is enough; parallelism is a later optimization, not in scope.
- Score normalization/calibration. Fusion is purely rank-based (RRF).
- Cross-index global re-ranking with a cross-encoder.

## Configuration & resolution

New env var **`LANCEDB_PATHS`** — multiple index directories joined by the OS
path separator (`os.pathsep`: `:` on macOS/Linux), like `PATH`.

A single shared helper `resolve_db_dirs(explicit: str | None) -> list[str]`
resolves the list, with this precedence:

1. Explicit `db_path` argument (tool/CLI) → `[that]` (lets a caller still
   target one index even when many are configured).
2. `LANCEDB_PATHS` set → split on `os.pathsep`, strip blanks, de-dup
   (preserve order) → absolute paths.
3. `LANCEDB_PATH` (singular) set → `[that]`.
4. Neither → `[./lancedb]`.

Writes need exactly one target. `resolve_write_dir(explicit)` reuses the same
order but collapses to a single dir: explicit → `LANCEDB_PATH` →
**first of `LANCEDB_PATHS`** (logged: "indexing into <dir> (first of
LANCEDB_PATHS)") → `./lancedb`.

## Architecture & components

- **`server/multi_search.py`** *(new)* — owns fan-out + fusion:
  - `search_indexes(query, db_dirs, top_k, folder_path, file_type, mode)
    -> dict` — loops indexes, calls `search_one` per index, RRF-fuses,
    returns the merged response (with per-result provenance + a `skipped`
    list).
  - `_rrf_merge(per_index_results, top_k, rrf_k=60)` — fuse ranked lists.
- **`server/search.py`** *(refactor)* — extract the existing single-index
  body into `search_one(db_dir, query, top_k, folder_path, file_type, mode)
  -> list[dict]` (embed with the index's recorded model, run store
  vector/hybrid search, resolve stored paths to absolute, **tag each result
  with `index_path` and `model_alias`**). `semantic_search(...)` becomes a
  thin wrapper: `resolve_db_dirs(db_path)` → `search_indexes(...)`.
- **`server/db_paths.py`** *(new small module)* — `resolve_db_dirs` /
  `resolve_write_dir`. Keeps index-dir env-resolution in one place instead of
  the duplicated `os.environ.get("LANCEDB_PATH", "./lancedb")` lines currently
  scattered across `search.py`, `indexer.py`, and `main.py`. (Distinct from
  the existing `server/paths.py`, which handles relative/absolute *file* path
  conversion.)
- **`server/main.py`** — `semantic_search` tool passes `db_path` through
  unchanged (resolution happens downstream). `get_index_status` aggregates
  (below). Write tools (`index_folder`, `reindex_file`) call
  `resolve_write_dir`.
- **`cli/main.py` / `cli/commands.py`** — `search` and `status` verbs fan out;
  `index`/`reindex` use the write resolver.

### Data flow (one query)

```
db_dirs = resolve_db_dirs(explicit_db_path)
per_index = []
skipped = []
for db_dir in db_dirs:
    try:
        hits = search_one(db_dir, query, top_k*2, folder, ftype, mode)  # over-fetch
        per_index.append((db_dir, hits))
    except Exception as e:
        skipped.append({"index": db_dir, "reason": str(e)})
merged = _rrf_merge(per_index, top_k)
return {query, mode, results: merged, total_results, indexes_searched, skipped}
```

### RRF fusion & dedup

Reuses the rank-based scheme already in `store.hybrid_search`. Fusion key is
the **absolute** `source_file` + `chunk_index` (paths are resolved to absolute
in `search_one`, so the same physical chunk in two indexes collides on one
key). For each index's ranked list, add `1 / (rrf_k + rank + 1)` to that key's
score. A chunk appearing in multiple indexes therefore **sums** contributions
(cross-index agreement boosts it) and is returned once. Each merged result
keeps the `index_path`/`model_alias` of the **highest-ranked** index it came
from, plus a `from_indexes` list of every contributing index.

## Status aggregation

`get_index_status` (and the CLI `status`) report a per-index breakdown plus
totals:

```json
{
  "indexes": [
    {"index_path": "...", "model_alias": "qwen3-0.6b", "dim": 256,
     "total_chunks": 1200, "total_files": 80, "db_size_bytes": ...,
     "db_size": "..."},
    {"index_path": "...", "model_alias": "bge-small", "dim": 384, ...}
  ],
  "total_chunks": <sum>,
  "total_files": <sum>,
  "db_size_bytes": <sum>,
  "db_size": "<human sum>",
  "jobs": { ... }            // unchanged
}
```

A missing/unreadable index contributes a stub entry (`model_alias: null`,
zero counts) rather than failing the call.

## Error handling

- Per-index search failure (missing dir, no table, live-index lock, dim
  guard) → caught, index added to `skipped`, fan-out continues.
- Empty resolved list is impossible (always falls back to `./lancedb`).
- If **every** index is skipped, return an empty result list with the
  `skipped` detail so the caller sees why.

## Testing

- Two seeded indexes (different models/dims) → merged results, each tagged
  with its origin `index_path`; ordering follows RRF.
- Same chunk present in two indexes → returned once, RRF score is the sum,
  ranked above a single-index-only chunk of equal per-index rank.
- One index dir missing/corrupt → search still returns the other's hits and
  lists the bad one under `skipped`.
- Backward compat: `LANCEDB_PATH` only, `./lancedb` default, and an explicit
  `db_path` arg each resolve to a single index and behave as before.
- `resolve_write_dir`: with only `LANCEDB_PATHS` set, indexing targets the
  first path (and logs it).
- `get_index_status` aggregates per-index entries + correct totals across two
  indexes; tolerates a missing index.
