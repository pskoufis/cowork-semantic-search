# Configurable embedding model & dimension — design

Make the embedding model and vector dimension selectable per execution
instead of hardcoded. The model and dim are chosen at index-creation
time via environment variables, recorded into the index itself, and
read back automatically for search and status — so each index is
permanently and safely bound to the model that built it.

## Goals

- Select the embedding model and dimension without editing source.
- Support multiple independent indexes, each built with a different
  model/dim (e.g. `lancedb-qwen`, `lancedb-bge`), per the chosen
  "separate index per model" workflow.
- Make an index self-describing: search/status load the correct model
  automatically, with no env configuration required at query time.
- Prevent silent mismatch — pointing the wrong model/dim at an existing
  index fails loudly instead of returning meaningless results.
- Preserve today's behaviour for existing indexes (Qwen / 256) with no
  breaking changes and no forced re-index.

## Non-goals

- Swapping the model on an existing index in place. Switching models
  means building a new index directory. (Chosen workflow: separate
  index per model.)
- CLI flags for model/dim. Configuration is env-var only
  (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), matching the existing
  `LANCEDB_PATH` pattern and keeping the MCP server's `env` block the
  single place to configure.
- Arbitrary/unknown model names. Only models in the curated registry
  are accepted; unknown aliases are rejected with the list of valid
  ones.
- Multi-index registry / named indexes (already out of scope project-wide).
- Re-ranking, cross-encoders, or API-hosted embedding backends.

## Configuration

Two environment variables, read only when **creating** an index:

- `EMBEDDING_MODEL` — an **alias** into the curated registry. Default
  `qwen3-0.6b` (current behaviour). Not a raw HuggingFace id.
- `EMBEDDING_DIM` — optional integer overriding the profile's default
  dim. Validated against the model (see registry rules below).

For an **existing** index these are ignored in favour of what the index
recorded; if they are set and conflict with the recorded values during
an `index`/`reindex` op, that is a hard error (see Behaviour).

## Curated model registry

New module `server/embedding_models.py`.

```python
@dataclass(frozen=True)
class EmbeddingProfile:
    alias: str                    # registry key, e.g. "bge-small"
    model_id: str                 # HF id, e.g. "BAAI/bge-small-en-v1.5"
    default_dim: int
    mrl: bool                     # True if Matryoshka-truncatable
    min_dim: int | None           # for mrl models; else None
    padding_side: str | None      # "left" for Qwen last-token pooling; else None
    query_prompt_name: str | None # ST built-in prompt name (Qwen: "query")
    query_prefix: str | None      # instruction prefix (bge/e5); mutually exclusive with prompt_name
    normalize: bool = True
```

Initial registry entries:

| alias | model_id | default_dim | mrl (min–max) | padding_side | query handling |
|---|---|---|---|---|---|
| `qwen3-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | 256 | yes (64–1024) | `left` | `query_prompt_name="query"` |
| `bge-small` | `BAAI/bge-small-en-v1.5` | 384 | no | none | `query_prefix="Represent this sentence for searching relevant passages: "` |
| `gte-small` | `thenlper/gte-small` | 384 | no | none | none |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | no | none | none |
| `static-mrl` | `sentence-transformers/static-retrieval-mrl-en-v1` | 256 | yes (64–1024) | none | none |

For MRL models, `min_dim`/native-max are the bounds shown above; every
candidate dim must additionally be divisible by 8.

`resolve_profile(alias, dim_override) -> (EmbeddingProfile, dim)`:

- Unknown `alias` → `ValueError` listing available aliases.
- `dim` = `dim_override or profile.default_dim`.
- `dim` must be divisible by 8 (IVF_PQ `num_sub_vectors = dim // 8`),
  else `ValueError`.
- For non-MRL models, `dim` must equal `default_dim`, else `ValueError`
  ("model X is fixed at N dims").
- For MRL models, `min_dim <= dim <= default_dim` (native max), else
  `ValueError`.

## Index metadata

New small module `server/index_meta.py`. A JSON sidecar
`index_meta.json` lives in the index directory (`db_dir`):

```json
{ "model_alias": "bge-small", "model_id": "BAAI/bge-small-en-v1.5", "dim": 384 }
```

- `write_meta(db_dir, profile, dim)` — called once when an index is
  first created.
- `read_meta(db_dir) -> meta | None` — returns `None` if absent.
- `resolve_index_profile(db_dir, env_alias, env_dim)`:
  - If meta exists → return its `(profile, dim)`; this is the source of
    truth. For `index`/`reindex`, if env config is set and disagrees,
    raise a clear mismatch error naming both recorded and configured
    values. For search/status, env is ignored entirely.
  - If meta absent and the index has data → **legacy fallback**: treat
    as `qwen3-0.6b` / 256 (today's hardcoded values) and write the meta
    file so it is recorded going forward.
  - If meta absent and index is new/empty → resolve from env/defaults
    and write meta.

## Component changes

- **`server/embedding_models.py`** (new) — profile dataclass, registry,
  `resolve_profile`. Qwen's special-casing becomes one registry entry.
- **`server/index_meta.py`** (new) — read/write/reconcile `index_meta.json`.
- **`server/indexer.py`** — remove the `EMBEDDING_MODEL` constant;
  `get_model()` takes `(profile, dim)` and caches keyed by
  `(model_id, dim)`; apply `padding_side` from the profile; on index
  creation, resolve via `index_meta` and write meta.
- **`server/store.py`** — remove the module-level `EMBEDDING_DIM = 256`;
  build the schema and IVF_PQ `num_sub_vectors` from a `dim` passed in
  (sourced from `index_meta`). Keep `_check_dim_compat` as a backstop.
- **`server/search.py`** — load the profile via `index_meta`; encode the
  query per profile: `query_prompt_name` (Qwen) → ST prompt; else
  `query_prefix` (bge) → prepend; else plain. No env read at query time.
- **`status` command** — display the index's recorded model alias and dim.

CLI surface (`cli/main.py`) gains no new flags; the MCP `env` block
remains the single configuration point.

## Behaviour & error handling

- **New index:** env (or defaults) → resolved profile/dim → written to
  `index_meta.json`.
- **Existing index, `index`/`reindex`:** configured model/dim must match
  recorded meta, else hard error:
  "index at <path> was built with qwen3-0.6b/256; you configured
  bge-small/384 — point --db-path at a new directory or unset the env
  override."
- **Search / status:** ignore env; load the model recorded in meta. Just
  works with zero query-time configuration.
- **Unknown alias:** error listing valid aliases.
- **Invalid dim:** error explaining the constraint that failed.
- **Backward compat:** an existing `lancedb` with no meta file is treated
  as `qwen3-0.6b`/256; nothing breaks and meta is written on the next op.

## Testing

- Registry: known alias resolves; unknown rejected; dim validation
  (MRL range ok, fixed-dim mismatch rejected, non-divisible-by-8
  rejected).
- `index_meta` round-trip: write then read; mismatch detection raises a
  clear error; legacy fallback returns qwen3-0.6b/256 and writes meta.
- End-to-end: build a tiny index with `minilm` in a temp dir, then
  search it with **no env set**, asserting the correct model loads from
  meta and results return.
- Backward compat: a pre-existing index dir with no meta file searches
  successfully under the legacy default.
- Coexistence: two index dirs built with different models/dims are each
  searchable independently in the same process.

## Migration

No data migration. Existing indexes keep working via the legacy
fallback; the meta file is added transparently on the next index/reindex.
New indexes are created with whatever `EMBEDDING_MODEL`/`EMBEDDING_DIM`
is set (or the unchanged Qwen/256 defaults).
