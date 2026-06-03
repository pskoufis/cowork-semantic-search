# Configurable Embedding Model & Dimension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the embedding model and vector dimension be chosen per execution via env vars, recorded into each index, and read back automatically so search/status always use the model that built the index.

**Architecture:** A curated registry (`server/embedding_models.py`) maps short aliases to `EmbeddingProfile`s. A JSON sidecar (`server/index_meta.py` → `index_meta.json` in the index dir) records the model alias + dim at creation. `VectorStore` self-resolves its dim from that sidecar (legacy fallback 256), so the LanceDB schema width follows the index. `get_model()` becomes profile-aware and cached by `(model_id, dim)`. Search/status read the sidecar; env vars only matter when creating a new index.

**Tech Stack:** Python, sentence-transformers, LanceDB, pyarrow, pytest.

---

## File Structure

- `server/embedding_models.py` *(new)* — `EmbeddingProfile`, `REGISTRY`, `resolve_profile(alias, dim_override)`. Pure data + validation, no heavy imports.
- `server/index_meta.py` *(new)* — `read_meta`, `write_meta`, `resolve_index_profile`. JSON sidecar I/O + reconciliation.
- `server/store.py` *(modify)* — `EMBEDDING_DIM` stays as the legacy default (256); add `make_schema(dim)`; `VectorStore.__init__(db_path, dim=None)` self-resolves dim from the sidecar; schema/`_check_dim_compat`/`create_vector_index` use the instance dim.
- `server/indexer.py` *(modify)* — `get_model(profile, dim)` cached by key; `embed_chunks(chunks, model)`; `index_folder` resolves profile+dim and writes the sidecar.
- `server/search.py` *(modify)* — resolve profile from sidecar, encode the query per profile.
- `cli/commands.py` + `server/main.py` *(modify)* — surface model alias + dim in status.

Backward compatibility is the spine: existing `VectorStore(db_dir)` call sites (≈12 of them) keep working untouched because `dim` defaults to sidecar-or-256.

---

### Task 1: Embedding model registry

**Files:**
- Create: `server/embedding_models.py`
- Test: `tests/test_embedding_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding_models.py
import pytest

from server.embedding_models import resolve_profile, REGISTRY, EmbeddingProfile


def test_default_alias_is_qwen():
    profile, dim = resolve_profile("qwen3-0.6b", None)
    assert profile.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert dim == 256
    assert profile.padding_side == "left"
    assert profile.query_prompt_name == "query"


def test_unknown_alias_lists_available():
    with pytest.raises(ValueError) as exc:
        resolve_profile("nope", None)
    msg = str(exc.value)
    for alias in REGISTRY:
        assert alias in msg


def test_fixed_dim_model_rejects_override():
    with pytest.raises(ValueError):
        resolve_profile("bge-small", 256)  # bge is fixed at 384


def test_mrl_model_accepts_in_range_override():
    profile, dim = resolve_profile("qwen3-0.6b", 512)
    assert dim == 512


def test_mrl_model_rejects_out_of_range():
    with pytest.raises(ValueError):
        resolve_profile("qwen3-0.6b", 2048)  # above native 1024


def test_dim_must_be_divisible_by_eight():
    with pytest.raises(ValueError):
        resolve_profile("qwen3-0.6b", 100)


def test_bge_has_query_prefix_not_prompt():
    profile, _ = resolve_profile("bge-small", None)
    assert profile.query_prefix is not None
    assert profile.query_prompt_name is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'server.embedding_models'`

- [ ] **Step 3: Write minimal implementation**

```python
# server/embedding_models.py
"""Curated registry of embedding models the indexer can use.

Each EmbeddingProfile captures everything that differs between models:
the HF id, the dimension(s) it supports, and the model-specific encoding
quirks (decoder left-padding, a built-in query prompt, or an instruction
prefix). The model is chosen by a short alias via the EMBEDDING_MODEL
env var; resolve_profile() validates the alias and an optional EMBEDDING_DIM
override.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    alias: str
    model_id: str
    default_dim: int
    mrl: bool                      # Matryoshka-truncatable (dim is a range)
    min_dim: int | None            # lower bound for mrl models; else None
    padding_side: str | None       # "left" for decoder/last-token pooling
    query_prompt_name: str | None  # sentence-transformers built-in prompt name
    query_prefix: str | None       # instruction prefix; mutually exclusive w/ prompt_name
    normalize: bool = True


_PROFILES = [
    EmbeddingProfile(
        alias="qwen3-0.6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        default_dim=256,
        mrl=True,
        min_dim=64,
        padding_side="left",
        query_prompt_name="query",
        query_prefix=None,
    ),
    EmbeddingProfile(
        alias="bge-small",
        model_id="BAAI/bge-small-en-v1.5",
        default_dim=384,
        mrl=False,
        min_dim=None,
        padding_side=None,
        query_prompt_name=None,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    EmbeddingProfile(
        alias="gte-small",
        model_id="thenlper/gte-small",
        default_dim=384,
        mrl=False,
        min_dim=None,
        padding_side=None,
        query_prompt_name=None,
        query_prefix=None,
    ),
    EmbeddingProfile(
        alias="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        default_dim=384,
        mrl=False,
        min_dim=None,
        padding_side=None,
        query_prompt_name=None,
        query_prefix=None,
    ),
    EmbeddingProfile(
        alias="static-mrl",
        model_id="sentence-transformers/static-retrieval-mrl-en-v1",
        default_dim=256,
        mrl=True,
        min_dim=64,
        padding_side=None,
        query_prompt_name=None,
        query_prefix=None,
    ),
]

REGISTRY: dict[str, EmbeddingProfile] = {p.alias: p for p in _PROFILES}

DEFAULT_ALIAS = "qwen3-0.6b"


def resolve_profile(alias: str, dim_override: int | None) -> tuple[EmbeddingProfile, int]:
    """Look up a profile by alias and resolve its effective dimension.

    Raises ValueError on an unknown alias or an invalid dim override.
    """
    profile = REGISTRY.get(alias)
    if profile is None:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown embedding model alias {alias!r}. Available: {available}."
        )

    dim = dim_override if dim_override is not None else profile.default_dim

    if dim % 8 != 0:
        raise ValueError(
            f"Embedding dim {dim} must be divisible by 8 (IVF_PQ requirement)."
        )

    if profile.mrl:
        lo = profile.min_dim or 8
        if not (lo <= dim <= profile.default_dim if dim_override is None else lo <= dim <= _native_max(profile)):
            raise ValueError(
                f"{profile.alias} supports dims {lo}..{_native_max(profile)}; got {dim}."
            )
    else:
        if dim != profile.default_dim:
            raise ValueError(
                f"{profile.alias} is fixed at {profile.default_dim} dims; "
                f"cannot use {dim}."
            )

    return profile, dim


# Native (maximum) embedding width per MRL model. default_dim is the *chosen*
# default, which may be a truncation; the override ceiling is the native width.
_NATIVE_MAX = {
    "qwen3-0.6b": 1024,
    "static-mrl": 1024,
}


def _native_max(profile: EmbeddingProfile) -> int:
    return _NATIVE_MAX.get(profile.alias, profile.default_dim)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding_models.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add server/embedding_models.py tests/test_embedding_models.py
git commit -m "feat(embeddings): curated model registry with dim validation"
```

---

### Task 2: Index metadata sidecar

**Files:**
- Create: `server/index_meta.py`
- Test: `tests/test_index_meta.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_meta.py
import pytest

from server.index_meta import read_meta, write_meta, resolve_index_profile
from server.embedding_models import resolve_profile


def test_write_then_read_roundtrip(tmp_path):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)
    write_meta(db, profile, dim)
    meta = read_meta(db)
    assert meta == {
        "model_alias": "minilm",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    }


def test_read_missing_returns_none(tmp_path):
    assert read_meta(str(tmp_path / "nope")) is None


def test_resolve_existing_index_ignores_env(tmp_path):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)
    write_meta(db, profile, dim)
    # Env says qwen, but the recorded index wins for a read path.
    got_profile, got_dim = resolve_index_profile(
        db, env_alias="qwen3-0.6b", env_dim=None, for_write=False
    )
    assert got_profile.alias == "minilm"
    assert got_dim == 384


def test_resolve_write_mismatch_raises(tmp_path):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)
    write_meta(db, profile, dim)
    with pytest.raises(ValueError) as exc:
        resolve_index_profile(db, env_alias="bge-small", env_dim=None, for_write=True)
    assert "minilm" in str(exc.value) and "bge-small" in str(exc.value)


def test_resolve_legacy_fallback(tmp_path):
    db = str(tmp_path / "idx")  # no meta file
    profile, dim = resolve_index_profile(db, env_alias=None, env_dim=None, for_write=False)
    assert profile.alias == "qwen3-0.6b"
    assert dim == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_index_meta.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'server.index_meta'`

- [ ] **Step 3: Write minimal implementation**

```python
# server/index_meta.py
"""Per-index model/dim metadata sidecar.

Each index directory carries an ``index_meta.json`` recording which model
alias and dimension built it. This makes an index self-describing: search
and status load the correct model automatically, and an index/reindex op
with a conflicting EMBEDDING_MODEL is rejected instead of silently writing
incompatible vectors.
"""

import json
import os

from server.embedding_models import (
    EmbeddingProfile,
    resolve_profile,
    REGISTRY,
    DEFAULT_ALIAS,
)

META_FILENAME = "index_meta.json"


def _meta_path(db_dir: str) -> str:
    return os.path.join(db_dir, META_FILENAME)


def read_meta(db_dir: str) -> dict | None:
    """Return the recorded {model_alias, model_id, dim}, or None if absent."""
    path = _meta_path(db_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_meta(db_dir: str, profile: EmbeddingProfile, dim: int) -> None:
    """Record the model/dim for this index. Creates db_dir if needed."""
    os.makedirs(db_dir, exist_ok=True)
    payload = {
        "model_alias": profile.alias,
        "model_id": profile.model_id,
        "dim": dim,
    }
    with open(_meta_path(db_dir), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def resolve_index_profile(
    db_dir: str,
    env_alias: str | None,
    env_dim: int | None,
    for_write: bool,
) -> tuple[EmbeddingProfile, int]:
    """Resolve the (profile, dim) an index should use.

    - Existing meta is the source of truth. On a write path, a conflicting
      env config raises ValueError. On a read path, env is ignored.
    - Missing meta falls back to the legacy default (qwen3-0.6b / 256) so
      pre-existing indexes keep working unchanged.
    """
    meta = read_meta(db_dir)
    if meta is not None:
        recorded_profile = REGISTRY.get(meta["model_alias"])
        if recorded_profile is None:
            raise ValueError(
                f"Index at {db_dir!r} records unknown model "
                f"{meta['model_alias']!r}; not in the registry."
            )
        recorded_dim = int(meta["dim"])
        if for_write and env_alias is not None:
            want_profile, want_dim = resolve_profile(env_alias, env_dim)
            if want_profile.alias != recorded_profile.alias or want_dim != recorded_dim:
                raise ValueError(
                    f"Index at {db_dir!r} was built with "
                    f"{recorded_profile.alias}/{recorded_dim}; you configured "
                    f"{want_profile.alias}/{want_dim}. Point --db-path at a new "
                    f"directory or unset the EMBEDDING_MODEL/EMBEDDING_DIM override."
                )
        return recorded_profile, recorded_dim

    # No meta: legacy fallback for read; configured-or-default for write.
    alias = env_alias if env_alias is not None else DEFAULT_ALIAS
    return resolve_profile(alias, env_dim)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_index_meta.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add server/index_meta.py tests/test_index_meta.py
git commit -m "feat(embeddings): index_meta.json sidecar binds index to its model"
```

---

### Task 3: VectorStore dimension parameterization

**Files:**
- Modify: `server/store.py`
- Test: `tests/test_store.py` (add cases; existing must still pass)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_store.py
from server.index_meta import write_meta
from server.embedding_models import resolve_profile


def test_store_reads_dim_from_meta(tmp_path):
    db = str(tmp_path / "idx384")
    profile, dim = resolve_profile("minilm", None)  # 384
    write_meta(db, profile, dim)
    store = VectorStore(db)
    table = store._ensure_table()
    assert table.schema.field("vector").type.list_size == 384


def test_store_explicit_dim_overrides(tmp_path):
    store = VectorStore(str(tmp_path / "idx512"), dim=512)
    table = store._ensure_table()
    assert table.schema.field("vector").type.list_size == 512


def test_store_legacy_default_dim(tmp_path):
    store = VectorStore(str(tmp_path / "idxlegacy"))  # no meta
    table = store._ensure_table()
    assert table.schema.field("vector").type.list_size == EMBEDDING_DIM  # 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py::test_store_reads_dim_from_meta tests/test_store.py::test_store_explicit_dim_overrides -v`
Expected: FAIL — `VectorStore.__init__` takes no `dim`; schema is fixed 256.

- [ ] **Step 3: Write minimal implementation**

In `server/store.py`, change the dim/schema region. Add a `make_schema(dim)` factory and keep `EMBEDDING_DIM = 256` as the legacy default. **Keep a module-level `SCHEMA = make_schema(EMBEDDING_DIM)`** — `tests/test_store.py`, `tests/test_indexer.py`, and `tests/test_mcp_tools.py` import `SCHEMA` for dim-agnostic column-name checks, so it must stay defined:

```python
# Legacy/default embedding dimension. Indexes created before the
# configurable-model change carry no index_meta.json and are assumed to be
# this width. New indexes record their own dim in index_meta.json.
EMBEDDING_DIM = 256


def make_schema(dim: int) -> pa.Schema:
    """Build the chunks-table schema for a given vector width."""
    return pa.schema([
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
        pa.field("chunk_kind", pa.string()),
        pa.field("sheet_name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


# Legacy/default schema, kept for tests and any caller that imports SCHEMA.
SCHEMA = make_schema(EMBEDDING_DIM)
```

Update `VectorStore.__init__` (lines 73-76) to resolve and hold its dim/schema:

```python
    def __init__(self, db_path: str, dim: int | None = None):
        self._db_path = db_path
        self._db = lancedb.connect(db_path)
        self._table = None
        if dim is None:
            # Self-resolve from the index's sidecar; legacy indexes (no
            # sidecar) fall back to the historical default width.
            from server.index_meta import read_meta
            meta = read_meta(db_path)
            dim = int(meta["dim"]) if meta else EMBEDDING_DIM
        self._dim = dim
        self._schema = make_schema(dim)
```

In `_ensure_table` (line 90) use the instance schema:

```python
            self._table = self._db.create_table(TABLE_NAME, schema=self._schema)
```

In `_check_dim_compat` (lines 104-110) compare to `self._dim`:

```python
        if stored_dim != self._dim:
            raise RuntimeError(
                f"Index at {self._db_path!r} has {stored_dim}-dim vectors, "
                f"but the configured embedding model produces {self._dim}-dim "
                f"vectors. The model/dim does not match this index. Use a "
                f"different db-path or matching EMBEDDING_MODEL/EMBEDDING_DIM."
            )
```

In `create_vector_index` (line 287) use the instance dim:

```python
            num_sub_vectors=self._dim // 8,  # IVF_PQ subspace count; 8 floats per sub-vector
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS — new cases pass and all pre-existing `test_store.py` tests still pass.

- [ ] **Step 5: Commit**

```bash
git add server/store.py tests/test_store.py
git commit -m "feat(store): VectorStore resolves vector dim per index"
```

---

### Task 4: Profile-aware model loading

**Files:**
- Modify: `server/indexer.py` (lines 89-134, 190-200)
- Test: `tests/test_embedding_loading.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding_loading.py
from unittest.mock import MagicMock, patch

from server.embedding_models import resolve_profile
from server.indexer import _model_load_kwargs, encode_documents, encode_query


def test_load_kwargs_qwen_left_pads_and_truncates():
    profile, dim = resolve_profile("qwen3-0.6b", None)
    kwargs = _model_load_kwargs(profile, dim)
    assert kwargs["truncate_dim"] == 256
    assert kwargs["tokenizer_kwargs"] == {"padding_side": "left"}


def test_load_kwargs_minilm_no_padding_no_truncate():
    profile, dim = resolve_profile("minilm", None)
    kwargs = _model_load_kwargs(profile, dim)
    assert "tokenizer_kwargs" not in kwargs
    # Fixed-dim model: no MRL truncation requested.
    assert "truncate_dim" not in kwargs


def test_encode_query_uses_prompt_for_qwen():
    profile, _ = resolve_profile("qwen3-0.6b", None)
    model = MagicMock()
    model.encode.return_value = [[0.0]]
    encode_query(model, profile, "hi")
    _, kwargs = model.encode.call_args
    assert kwargs.get("prompt_name") == "query"


def test_encode_query_prepends_prefix_for_bge():
    profile, _ = resolve_profile("bge-small", None)
    model = MagicMock()
    model.encode.return_value = [[0.0]]
    encode_query(model, profile, "hi")
    args, _ = model.encode.call_args
    assert args[0] == ["Represent this sentence for searching relevant passages: hi"]


def test_encode_query_plain_for_gte():
    profile, _ = resolve_profile("gte-small", None)
    model = MagicMock()
    model.encode.return_value = [[0.0]]
    encode_query(model, profile, "hi")
    args, kwargs = model.encode.call_args
    assert args[0] == ["hi"]
    assert "prompt_name" not in kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_loading.py -v`
Expected: FAIL — `_model_load_kwargs`, `encode_documents`, `encode_query` do not exist.

- [ ] **Step 3: Write minimal implementation**

In `server/indexer.py`, replace the model block (lines 89-134). Remove the module-level `EMBEDDING_MODEL` constant and the single-slot `_model`. Add a keyed cache and profile-aware helpers:

```python
_model_cache: dict[tuple[str, int], object] = {}


def _model_load_kwargs(profile, dim: int) -> dict:
    """sentence-transformers load kwargs derived from a profile.

    MRL models truncate (and renormalise) to `dim`; fixed-dim models load at
    native width. Decoder models (Qwen) need left-padding so last-token
    pooling never pools a PAD token.
    """
    kwargs: dict = {"device": _select_device()}
    if profile.mrl:
        kwargs["truncate_dim"] = dim
    if profile.padding_side is not None:
        kwargs["tokenizer_kwargs"] = {"padding_side": profile.padding_side}
    return kwargs


def get_model(profile, dim: int):
    """Load (and cache) the SentenceTransformer for a profile/dim."""
    key = (profile.model_id, dim)
    model = _model_cache.get(key)
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(profile.model_id, **_model_load_kwargs(profile, dim))
        if profile.query_prompt_name is not None:
            if profile.query_prompt_name not in getattr(model, "prompts", {}):
                raise RuntimeError(
                    f"{profile.model_id} did not expose a "
                    f"{profile.query_prompt_name!r} prompt expected by its profile."
                )
        _model_cache[key] = model
    return model


def encode_documents(model, profile, texts: list[str]):
    """Encode passages — no query prompt/prefix, always normalised."""
    return model.encode(
        texts, show_progress_bar=False, normalize_embeddings=profile.normalize,
    )


def encode_query(model, profile, query: str):
    """Encode a single query, applying the profile's query prompt/prefix."""
    if profile.query_prompt_name is not None:
        return model.encode(
            [query],
            normalize_embeddings=profile.normalize,
            prompt_name=profile.query_prompt_name,
        )
    text = query
    if profile.query_prefix is not None:
        text = profile.query_prefix + query
    return model.encode([text], normalize_embeddings=profile.normalize)
```

Update `embed_chunks` (lines 190-200) to take the model + profile:

```python
def embed_chunks(chunks: list[dict], model, profile) -> list[dict]:
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = encode_documents(model, profile, batch)
        all_embeddings.extend(embeddings)
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["vector"] = embedding.tolist()
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding_loading.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add server/indexer.py tests/test_embedding_loading.py
git commit -m "feat(indexer): profile-aware model loading + query/doc encoding"
```

---

### Task 5: Wire profile/dim through index_folder and reindex

**Files:**
- Modify: `server/indexer.py` (`index_folder` ~344-388, `reindex_one_file` ~203-, and all `embed_chunks(`/`VectorStore(` call sites within)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_indexer.py (top: import os, from unittest.mock import patch)
import os
from pathlib import Path

from server.indexer import index_folder
from server.index_meta import read_meta


def test_index_folder_writes_meta_and_respects_env(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world from minilm")
    db = str(tmp_path / "idx")
    monkeypatch.setenv("EMBEDDING_MODEL", "minilm")
    index_folder(str(corpus), db_path=db, unpack_first=False)
    meta = read_meta(db)
    assert meta["model_alias"] == "minilm"
    assert meta["dim"] == 384
```

(This test actually loads the `minilm` model — small, ~80 MB — and exercises the full path. If model downloads are undesirable in CI, mark with `@pytest.mark.slow`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indexer.py::test_index_folder_writes_meta_and_respects_env -v`
Expected: FAIL — env is ignored, no meta written, `embed_chunks`/`get_model` signatures mismatch.

- [ ] **Step 3: Write minimal implementation**

Add a resolver helper near the top of `index_folder`'s body (after `db_dir` is computed, ~line 362). Read env once:

```python
    from server.index_meta import resolve_index_profile, write_meta
    env_alias = os.environ.get("EMBEDDING_MODEL")
    _env_dim_raw = os.environ.get("EMBEDDING_DIM")
    env_dim = int(_env_dim_raw) if _env_dim_raw else None
    profile, dim = resolve_index_profile(
        db_dir, env_alias=env_alias, env_dim=env_dim, for_write=True
    )
```

Change `store = VectorStore(db_dir)` (line 381) to pass the dim, then persist meta after the store exists:

```python
    store = VectorStore(db_dir, dim=dim)
    store.ensure_schema()
    write_meta(db_dir, profile, dim)
```

Load the model once and thread it + profile into every `embed_chunks` call inside `index_folder`:

```python
    model = get_model(profile, dim)
```

Replace each `embed_chunks(<chunks>)` in `index_folder` with `embed_chunks(<chunks>, model, profile)`.

Apply the identical resolver + `VectorStore(db_dir, dim=dim)` + `get_model(profile, dim)` + `embed_chunks(..., model, profile)` wiring inside `reindex_one_file` (it independently builds a store and embeds — same four changes). Use `for_write=True` there too.

Find every call site to update:

Run: `grep -n "embed_chunks(\|VectorStore(db_dir)\|get_model(" server/indexer.py`
Update each `embed_chunks(`/`VectorStore(db_dir)`/`get_model(` inside `index_folder` and `reindex_one_file`.

**Also wire the spreadsheet/file-description write paths in `server/main.py`** — `submit_description` (`embed_chunks` at ~556) and `_auto_drain_descriptions` (`embed_chunks` at ~717) both embed text and write vectors into the same `chunks` table. These append to an **existing** index, so they must use the index's *recorded* model (env ignored, `for_write=False`); otherwise a non-default index would get wrong-dim description vectors. In each function, before the embed:

```python
from server.index_meta import resolve_index_profile
from server.indexer import get_model
profile, dim = resolve_index_profile(db_dir, env_alias=None, env_dim=None, for_write=False)
store = VectorStore(db_dir, dim=dim)
model = get_model(profile, dim)
```

and change `embed_chunks([chunk])` → `embed_chunks([chunk], model, profile)` (submit) and `embed_chunks(chunks)` → `embed_chunks(chunks, model, profile)` (drain).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_indexer.py -v`
Expected: PASS — new test passes; existing indexer tests still pass (they index with no env → qwen/256 default, unchanged behaviour).

- [ ] **Step 5: Commit**

```bash
git add server/indexer.py tests/test_indexer.py
git commit -m "feat(indexer): resolve model/dim per run and record it in the index"
```

---

### Task 6: Search uses the index's recorded model

**Files:**
- Modify: `server/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_search.py
from unittest.mock import patch

from server.index_meta import write_meta
from server.embedding_models import resolve_profile


def test_search_loads_model_from_meta_ignoring_env(tmp_path, monkeypatch):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)  # 384-dim index
    write_meta(db, profile, dim)
    store = VectorStore(db, dim=384)
    _seed_store(store, ["alpha doc", "beta doc"])  # uses EMBEDDING_DIM? -> see note

    captured = {}

    def fake_get_model(p, d):
        captured["alias"] = p.alias
        captured["dim"] = d
        m = type("M", (), {})()
        m.encode = lambda texts, **kw: _fake_embed(texts)
        return m

    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-0.6b")  # should be ignored
    with patch("server.search.get_model", fake_get_model):
        semantic_search("alpha", db_path=db)
    assert captured["alias"] == "minilm"
    assert captured["dim"] == 384
```

Note: `_fake_embed` in `test_search.py` is keyed to `EMBEDDING_DIM` (256). For this 384-dim test, add a local 384 fake-embed (`rng.randn(384)`) and seed with it so vectors match the store width. Keep the assertion on which profile/dim `get_model` received — that is the behaviour under test.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py::test_search_loads_model_from_meta_ignoring_env -v`
Expected: FAIL — `semantic_search` calls `get_model()` with no args and ignores the sidecar.

- [ ] **Step 3: Write minimal implementation**

In `server/search.py`, resolve the profile from the sidecar and encode via the profile. Replace lines 5-30:

```python
from server.store import VectorStore
from server.indexer import get_model, encode_query
from server.index_meta import resolve_index_profile
from server.paths import to_relative, to_absolute


def semantic_search(
    query: str,
    db_path: str | None = None,
    folder_path: str | None = None,
    top_k: int = 25,
    file_type: str | None = None,
    mode: str = "vector",
) -> dict:
    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    # The index records the model/dim that built it; search loads that model
    # regardless of any EMBEDDING_MODEL env override (read path).
    profile, dim = resolve_index_profile(
        db_dir, env_alias=None, env_dim=None, for_write=False
    )
    store = VectorStore(db_dir, dim=dim)

    model = get_model(profile, dim)
    query_embedding = encode_query(model, profile, query)[0].tolist()
```

(The rest of `semantic_search` — folder filter, hybrid/vector branch, path resolution, return — is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS — new test passes; existing search tests still pass (legacy fallback → qwen/256, unchanged).

- [ ] **Step 5: Commit**

```bash
git add server/search.py tests/test_search.py
git commit -m "feat(search): load the index's recorded model, ignore env at query time"
```

---

### Task 7: Surface model/dim in status

**Files:**
- Modify: `cli/commands.py` (status printer ~61-81), `server/main.py` (`get_index_status` ~316-335)
- Test: `tests/test_mcp_tools.py` (status shape) or `tests/cli`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_mcp_tools.py (or appropriate status test module)
from server.index_meta import write_meta
from server.embedding_models import resolve_profile
from server.main import get_index_status


def test_status_reports_model_and_dim(tmp_path):
    db = str(tmp_path / "idx")
    profile, dim = resolve_profile("minilm", None)
    write_meta(db, profile, dim)
    status = get_index_status(db_path=db)
    assert status["model_alias"] == "minilm"
    assert status["dim"] == 384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test_status_reports_model_and_dim -v`
Expected: FAIL — `get_index_status` returns no `model_alias`/`dim`.

- [ ] **Step 3: Write minimal implementation**

In `server/main.py` `get_index_status`, after `db_dir` is set, read the sidecar and add to the output. Insert before the `try:` (around line 322):

```python
    from server.index_meta import read_meta
    _meta = read_meta(db_dir)
    stats["model_alias"] = _meta["model_alias"] if _meta else None
    stats["dim"] = _meta["dim"] if _meta else None
```

(Because `out = {**stats, ...}`, these flow through to the response.)

In `cli/commands.py` status printer, after the `total files` line (~81), print the model when known:

```python
    from server.index_meta import read_meta
    _meta = read_meta(db_path)
    if _meta:
        print(f"  model:        {_meta['model_alias']} ({_meta['dim']}-dim)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcp_tools.py::test_status_reports_model_and_dim -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/main.py cli/commands.py tests/test_mcp_tools.py
git commit -m "feat(status): report the index's embedding model and dim"
```

---

### Task 8: Full suite + docs

**Files:**
- Modify: `README.md` (embedding model env section)

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass (excluding any pre-existing unrelated failures; note them if present).

- [ ] **Step 2: Document the env vars**

Add a short README subsection under the configuration/env area:

```markdown
### Choosing an embedding model

Set `EMBEDDING_MODEL` (and optionally `EMBEDDING_DIM`) when **creating** an
index to pick the model. Available aliases: `qwen3-0.6b` (default),
`bge-small`, `gte-small`, `minilm`, `static-mrl`. Each index records its
model in `index_meta.json`; search and status use that automatically, so you
only set these at index time. Use a separate `--db-path` / `LANCEDB_PATH`
per model — an index is permanently bound to the model that built it.

    EMBEDDING_MODEL=bge-small csemsearch --db-path ./idx-bge index ./corpus --types .txt
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document EMBEDDING_MODEL / EMBEDDING_DIM configuration"
```

---

## Self-Review Notes

- **Spec coverage:** registry (Task 1), sidecar + reconciliation/legacy fallback (Task 2), dim-parameterized store + IVF_PQ (Task 3), profile-aware load + per-model encoding (Task 4), index-time resolution + meta write + reindex (Task 5), search-time auto-load (Task 6), status surfacing (Task 7), docs (Task 8). All spec sections map to a task.
- **Backward compat:** `EMBEDDING_DIM = 256` kept as the legacy default; all existing `VectorStore(db_dir)` call sites untouched; no-env indexing reproduces today's qwen/256 behaviour.
- **Type consistency:** `resolve_profile(alias, dim_override) -> (EmbeddingProfile, int)`; `resolve_index_profile(db_dir, env_alias, env_dim, for_write) -> (EmbeddingProfile, int)`; `get_model(profile, dim)`; `embed_chunks(chunks, model, profile)`; `encode_query(model, profile, query)`; `encode_documents(model, profile, texts)`. Used consistently across Tasks 4-7.
