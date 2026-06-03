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
    try:
        with open(_meta_path(db_dir), "r", encoding="utf-8") as fh:
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
