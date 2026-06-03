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
    # The index records the model/dim that built it; search loads that model
    # regardless of any EMBEDDING_MODEL env override (read path).
    profile, dim = resolve_index_profile(
        db_dir, env_alias=None, env_dim=None, for_write=False
    )
    store = VectorStore(db_dir, dim=dim)

    model = get_model(profile, dim)
    # encode_query applies the profile's query handling: Qwen's built-in
    # "query" prompt, a bge/e5-style instruction prefix, or plain — and always
    # encodes documents (at index time) without it.
    query_embedding = encode_query(model, profile, query)[0].tolist()

    # The caller passes an absolute folder path; the store holds paths in the
    # form chosen by to_relative (relative for same-volume, absolute for
    # cross-volume). Recomputing the filter with the same function here keeps
    # the storage form aligned so the exact-match WHERE clause hits the
    # corresponding rows.
    folder_filter = (
        to_relative(folder_path, db_dir) if folder_path else None
    )

    if mode == "hybrid":
        # The FTS index is built at index time (_finalize_index), not per query.
        results = store.hybrid_search(
            query_text=query,
            query_vector=query_embedding,
            top_k=top_k,
            folder_path=folder_filter,
            file_type=file_type,
        )
    else:
        results = store.vector_search(
            query_vector=query_embedding,
            top_k=top_k,
            folder_path=folder_filter,
            file_type=file_type,
        )

    # Resolve stored paths back to absolute for display, and tag each hit with
    # the index it came from. Stored paths are relative for same-volume entries
    # and absolute for cross-volume entries; to_absolute handles both.
    for r in results:
        r["source_file"] = to_absolute(r["source_file"], db_dir)
        r["folder_path"] = to_absolute(r["folder_path"], db_dir)
        r["index_path"] = db_dir
        r["model_alias"] = profile.alias
    return results


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
    # Imported lazily: multi_search imports search_one from this module, so a
    # top-level import here would create a cycle.
    from server.multi_search import search_indexes

    db_dirs = resolve_db_dirs(db_path)
    folder_abs = os.path.abspath(folder_path) if folder_path else None
    return search_indexes(
        query=query, db_dirs=db_dirs, top_k=top_k,
        folder_path=folder_abs, file_type=file_type, mode=mode,
    )
