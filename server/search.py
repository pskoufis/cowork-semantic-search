"""Search logic: embed query, vector/hybrid search, format results."""

import os

from server.store import VectorStore
from server.indexer import get_model
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

    store = VectorStore(db_dir)

    model = get_model()
    # Qwen3-Embedding ships a built-in "query" prompt for retrieval — wraps the
    # query in an "Instruct: ... Query: ..." template and yields a 1–5% lift
    # vs encoding plain. Documents are still encoded without a prompt.
    query_embedding = model.encode(
        [query], normalize_embeddings=True, prompt_name="query",
    )[0].tolist()

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

    # Resolve stored paths back to absolute for display. Stored paths are
    # relative for same-volume entries and absolute for cross-volume entries;
    # to_absolute handles both.
    for r in results:
        r["source_file"] = to_absolute(r["source_file"], db_dir)
        r["folder_path"] = to_absolute(r["folder_path"], db_dir)

    return {
        "query": query,
        "mode": mode,
        "results": results,
        "total_results": len(results),
    }
