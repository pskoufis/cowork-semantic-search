"""Search logic: embed query, vector/hybrid search, format results."""

import os

from server.store import VectorStore
from server.indexer import get_model
from server.paths import to_relative, to_absolute


def semantic_search(
    query: str,
    db_path: str | None = None,
    folder_path: str | None = None,
    top_k: int = 10,
    file_type: str | None = None,
    mode: str = "vector",
) -> dict:
    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    store = VectorStore(db_dir)

    model = get_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    # The caller passes an absolute folder path; the store holds relative paths.
    folder_filter = (
        to_relative(folder_path, db_dir, check_volume=False)
        if folder_path else None
    )

    if mode == "hybrid":
        store.create_fts_index()
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

    # Resolve stored relative paths back to absolute for display.
    for r in results:
        r["source_file"] = to_absolute(r["source_file"], db_dir)
        r["folder_path"] = to_absolute(r["folder_path"], db_dir)

    return {
        "query": query,
        "mode": mode,
        "results": results,
        "total_results": len(results),
    }
