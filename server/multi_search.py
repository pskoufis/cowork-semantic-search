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
        # Keep the native per-index `score` (cosine distance / fts score) intact
        # so single-index output is byte-for-byte as before; expose the fused
        # rank score separately. RRF is monotonic in rank, so sorting by
        # rrf_score reproduces the single-index order exactly.
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
