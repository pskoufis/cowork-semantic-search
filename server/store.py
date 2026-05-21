"""LanceDB vector store abstraction."""

import os
from datetime import timedelta
from pathlib import Path

import lancedb
import pyarrow as pa

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("file_name", pa.string()),
    pa.field("file_type", pa.string()),
    pa.field("folder_path", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("content_hash", pa.string()),
    # st_mtime_ns + st_size — a cheap stat pre-check that avoids re-hashing
    # unchanged files. int64 (not float st_mtime) for exact equality.
    pa.field("mtime_ns", pa.int64()),
    pa.field("file_size", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), 384)),
])

TABLE_NAME = "chunks"

# Minimum row count before an ANN index is worth building. Below this a flat
# scan is already sub-millisecond and IVF_PQ has too few rows to train a useful
# codebook; above it is the regime an ANN index exists for.
VECTOR_INDEX_MIN_ROWS = 20_000


def _escape(value: str) -> str:
    """Double single quotes for safe interpolation into LanceDB where-clauses."""
    return value.replace("'", "''")


class VectorStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db = lancedb.connect(db_path)
        self._table = None

    def _get_table(self):
        if self._table is None:
            try:
                self._table = self._db.open_table(TABLE_NAME)
            except Exception:
                return None
        return self._table

    def _ensure_table(self):
        table = self._get_table()
        if table is None:
            self._table = self._db.create_table(TABLE_NAME, schema=SCHEMA)
        return self._table

    def ensure_schema(self) -> None:
        """Migrate a pre-Tier-2 index in place by adding any missing columns.

        Adds mtime_ns / file_size (backfilled NULL) to a table that predates
        them. Idempotent. Called only on write paths; the single-writer
        invariant guarantees no two callers race this migration.
        """
        table = self._get_table()
        if table is None:
            return  # fresh DB: _ensure_table creates with the current SCHEMA
        existing = set(table.schema.names)
        missing = [
            field for field in (
                pa.field("mtime_ns", pa.int64()),
                pa.field("file_size", pa.int64()),
            )
            if field.name not in existing
        ]
        if missing:
            table.add_columns(missing)
            self._table = self._db.open_table(TABLE_NAME)  # refresh stale handle

    def add_chunks(self, chunks: list[dict]) -> None:
        table = self._ensure_table()
        rows = []
        for c in chunks:
            rows.append({
                "id": c["id"],
                "text": c["text"],
                "source_file": c["source_file"],
                "file_name": c["file_name"],
                "file_type": c["file_type"],
                "folder_path": c["folder_path"],
                "chunk_index": c["chunk_index"],
                "content_hash": c["content_hash"],
                "mtime_ns": c["mtime_ns"],
                "file_size": c["file_size"],
                "vector": c["vector"],
            })
        table.add(rows)

    def count_chunks(self) -> int:
        table = self._get_table()
        if table is None:
            return 0
        return table.count_rows()

    def db_size_bytes(self) -> int:
        """Total size on disk of the LanceDB directory — chunk text, vectors,
        FTS/ANN indexes and version history. This can exceed the source corpus.
        Returns 0 if the directory does not exist yet."""
        total = 0
        for root, _dirs, files in os.walk(self._db_path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass  # a file vanished mid-walk (compaction) — skip it
        return total

    def get_file_hash(self, source_file: str) -> str | None:
        table = self._get_table()
        if table is None:
            return None
        results = (
            table.search()
            .where(f"source_file = '{_escape(source_file)}'", prefilter=True)
            .select(["content_hash"])
            .limit(1)
            .to_list()
        )
        if results:
            return results[0]["content_hash"]
        return None

    def get_file_index(self) -> dict[str, dict]:
        """Bulk-load {source_file: {content_hash, mtime_ns, file_size}} in one
        projected scan — replaces a per-file get_file_hash query during indexing.
        """
        table = self._get_table()
        if table is None:
            return {}
        n = table.count_rows()
        if n == 0:
            return {}
        # Projected scan: select() reads only these columns, not the vectors.
        rows = (
            table.search()
            .select(["source_file", "content_hash", "mtime_ns", "file_size"])
            .limit(n)
            .to_list()
        )
        index: dict[str, dict] = {}
        for r in rows:
            index[r["source_file"]] = {
                "content_hash": r["content_hash"],
                "mtime_ns": r["mtime_ns"],
                "file_size": r["file_size"],
            }
        return index

    def delete_by_file(self, source_file: str) -> None:
        table = self._get_table()
        if table is None:
            return
        table.delete(f"source_file = '{_escape(source_file)}'")

    def update_file_stat(self, source_file: str, mtime_ns: int, file_size: int) -> None:
        """Refresh the stored mtime/size for a file whose content is unchanged
        (e.g. after a drive move) so later runs hit the cheap fast-path."""
        table = self._get_table()
        if table is None:
            return
        table.update(
            where=f"source_file = '{_escape(source_file)}'",
            values={"mtime_ns": mtime_ns, "file_size": file_size},
        )

    def get_all_files(self) -> list[str]:
        table = self._get_table()
        if table is None:
            return []
        n = table.count_rows()
        if n == 0:
            return []
        # Projected scan: select() reads only source_file, never the vectors.
        rows = table.search().select(["source_file"]).limit(n).to_list()
        return list({r["source_file"] for r in rows})

    def create_fts_index(self) -> None:
        """Create or rebuild the full-text search index on the text column."""
        table = self._get_table()
        if table is None:
            return
        table.create_fts_index("text", replace=True)

    def optimize_table(self) -> None:
        """Compact small fragments and drop superseded versions.

        cleanup_older_than=0 removes all old versions immediately; safe because
        the single-writer invariant guarantees exclusive write access.
        """
        table = self._get_table()
        if table is None:
            return
        table.optimize(cleanup_older_than=timedelta(seconds=0))

    def create_vector_index(self) -> None:
        """Build (or rebuild) an IVF_PQ ANN index on the vector column.

        No-ops while the table is small (below VECTOR_INDEX_MIN_ROWS rows):
        a flat scan is already fast there and IVF_PQ cannot train. Above the
        threshold the index is built with replace=True, so each call supersedes
        the previous index as the table grows.
        """
        table = self._get_table()
        if table is None:
            return
        n = table.count_rows()
        if n < VECTOR_INDEX_MIN_ROWS:
            return
        table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_PQ",
            num_partitions=max(1, n // 4096),
            num_sub_vectors=48,  # embedding dimension 384 / 8
            replace=True,
        )

    def fts_search(
        self,
        query_text: str,
        top_k: int = 25,
        folder_path: str | None = None,
        file_type: str | None = None,
    ) -> list[dict]:
        table = self._get_table()
        if table is None:
            return []
        if table.count_rows() == 0:
            return []

        try:
            query = table.search(query_text, query_type="fts").limit(top_k)
        except Exception:
            return []

        where_clauses = []
        if folder_path:
            where_clauses.append(f"folder_path = '{_escape(folder_path)}'")
        if file_type:
            where_clauses.append(f"file_type = '{_escape(file_type)}'")
        if where_clauses:
            query = query.where(" AND ".join(where_clauses), prefilter=True)

        results = query.to_list()
        return [
            {
                "text": r["text"],
                "source_file": r["source_file"],
                "file_name": r["file_name"],
                "file_type": r["file_type"],
                "folder_path": r["folder_path"],
                "chunk_index": r["chunk_index"],
                "score": r.get("_score", 0.0),
            }
            for r in results
        ]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 25,
        folder_path: str | None = None,
        file_type: str | None = None,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Combine vector and FTS results using Reciprocal Rank Fusion."""
        vector_results = self.vector_search(
            query_vector, top_k=top_k * 2,
            folder_path=folder_path, file_type=file_type,
        )
        fts_results = self.fts_search(
            query_text, top_k=top_k * 2,
            folder_path=folder_path, file_type=file_type,
        )

        # Build RRF scores keyed by (source_file, chunk_index)
        scores: dict[tuple, float] = {}
        result_map: dict[tuple, dict] = {}

        for rank, r in enumerate(vector_results):
            key = (r["source_file"], r["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            result_map[key] = r

        for rank, r in enumerate(fts_results):
            key = (r["source_file"], r["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            if key not in result_map:
                result_map[key] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**result_map[key], "rrf_score": score, "score": score}
            for key, score in ranked
        ]

    def vector_search(
        self,
        query_vector: list[float],
        top_k: int = 25,
        folder_path: str | None = None,
        file_type: str | None = None,
    ) -> list[dict]:
        table = self._get_table()
        if table is None:
            return []
        if table.count_rows() == 0:
            return []

        query = table.search(query_vector).metric("cosine").limit(top_k)

        where_clauses = []
        if folder_path:
            where_clauses.append(f"folder_path = '{_escape(folder_path)}'")
        if file_type:
            where_clauses.append(f"file_type = '{_escape(file_type)}'")
        if where_clauses:
            query = query.where(" AND ".join(where_clauses), prefilter=True)

        results = query.to_list()
        return [
            {
                "text": r["text"],
                "source_file": r["source_file"],
                "file_name": r["file_name"],
                "file_type": r["file_type"],
                "folder_path": r["folder_path"],
                "chunk_index": r["chunk_index"],
                "score": r.get("_distance", 0.0),
            }
            for r in results
        ]
