"""LanceDB vector store abstraction."""

import os
from datetime import timedelta
from pathlib import Path

import lancedb
import pyarrow as pa

# Legacy/default embedding dimension. Indexes created before the
# configurable-model change carry no index_meta.json and are assumed to be
# this width; new indexes record their own dim in index_meta.json and a
# VectorStore self-resolves to it. The schema's FixedSizeList width must match
# whatever the index in question uses, so the schema is built per-dim below.
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
        # st_mtime_ns + st_size — a cheap stat pre-check that avoids re-hashing
        # unchanged files. int64 (not float st_mtime) for exact equality.
        pa.field("mtime_ns", pa.int64()),
        pa.field("file_size", pa.int64()),
        # chunk_kind: "text" (default, NULL on legacy rows), "sheet_description",
        # or "file_description". sheet_name is non-null only on sheet_description
        # chunks. Both nullable so the in-place migration can backfill with NULL.
        pa.field("chunk_kind", pa.string()),
        pa.field("sheet_name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


# Legacy/default schema, kept for tests and any caller that imports SCHEMA.
SCHEMA = make_schema(EMBEDDING_DIM)

TABLE_NAME = "chunks"

# Minimum row count before an ANN index is worth building. Below this a flat
# scan is already sub-millisecond and IVF_PQ has too few rows to train a useful
# codebook; above it is the regime an ANN index exists for.
VECTOR_INDEX_MIN_ROWS = 20_000

# Growth multiplier that triggers a full IVF retrain. Between rebuilds, new
# rows are folded into the existing index by optimize_table() — cheap and
# incremental. A full retrain (which recomputes the partition count, frozen at
# build time) only happens once the table has grown this many times larger than
# at the last full build, so routine runs never pay an O(n) rebuild.
VECTOR_INDEX_RETRAIN_GROWTH = 4

# Spreadsheet descriptions awaiting LLM-generated text. Stored in the same
# LanceDB as `chunks` but as a separate table — no vectors, pure metadata.
PENDING_TABLE_NAME = "pending_descriptions"

PENDING_SCHEMA = pa.schema([
    pa.field("file_path", pa.string()),
    pa.field("needs", pa.list_(pa.string())),
    pa.field("preview_json", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("enqueued_at_ns", pa.int64()),
])

# A file the LLM has explicitly declined to describe. The content_hash is
# captured at dismissal time so that if the file later changes, the
# dismissal becomes stale and the file is re-enqueued.
DISMISSED_TABLE_NAME = "dismissed_files"

DISMISSED_SCHEMA = pa.schema([
    pa.field("file_path", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("dismissed_at_ns", pa.int64()),
])


def _escape(value: str) -> str:
    """Double single quotes for safe interpolation into LanceDB where-clauses."""
    return value.replace("'", "''")


class VectorStore:
    def __init__(self, db_path: str, dim: int | None = None):
        self._db_path = db_path
        self._db = lancedb.connect(db_path)
        self._table = None
        if dim is None:
            # Self-resolve from the index's sidecar; legacy indexes (no
            # sidecar) fall back to the historical default width.
            from server.index_meta import read_meta
            meta = read_meta(db_path)
            dim = int(meta["dim"]) if meta and meta.get("dim") is not None else EMBEDDING_DIM
        self._dim = dim
        self._schema = make_schema(dim)

    def _get_table(self):
        if self._table is None:
            try:
                self._table = self._db.open_table(TABLE_NAME)
            except Exception:
                return None
            self._check_dim_compat(self._table)
        return self._table

    def _ensure_table(self):
        table = self._get_table()
        if table is None:
            self._table = self._db.create_table(TABLE_NAME, schema=self._schema)
        return self._table

    def _check_dim_compat(self, table) -> None:
        """Fail fast when the on-disk vector dim doesn't match the embedding
        model in use. A 256-dim add()/search() against a 384-dim table would
        otherwise die with a cryptic Arrow error somewhere downstream — this
        gives the user a clear next step instead.
        """
        try:
            vector_field = table.schema.field("vector")
            stored_dim = vector_field.type.list_size
        except Exception:
            return  # nothing we can do; let the underlying error surface
        if stored_dim != self._dim:
            raise RuntimeError(
                f"Index at {self._db_path!r} has {stored_dim}-dim vectors, "
                f"but the configured embedding model produces {self._dim}-dim "
                f"vectors. The model/dim does not match this index. Use a "
                f"different db-path or matching EMBEDDING_MODEL/EMBEDDING_DIM."
            )

    def ensure_schema(self) -> None:
        """Migrate an older index in place by adding any missing columns.

        Adds mtime_ns / file_size (Tier-2) and chunk_kind / sheet_name
        (description support) to a table that predates them. All backfill to
        NULL. Idempotent. Called only on write paths; the single-writer
        invariant guarantees no two callers race this migration.
        """
        table = self._get_table()
        if table is None:
            return  # fresh DB: _ensure_table creates with the current SCHEMA
        existing = set(table.schema.names)
        # LanceDB add_columns takes {col_name: sql_expression}; CAST(NULL AS T)
        # is the cheapest backfill. Application code treats NULL chunk_kind as
        # equivalent to "text" downstream.
        defaults = {
            "mtime_ns": "CAST(NULL AS BIGINT)",
            "file_size": "CAST(NULL AS BIGINT)",
            "chunk_kind": "CAST(NULL AS STRING)",
            "sheet_name": "CAST(NULL AS STRING)",
        }
        missing = {
            name: expr for name, expr in defaults.items()
            if name not in existing
        }
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
                "chunk_kind": c.get("chunk_kind", "text"),
                "sheet_name": c.get("sheet_name"),
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

    def _has_index_on(self, column: str) -> bool:
        """True if any index covers ``column``. Used to build each index once
        and then leave it to optimize_table() to maintain incrementally."""
        table = self._get_table()
        if table is None:
            return False
        try:
            return any(column in (idx.columns or []) for idx in table.list_indices())
        except Exception:
            return False

    def create_fts_index(self) -> None:
        """Build the native FTS index once, on the text column.

        No-op if an FTS index already exists: new rows are folded into it
        incrementally by optimize_table(), so routine runs never rebuild it.
        Native FTS (use_tantivy=False) is what supports that incremental
        maintenance — a Tantivy index would not.
        """
        table = self._get_table()
        if table is None:
            return
        if self._has_index_on("text"):
            return
        table.create_fts_index("text", use_tantivy=False, replace=True)

    def optimize_table(self) -> None:
        """Compact small fragments and drop superseded versions.

        cleanup_older_than=0 removes all old versions immediately; safe because
        the single-writer invariant guarantees exclusive write access.
        """
        table = self._get_table()
        if table is None:
            return
        table.optimize(cleanup_older_than=timedelta(seconds=0))

    def _build_vector_index(self, table, n: int) -> None:
        """Do the full IVF_PQ (re)build and record the row count it trained on."""
        from server.index_meta import write_vector_index_rows
        table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_PQ",
            num_partitions=max(1, n // 4096),
            num_sub_vectors=self._dim // 8,  # IVF_PQ subspace count; 8 floats per sub-vector
            replace=True,
        )
        write_vector_index_rows(self._db_path, n)

    def create_vector_index(self) -> None:
        """Build the IVF_PQ ANN index, retraining only after substantial growth.

        Incremental by default: once an index exists, new rows are folded into
        it by optimize_table(), so routine runs never pay an O(n) rebuild. A
        full (re)build happens only when:

          * no vector index exists yet and the table has crossed
            VECTOR_INDEX_MIN_ROWS (the first build), or
          * the table has grown >= VECTOR_INDEX_RETRAIN_GROWTH x since the last
            full build — IVF's partition count is frozen at build time, so a
            retrain recomputes it for the larger corpus and keeps search fast.

        No-ops below VECTOR_INDEX_MIN_ROWS: a flat scan is already fast there
        and IVF_PQ cannot train a useful codebook on so few rows.
        """
        from server.index_meta import read_vector_index_rows
        table = self._get_table()
        if table is None:
            return
        n = table.count_rows()
        if n < VECTOR_INDEX_MIN_ROWS:
            return
        if not self._has_index_on("vector"):
            self._build_vector_index(table, n)
            return
        last = read_vector_index_rows(self._db_path)
        if last is None or n >= last * VECTOR_INDEX_RETRAIN_GROWTH:
            self._build_vector_index(table, n)

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

    # --- pending_descriptions table ---------------------------------------

    def _get_pending_table(self):
        try:
            return self._db.open_table(PENDING_TABLE_NAME)
        except Exception:
            return None

    def _ensure_pending_table(self):
        table = self._get_pending_table()
        if table is None:
            table = self._db.create_table(
                PENDING_TABLE_NAME, schema=PENDING_SCHEMA
            )
        return table

    def enqueue_pending(
        self,
        file_path: str,
        needs: list[str],
        preview_json: str,
        content_hash: str,
    ) -> None:
        """Idempotent: any existing entry for this path is replaced."""
        import time
        table = self._ensure_pending_table()
        table.delete(f"file_path = '{_escape(file_path)}'")
        table.add([{
            "file_path": file_path,
            "needs": needs,
            "preview_json": preview_json,
            "content_hash": content_hash,
            "enqueued_at_ns": time.time_ns(),
        }])

    def list_pending(
        self, folder_path: str | None = None, limit: int = 20,
    ) -> list[dict]:
        table = self._get_pending_table()
        if table is None or table.count_rows() == 0:
            return []
        query = table.search().limit(limit)
        if folder_path is not None:
            query = query.where(
                f"file_path LIKE '{_escape(folder_path)}%'", prefilter=True,
            )
        return [
            {
                "file_path": r["file_path"],
                "needs": list(r["needs"]),
                "preview_json": r["preview_json"],
                "content_hash": r["content_hash"],
            }
            for r in query.to_list()
        ]

    def remove_pending(self, file_path: str) -> None:
        table = self._get_pending_table()
        if table is None:
            return
        table.delete(f"file_path = '{_escape(file_path)}'")

    def update_pending_needs(self, file_path: str, new_needs: list[str]) -> None:
        """Rewrite an entry's `needs` list.

        LanceDB doesn't support updating a list-typed column in place, so
        we delete+reinsert, preserving the rest of the row."""
        table = self._get_pending_table()
        if table is None:
            return
        existing = (
            table.search()
            .where(f"file_path = '{_escape(file_path)}'", prefilter=True)
            .limit(1)
            .to_list()
        )
        if not existing:
            return
        row = existing[0]
        table.delete(f"file_path = '{_escape(file_path)}'")
        table.add([{
            "file_path": row["file_path"],
            "needs": new_needs,
            "preview_json": row["preview_json"],
            "content_hash": row["content_hash"],
            "enqueued_at_ns": row["enqueued_at_ns"],
        }])

    def pending_count(self) -> int:
        table = self._get_pending_table()
        if table is None:
            return 0
        return table.count_rows()

    def get_pending_entry(self, file_path: str) -> dict | None:
        table = self._get_pending_table()
        if table is None:
            return None
        rows = (
            table.search()
            .where(f"file_path = '{_escape(file_path)}'", prefilter=True)
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "file_path": r["file_path"],
            "needs": list(r["needs"]),
            "preview_json": r["preview_json"],
            "content_hash": r["content_hash"],
        }

    # --- dismissed_files table --------------------------------------------

    def _get_dismissed_table(self):
        try:
            return self._db.open_table(DISMISSED_TABLE_NAME)
        except Exception:
            return None

    def _ensure_dismissed_table(self):
        table = self._get_dismissed_table()
        if table is None:
            table = self._db.create_table(
                DISMISSED_TABLE_NAME, schema=DISMISSED_SCHEMA
            )
        return table

    def dismiss(self, file_path: str, content_hash: str) -> None:
        """Record a dismissal. Replaces any prior dismissal of the same path."""
        import time
        table = self._ensure_dismissed_table()
        table.delete(f"file_path = '{_escape(file_path)}'")
        table.add([{
            "file_path": file_path,
            "content_hash": content_hash,
            "dismissed_at_ns": time.time_ns(),
        }])

    def is_dismissed(self, file_path: str, content_hash: str) -> bool:
        table = self._get_dismissed_table()
        if table is None:
            return False
        rows = (
            table.search()
            .where(
                f"file_path = '{_escape(file_path)}' AND "
                f"content_hash = '{_escape(content_hash)}'",
                prefilter=True,
            )
            .limit(1)
            .to_list()
        )
        return bool(rows)

    def clear_dismissal(self, file_path: str) -> None:
        table = self._get_dismissed_table()
        if table is None:
            return
        table.delete(f"file_path = '{_escape(file_path)}'")

    # --- one-shot legacy eviction -----------------------------------------

    def evict_legacy_spreadsheet_chunks(self) -> set[str]:
        """Delete any CSV/XLSX chunks stored under the legacy raw-row scheme.

        Legacy = chunk_kind IS NULL OR chunk_kind = 'text'. Description
        chunks (sheet_description, file_description) are left alone.

        Returns the set of source_file paths whose chunks were evicted so
        the caller can force re-processing on the next pass even when the
        content hash hasn't changed.
        """
        table = self._get_table()
        if table is None:
            return set()
        n = table.count_rows()
        if n == 0:
            return set()
        legacy_predicate = (
            "(file_type = '.csv' OR file_type = '.xlsx') AND "
            "(chunk_kind IS NULL OR chunk_kind = 'text')"
        )
        rows = (
            table.search()
            .where(legacy_predicate, prefilter=True)
            .select(["source_file"])
            .limit(n)
            .to_list()
        )
        affected = {r["source_file"] for r in rows}
        if affected:
            table.delete(legacy_predicate)
        return affected
