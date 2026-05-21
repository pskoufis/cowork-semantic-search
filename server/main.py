"""FastMCP server entry point with tool definitions."""

import asyncio
import os
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("Semantic Search")


async def _run_index_job(job, file_types, recursive) -> None:
    """Run the synchronous index_folder in a worker thread, recording progress
    and the final outcome on the job record.

    The CPU-bound embedding work runs off the event loop via asyncio.to_thread.
    Any exception is captured onto the job so it never silently hangs in the
    'running' state.
    """
    from server.jobs import registry
    from server.indexer import index_folder as _index_folder

    def progress(processed: int, total: int) -> None:
        registry.update_progress(job.job_id, processed, total)

    try:
        result = await asyncio.to_thread(
            _index_folder,
            job.folder_path,
            file_types,
            recursive,
            job.db_path,
            progress,
        )
        registry.mark_completed(
            job.job_id, result, result.get("finalize_warnings", [])
        )
    except asyncio.CancelledError:
        registry.mark_failed(job.job_id, "cancelled (server shutting down)")
        raise
    except Exception as e:
        registry.mark_failed(job.job_id, str(e))


@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}
)
async def index_folder(
    folder_path: Annotated[str, Field(description="Absolute path to the folder to index")],
    file_types: Annotated[
        list[str] | None,
        Field(
            description="File extensions to index, e.g. ['.pdf', '.md']. "
                        "Defaults to all supported types: .txt, .md, .pdf, .docx, .pptx, .csv",
            default=None,
        ),
    ] = None,
    recursive: Annotated[
        bool,
        Field(description="Whether to index subdirectories recursively", default=True),
    ] = True,
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Start a background job to index or re-index all documents in a folder.

    Scans the folder for supported document types (.txt, .md, .pdf, .docx,
    .pptx, .csv), extracts text, splits into chunks, computes embeddings, stores
    them in a local vector database, and builds an ANN index for fast search.
    Only files that have changed since the last run are re-processed.

    Indexing runs in the background: this call returns immediately with a
    job_id. Poll get_index_status to follow progress and read the final result.
    Only one indexing job runs at a time — a call made while another job is
    running is rejected. Job state is in-memory and does not survive a server
    restart (re-running is cheap: unchanged files are skipped).
    """
    from server.jobs import registry

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    if registry.has_running():
        active = registry.active()
        return {
            "status": "rejected",
            "message": "An indexing job is already running. "
                       "Poll get_index_status for progress.",
            "running_job_id": active[0].job_id if active else None,
        }

    job = registry.create(folder_path, db_dir)
    job.task = asyncio.create_task(_run_index_job(job, file_types, recursive))
    return {
        "status": "started",
        "job_id": job.job_id,
        "folder_path": folder_path,
        "message": "Indexing started in the background. "
                   "Poll get_index_status for progress.",
    }


@mcp.tool(
    annotations={"readOnlyHint": True}
)
def semantic_search(
    query: Annotated[str, Field(description="Natural language search query")],
    folder_path: Annotated[
        str | None,
        Field(
            description="Limit search to a specific indexed folder. "
                        "If omitted, searches all indexed folders.",
            default=None,
        ),
    ] = None,
    top_k: Annotated[
        int,
        Field(description="Number of results to return", default=10, ge=1, le=50),
    ] = 10,
    file_type: Annotated[
        str | None,
        Field(
            description="Filter results by file extension, e.g. '.pdf'",
            default=None,
        ),
    ] = None,
    mode: Annotated[
        str,
        Field(
            description="Search mode: 'vector' (default) or 'hybrid' (vector + full-text via RRF)",
            default="vector",
        ),
    ] = "vector",
) -> dict:
    """Search indexed documents using natural language.

    Finds the most relevant document chunks matching the query using
    semantic similarity. Returns ranked results with source file paths
    and relevance scores.

    Use mode='hybrid' to combine vector search with full-text search
    via Reciprocal Rank Fusion for better keyword + semantic matching.

    The folder must be indexed first with index_folder.
    """
    from server.search import semantic_search as _semantic_search

    return _semantic_search(
        query=query,
        folder_path=folder_path,
        top_k=top_k,
        file_type=file_type,
        mode=mode,
    )


@mcp.tool(
    annotations={"readOnlyHint": True}
)
def get_index_status(
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Get status of the search index and any background indexing jobs.

    Returns total chunks and the list of indexed files, plus a `jobs` section
    with active and recently-finished indexing jobs — poll this to follow an
    index_folder run. Job state is in-memory and resets on server restart.
    """
    from server.store import VectorStore
    from server.paths import to_absolute
    from server.jobs import registry

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    stats = {"total_chunks": 0, "total_files": 0, "indexed_files": []}
    try:
        store = VectorStore(db_dir)
        indexed_files = sorted(
            to_absolute(f, db_dir) for f in store.get_all_files()
        )
        stats = {
            "total_chunks": store.count_chunks(),
            "total_files": len(indexed_files),
            "indexed_files": indexed_files,
        }
    except Exception:
        pass  # status must stay readable even if the DB is missing or busy

    return {
        **stats,
        "jobs": {
            "active": [j.to_dict() for j in registry.active()],
            "recent": [j.to_dict() for j in registry.recent()],
        },
    }


@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}
)
def reindex_file(
    file_path: Annotated[str, Field(description="Absolute path to the file to re-index")],
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Force re-index a single file, ignoring the content hash cache.

    Deletes existing chunks for this file, re-parses, re-chunks,
    re-embeds, and stores new chunks. Useful when you know a file
    has changed or when parsing was updated.

    Rejected while a background index_folder job is running, to keep a single
    writer on the index — retry once that job finishes.
    """
    from server.jobs import registry

    if registry.has_running():
        active = registry.active()
        return {
            "status": "rejected",
            "message": "An indexing job is running; retry reindex_file once it finishes.",
            "running_job_id": active[0].job_id if active else None,
        }

    from pathlib import Path
    from server.parsers import extract_text
    from server.chunker import chunk_document
    from server.indexer import embed_chunks, compute_file_hash
    from server.store import VectorStore
    from server.paths import to_relative

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    source_rel = to_relative(str(path), db_dir)

    store = VectorStore(db_dir)
    store.ensure_schema()  # migrate a pre-Tier-2 index in place, if needed
    store.delete_by_file(source_rel)

    parts = extract_text(path)
    chunks = chunk_document(parts, source_rel)
    file_hash = compute_file_hash(path)
    stat = path.stat()

    if chunks:
        chunks = embed_chunks(chunks)
        for c in chunks:
            c["content_hash"] = file_hash
            c["mtime_ns"] = stat.st_mtime_ns
            c["file_size"] = stat.st_size
        store.add_chunks(chunks)

    return {
        "status": "reindexed",
        "file_path": file_path,
        "chunks_created": len(chunks),
    }


def run():
    mcp.run()


if __name__ == "__main__":
    run()
