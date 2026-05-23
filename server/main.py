"""FastMCP server entry point with tool definitions."""

import asyncio
import os
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("Semantic Search")


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string, e.g. '4.2 GB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


async def _run_index_job(job, file_types, recursive, exclude) -> None:
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
            exclude,
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
                        "Defaults to all supported types: .txt, .md, .pdf, "
                        ".docx, .pptx, .csv, .pst",
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
    exclude: Annotated[
        list[str] | None,
        Field(
            description="Optional list of gitignore-style patterns to exclude. "
                        "Patterns are matched against paths relative to "
                        "folder_path. Combined with any patterns in a "
                        ".semanticignore file at folder_path's root. Examples: "
                        "['node_modules/**', '*.log', '!keep.log'].",
            default=None,
        ),
    ] = None,
) -> dict:
    """Start a background job to index or re-index all documents in a folder.

    Scans the folder for supported document types (.txt, .md, .pdf, .docx,
    .pptx, .csv, .pst), extracts text, splits into chunks, computes embeddings,
    stores them in a local vector database, and builds an ANN index for fast
    search. Only files that have changed since the last run are re-processed.

    Indexing runs in the background: this call returns immediately with a
    job_id. Poll get_index_status to follow progress and read the final result.
    Only one indexing job runs at a time — a call made while another job is
    running is rejected. Job records are persisted next to the index, so a run
    interrupted by a server restart shows up afterwards as 'interrupted';
    simply re-run to recover (unchanged files are skipped, so it is cheap).

    Files larger than the MAX_FILE_SIZE_MB cap (default 100 MB) are skipped and
    reported in the result's oversized_files list rather than indexed; .pst
    archives stream from disk and are exempt from the cap.

    Exclusion rules come from two sources, combined: a `.semanticignore` file
    at folder_path's root (gitignore syntax), and the `exclude` parameter. On
    a re-run, files that newly match a rule are pruned from the index and
    counted in the result's `files_excluded_pruned`. A syntactically invalid
    pattern is rejected up front with `reason: 'invalid_exclusion_pattern'`
    — no job is started.
    """
    from server.jobs import registry
    from server.exclusions import ExclusionRules
    from pathlib import Path as _Path

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    # Compile exclusions up front so a bad pattern is reported synchronously
    # rather than as a delayed job failure. The same compile runs inside
    # index_folder, but doing it twice is cheap and gives the user a clean
    # rejection path.
    try:
        ExclusionRules.load(_Path(folder_path), extra_patterns=exclude, db_dir=db_dir)
    except ValueError as e:
        return {
            "status": "rejected",
            "reason": "invalid_exclusion_pattern",
            "message": str(e),
        }

    if registry.has_running():
        active = registry.active()
        return {
            "status": "rejected",
            "message": "An indexing job is already running. "
                       "Poll get_index_status for progress.",
            "running_job_id": active[0].job_id if active else None,
        }

    job = registry.create(folder_path, db_dir)
    job.task = asyncio.create_task(
        _run_index_job(job, file_types, recursive, exclude)
    )
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
        Field(description="Number of results to return (1–100)", default=25, ge=1, le=100),
    ] = 25,
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
    folder_path: Annotated[
        str | None,
        Field(
            description="Optional folder to report exclusion rules for. When "
                        "supplied, the response includes an `exclusions` field "
                        "describing the .semanticignore at that folder's root "
                        "(or null when none exists). Omitted otherwise.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Get status of the search index and any background indexing jobs.

    Returns total chunks, the list of indexed files, and the index size on disk
    (`db_size_bytes` / `db_size`), plus a `jobs` section with active and
    recently-finished indexing jobs — poll this to follow an index_folder run.
    Job records are persisted next to the index; a run cut short by a server
    restart appears here as 'interrupted'.

    When `folder_path` is supplied, the response also includes an `exclusions`
    object describing the active .semanticignore at that folder (or `null` if
    no file is present). The indexed-root is not persisted, so there's no
    honest way to list all folders' exclusions in one call — pass the folder
    you care about.
    """
    from server.store import VectorStore
    from server.paths import to_absolute
    from server.jobs import registry
    from server.exclusions import ExclusionRules, SEMANTICIGNORE_FILENAME
    from pathlib import Path as _Path

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    stats = {
        "total_chunks": 0,
        "total_files": 0,
        "indexed_files": [],
        "db_size_bytes": 0,
        "db_size": "0 B",
    }
    try:
        store = VectorStore(db_dir)
        indexed_files = sorted(
            to_absolute(f, db_dir) for f in store.get_all_files()
        )
        size_bytes = store.db_size_bytes()
        stats = {
            "total_chunks": store.count_chunks(),
            "total_files": len(indexed_files),
            "indexed_files": indexed_files,
            "db_size_bytes": size_bytes,
            "db_size": _human_size(size_bytes),
        }
    except Exception:
        pass  # status must stay readable even if the DB is missing or busy

    out: dict = {
        **stats,
        "jobs": {
            "active": [j.to_dict() for j in registry.active()],
            "recent": [j.to_dict() for j in registry.recent()],
        },
    }

    if folder_path is not None:
        folder = _Path(folder_path)
        ignore_file = folder / SEMANTICIGNORE_FILENAME
        if ignore_file.is_file():
            try:
                rules = ExclusionRules.load(
                    folder, extra_patterns=None, db_dir=db_dir
                )
                out["exclusions"] = {
                    "folder_path": str(folder),
                    "semanticignore_path": str(ignore_file),
                    "patterns": rules.patterns,
                }
            except ValueError as e:
                # Surface the parse failure as the exclusions payload so the
                # caller can see *why* their .semanticignore isn't working.
                out["exclusions"] = {
                    "folder_path": str(folder),
                    "semanticignore_path": str(ignore_file),
                    "error": str(e),
                }
        else:
            out["exclusions"] = None

    return out


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
    writer on the index — retry once that job finishes. A file over the
    MAX_FILE_SIZE_MB cap (default 100 MB) is returned with status 'skipped' and
    the index is left untouched; .pst archives are exempt from the cap.

    Exclusion rules (.semanticignore / index_folder's `exclude` param) are
    bypassed: reindex_file is an explicit per-file act and force-indexes the
    requested file. The one hard rule that is NOT bypassed is the LanceDB
    self-protection — a file inside the active index directory is rejected
    with `reason: 'inside_index_dir'`.
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
    from server.indexer import (
        embed_chunks, compute_file_hash, MAX_FILE_SIZE_BYTES, exceeds_size_cap,
    )
    from server.store import VectorStore
    from server.paths import to_relative

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    # Hard rule: a path inside the active LanceDB directory is never indexed —
    # otherwise a stray call could try to parse the index's own files.
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    db_resolved = Path(db_dir).resolve() if Path(db_dir).exists() else Path(db_dir)
    if resolved.is_relative_to(db_resolved):
        return {
            "status": "rejected",
            "reason": "inside_index_dir",
            "file_path": file_path,
            "message": (
                "file_path is inside the active LanceDB directory; "
                "reindex_file refuses to parse the index's own files."
            ),
        }

    stat = path.stat()
    if exceeds_size_cap(path, stat.st_size):
        # Too large to parse without risking an OOM; leave the index untouched.
        # Streaming formats (.pst) are exempt from the cap.
        return {
            "status": "skipped",
            "file_path": file_path,
            "reason": (
                f"file is {stat.st_size / 1024 / 1024:.1f} MB, over the "
                f"{MAX_FILE_SIZE_BYTES // 1024 // 1024} MB indexing cap "
                f"(set via the MAX_FILE_SIZE_MB env var)"
            ),
            "chunks_created": 0,
        }

    source_rel = to_relative(str(path), db_dir)

    store = VectorStore(db_dir)
    store.ensure_schema()  # migrate a pre-Tier-2 index in place, if needed
    store.delete_by_file(source_rel)

    parts = extract_text(path)
    chunks = chunk_document(parts, source_rel)
    file_hash = compute_file_hash(path)

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
