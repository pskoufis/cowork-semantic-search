"""FastMCP server entry point with tool definitions."""

import asyncio
import os
from typing import Annotated

from fastmcp import FastMCP, Context
from pydantic import Field

mcp = FastMCP("Semantic Search")

# How long to wait between a failed ctx.sample call and its one retry.
# Module-level so tests can monkeypatch it to 0 and not spend seconds in
# backoff. Not env-configurable on purpose — the spec calls for a hard-coded
# value.
_SAMPLE_RETRY_BACKOFF_SECONDS = 2.0


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string, e.g. '4.2 GB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


async def _run_index_job(job, file_types, recursive, exclude, ctx=None) -> None:
    """Run the synchronous index_folder in a worker thread, recording progress
    and the final outcome on the job record.

    The CPU-bound embedding work runs off the event loop via asyncio.to_thread.
    Any exception is captured onto the job so it never silently hangs in the
    'running' state.

    After indexing, if the client supports MCP sampling, auto-drains any
    spreadsheet description requests via ctx.sample(). Failures leave entries
    in the queue for the host LLM to drain manually via submit_description.
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
        if ctx is not None and result.get("descriptions_queued", 0) > 0:
            sampled = await _auto_drain_descriptions(job.db_path, ctx)
            result["descriptions_sampled"] = sampled
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
    ctx: Context = None,
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
        _run_index_job(job, file_types, recursive, exclude, ctx=ctx)
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
    store.ensure_schema()  # migrate an older index in place, if needed
    store.delete_by_file(source_rel)

    # Spreadsheet path: extract a preview, enqueue, and return. Description
    # chunks land when the host LLM submits via submit_description (or when
    # ctx.sample drains the queue — but reindex_file has no ctx).
    from server.spreadsheets import (
        SPREADSHEET_EXTENSIONS, UnreadableSpreadsheetError,
        extract_preview, needs_for_preview,
    )
    import json as _json
    if path.suffix.lower() in SPREADSHEET_EXTENSIONS:
        try:
            preview = extract_preview(path)
        except UnreadableSpreadsheetError as exc:
            return {
                "status": "failed",
                "file_path": file_path,
                "reason": f"unreadable spreadsheet: {exc}",
            }
        file_hash = compute_file_hash(path)
        needs = needs_for_preview(preview)
        store.enqueue_pending(
            file_path=source_rel,
            needs=needs,
            preview_json=_json.dumps(preview),
            content_hash=file_hash,
        )
        return {
            "status": "queued",
            "file_path": file_path,
            "needs": needs,
        }

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


@mcp.tool(annotations={"readOnlyHint": True})
def list_pending_descriptions(
    folder_path: Annotated[
        str | None,
        Field(
            description="Filter to entries whose file_path begins with this "
                        "folder (matched as a path-prefix). If omitted, "
                        "returns entries across all folders.",
            default=None,
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Max entries to return per call", default=20, ge=1, le=100),
    ] = 20,
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """List spreadsheets awaiting LLM-generated descriptions.

    Each entry returns the preview (sheet names, headers, dtypes, sample
    rows) the host LLM needs to write a description, plus the list of items
    still needed: per-sheet descriptions like "sheet:Sales" and/or "file"
    for the file-level rollup.

    Submit a description back via `submit_description`. To skip a file
    permanently (until its content changes), call
    `dismiss_pending_description`.

    Files that have been deleted from disk since they were enqueued are
    auto-evicted from the queue and not surfaced.
    """
    import json as _json
    from pathlib import Path as _Path
    from server.store import VectorStore
    from server.paths import to_absolute

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    store = VectorStore(db_dir)
    # Pull extra so deleted-file evictions don't leave us short of `limit`.
    raw = store.list_pending(folder_path=folder_path, limit=limit * 2)

    pending: list[dict] = []
    for entry in raw:
        abs_path = _Path(to_absolute(entry["file_path"], db_dir))
        if not abs_path.exists():
            store.remove_pending(entry["file_path"])
            continue
        pending.append({
            "file_path": entry["file_path"],
            "needs": entry["needs"],
            "preview": _json.loads(entry["preview_json"]),
        })
        if len(pending) >= limit:
            break

    return {
        "pending": pending,
        "total_remaining": store.pending_count(),
    }


def _submit_one_description(
    *,
    db_dir: str,
    file_path: str,
    sheet_name: str | None,
    description: str,
) -> dict:
    """Embed and store one description chunk for a queued spreadsheet.

    Shared between the submit_description MCP tool and the auto-drainer.
    Returns {status, file_complete, reason?}. Does not raise on expected
    rejection paths so tools can surface the reason verbatim.
    """
    import json as _json
    from pathlib import Path
    from server.store import VectorStore
    from server.paths import to_relative
    from server.indexer import embed_chunks, compute_file_hash
    from server.chunker import _short_hash
    from server.spreadsheets import MAX_DESCRIPTION_BYTES, needs_for_preview

    path = Path(file_path)
    if not path.exists():
        return {"status": "rejected", "reason": "file_not_found"}

    if not description or not description.strip():
        return {"status": "rejected", "reason": "empty_description"}
    if len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
        return {"status": "rejected", "reason": "description_too_large"}

    source_rel = to_relative(str(path), db_dir)
    store = VectorStore(db_dir)
    store.ensure_schema()

    entry = store.get_pending_entry(source_rel)
    if entry is None:
        return {"status": "rejected", "reason": "not_pending"}

    needed_item = "file" if sheet_name is None else f"sheet:{sheet_name}"
    if needed_item not in entry["needs"]:
        return {"status": "rejected", "reason": "item_not_needed"}

    # chunk_index = position in the canonical needs order from the preview,
    # so re-submissions don't collide and id = {file_hash}_{slot} stays stable.
    preview = _json.loads(entry["preview_json"])
    full_needs = needs_for_preview(preview)
    slot = full_needs.index(needed_item)

    stat = path.stat()
    file_hash = compute_file_hash(path)
    chunk_kind = "file_description" if sheet_name is None else "sheet_description"

    chunk = {
        "id": f"{_short_hash(source_rel)}_{slot}",
        "text": description.strip(),
        "source_file": source_rel,
        "file_name": path.name,
        "file_type": path.suffix.lower(),
        "folder_path": os.path.dirname(source_rel) or ".",
        "chunk_index": slot,
        "content_hash": file_hash,
        "mtime_ns": stat.st_mtime_ns,
        "file_size": stat.st_size,
        "chunk_kind": chunk_kind,
        "sheet_name": sheet_name,
    }
    [chunk] = embed_chunks([chunk])
    store.add_chunks([chunk])

    remaining = [n for n in entry["needs"] if n != needed_item]
    if not remaining:
        store.remove_pending(source_rel)
        return {"status": "stored", "file_complete": True}
    store.update_pending_needs(source_rel, remaining)
    return {"status": "stored", "file_complete": False}


async def _sample_with_retry(ctx, prompt: str) -> str | None:
    """Call ctx.sample with one retry. Returns the description text on
    success, None on permanent failure (caller treats None as 'abort this
    file's commit and leave its queue entry alone').

    Rejects empty / too-large responses up front so a sloppy LLM reply
    doesn't pollute the index.
    """
    from server.spreadsheets import MAX_DESCRIPTION_BYTES

    for attempt in range(2):
        try:
            result = await ctx.sample(prompt)
            text = getattr(result, "text", None)
            if text is None:
                text = str(result)
            text = text.strip()
            if not text:
                raise ValueError("empty sampling response")
            if len(text.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
                raise ValueError("sampling response over size cap")
            return text
        except Exception:
            if attempt == 0:
                await asyncio.sleep(_SAMPLE_RETRY_BACKOFF_SECONDS)
                continue
            return None
    return None


def _build_description_chunk(
    *,
    db_dir: str,
    abs_path,
    source_rel: str,
    content_hash: str,
    slot: int,
    description: str,
    chunk_kind: str,
    sheet_name: str | None,
) -> dict:
    """Build a description chunk dict ready for embed_chunks + add_chunks."""
    from server.chunker import _short_hash

    stat = abs_path.stat()
    return {
        "id": f"{_short_hash(source_rel)}_{slot}",
        "text": description,
        "source_file": source_rel,
        "file_name": abs_path.name,
        "file_type": abs_path.suffix.lower(),
        "folder_path": os.path.dirname(source_rel) or ".",
        "chunk_index": slot,
        "content_hash": content_hash,
        "mtime_ns": stat.st_mtime_ns,
        "file_size": stat.st_size,
        "chunk_kind": chunk_kind,
        "sheet_name": sheet_name,
    }


async def _auto_drain_descriptions(db_dir: str, ctx) -> int:
    """Drain the pending_descriptions queue via ctx.sample().

    Per-file atomic commit: all of a file's descriptions are buffered and
    written together only if every sample succeeds. On any failure inside a
    file, the buffer is discarded and the queue entry stays for manual
    drain later — a workbook is never half-sampled half-queued.

    Returns the number of description chunks actually written.
    """
    import json as _json
    from pathlib import Path
    from server.store import VectorStore
    from server.paths import to_absolute
    from server.indexer import embed_chunks
    from server.spreadsheets import (
        build_sheet_prompt, build_file_prompt, needs_for_preview,
    )

    store = VectorStore(db_dir)
    written = 0
    failed_paths: set[str] = set()

    while True:
        batch = [
            e for e in store.list_pending(limit=20)
            if e["file_path"] not in failed_paths
        ]
        if not batch:
            break
        for entry in batch:
            preview = _json.loads(entry["preview_json"])
            abs_path = Path(to_absolute(entry["file_path"], db_dir))
            if not abs_path.exists():
                store.remove_pending(entry["file_path"])
                continue

            full_needs = needs_for_preview(preview)
            sheet_descriptions: dict[str, str] = {}
            chunks: list[dict] = []
            aborted = False

            # Per-sheet first
            for slot, item in enumerate(full_needs):
                if item == "file":
                    continue
                sheet_name = item.removeprefix("sheet:")
                desc = await _sample_with_retry(
                    ctx, build_sheet_prompt(preview, sheet_name)
                )
                if desc is None:
                    aborted = True
                    break
                sheet_descriptions[sheet_name] = desc
                chunks.append(_build_description_chunk(
                    db_dir=db_dir,
                    abs_path=abs_path,
                    source_rel=entry["file_path"],
                    content_hash=entry["content_hash"],
                    slot=slot,
                    description=desc,
                    chunk_kind="sheet_description",
                    sheet_name=sheet_name,
                ))

            if aborted:
                failed_paths.add(entry["file_path"])
                continue

            # File-level rollup
            file_slot = full_needs.index("file")
            file_desc = await _sample_with_retry(
                ctx, build_file_prompt(preview, sheet_descriptions)
            )
            if file_desc is None:
                failed_paths.add(entry["file_path"])
                continue
            chunks.append(_build_description_chunk(
                db_dir=db_dir,
                abs_path=abs_path,
                source_rel=entry["file_path"],
                content_hash=entry["content_hash"],
                slot=file_slot,
                description=file_desc,
                chunk_kind="file_description",
                sheet_name=None,
            ))

            # Atomic commit
            chunks = embed_chunks(chunks)
            store.add_chunks(chunks)
            store.remove_pending(entry["file_path"])
            written += len(chunks)

    return written


@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False}
)
def submit_description(
    file_path: Annotated[str, Field(description="Absolute path to the spreadsheet")],
    sheet_name: Annotated[
        str | None,
        Field(
            description="Name of the sheet being described. Pass null for "
                        "the file-level rollup. For CSV, always pass null.",
            default=None,
        ),
    ] = None,
    description: Annotated[
        str,
        Field(description="The LLM-generated description text"),
    ] = "",
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Store a description for a queued spreadsheet sheet or file-level rollup.

    Embeds the description text and writes a chunk with chunk_kind set to
    `sheet_description` (with sheet_name) or `file_description`. Updates the
    matching pending-queue entry: removes the satisfied item, and when the
    last item lands, removes the entry entirely (file_complete=true).

    Returns one of:
    - {"status": "stored", "file_complete": bool}
    - {"status": "rejected", "reason": "file_not_found" | "empty_description"
      | "description_too_large" | "not_pending" | "item_not_needed"}
    """
    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)
    return _submit_one_description(
        db_dir=db_dir,
        file_path=file_path,
        sheet_name=sheet_name,
        description=description,
    )


@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}
)
def dismiss_pending_description(
    file_path: Annotated[str, Field(description="Absolute path to the spreadsheet to dismiss")],
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Decline to describe a queued spreadsheet.

    Removes the entry from the pending queue and records a dismissal keyed
    by the file's current content hash. On future index_folder runs the
    file is skipped silently as long as the hash matches; if the file's
    content changes, the dismissal becomes stale and the file is
    re-enqueued automatically.

    Returns:
    - {"status": "dismissed", "file_path": str}
    - {"status": "rejected", "reason": "file_not_found" | "not_pending"}
    """
    from pathlib import Path
    from server.store import VectorStore
    from server.paths import to_relative
    from server.indexer import compute_file_hash

    path = Path(file_path)
    if not path.exists():
        return {"status": "rejected", "reason": "file_not_found"}

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)
    source_rel = to_relative(str(path), db_dir)

    store = VectorStore(db_dir)
    entry = store.get_pending_entry(source_rel)
    if entry is None:
        return {"status": "rejected", "reason": "not_pending"}

    file_hash = compute_file_hash(path)
    store.remove_pending(source_rel)
    store.dismiss(source_rel, file_hash)
    return {"status": "dismissed", "file_path": file_path}


def run():
    mcp.run()


if __name__ == "__main__":
    run()
