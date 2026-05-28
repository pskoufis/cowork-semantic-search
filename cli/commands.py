"""Subcommand implementations for ``csemsearch``.

One function per verb; ``cli.main.main()`` dispatches based on the
parsed argparse namespace. Each verb returns an integer process exit
code (``0`` success, ``1`` operational error, ``130`` SIGINT).
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger("cowork_semantic_search.cli")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string (e.g. '4.2 GB').

    Mirrors ``server.main._human_size`` so CLI output is consistent with
    what ``get_index_status`` prints over MCP. Kept local to avoid pulling
    the FastMCP-decorated module just to use one util.
    """
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _format_timestamp(ts: float | None) -> str:
    if ts is None:
        return "(running)"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def status_cmd(db_path: str) -> int:
    """Print index size, counts, and recent job history.

    On an empty / non-existent ``db_path`` (no prior indexing) prints a
    pointer to ``csemsearch index`` and exits 0 — not an error, just a
    cold start.
    """
    print(f"Index: {db_path}")

    if not os.path.isdir(db_path):
        print("  (not yet created — run `csemsearch index <folder>` first)")
        return 0

    from server.store import VectorStore

    store = VectorStore(db_path)
    try:
        store.ensure_schema()
        chunks = store.count_chunks()
        files = len(store.get_all_files())
        size = store.db_size_bytes()
    except Exception as e:
        logger.error("failed to read index at %s: %s", db_path, e)
        return 1

    print(f"  size on disk: {_human_size(size)}")
    print(f"  total chunks: {chunks:,}")
    print(f"  total files:  {files:,}")

    jobs_path = db_path + ".jobs.json"
    if os.path.exists(jobs_path):
        from server.jobs import JobRegistry

        registry = JobRegistry(persist_path=jobs_path)
        active = registry.active()
        recent = registry.recent(limit=10)
        if active or recent:
            print()
            print("Recent jobs (most recent first):")
            print(
                f"  {'job_id':<14} {'state':<12} {'started':<20} "
                f"{'finished':<20} progress"
            )
            for job in active:
                print(
                    f"  {job.job_id[:12]:<14} {job.state:<12} "
                    f"{_format_timestamp(job.started_at):<20} "
                    f"{_format_timestamp(job.finished_at):<20} "
                    f"{job.files_processed}/{job.files_total}"
                )
            for job in recent:
                print(
                    f"  {job.job_id[:12]:<14} {job.state:<12} "
                    f"{_format_timestamp(job.started_at):<20} "
                    f"{_format_timestamp(job.finished_at):<20} "
                    f"{job.files_processed}/{job.files_total}"
                )
    return 0


# ---------------------------------------------------------------------------
# Shared preflight helpers
# ---------------------------------------------------------------------------


def _check_no_running_job(db_path: str) -> str | None:
    """If another indexing job is currently running against this db_path
    (e.g. an MCP background job, or another `csemsearch index` process),
    return a human-readable error message. Otherwise return None.

    Two writers on a LanceDB index would corrupt it; the MCP enforces
    "one job at a time" in its own tool, and the CLI extends the same
    contract to its verbs."""
    jobs_path = db_path + ".jobs.json"
    if not os.path.exists(jobs_path):
        return None
    from server.jobs import JobRegistry

    registry = JobRegistry(persist_path=jobs_path)
    if not registry.has_running():
        return None
    active = registry.active()
    if not active:
        return None
    job = active[0]
    return (
        f"another indexing job is already running on this index\n"
        f"  job_id:     {job.job_id}\n"
        f"  started:    {_format_timestamp(job.started_at)}\n"
        f"  progress:   {job.files_processed}/{job.files_total}\n"
        "wait for it to finish, or kill the other process and retry."
    )


def _install_sigint_handler() -> tuple[threading.Event, Any]:
    """Return (cancel_event, previous_handler).

    The first SIGINT sets cancel_event so the indexer's main loop breaks
    out at the next file boundary and flushes the chunk buffer. A second
    SIGINT raises KeyboardInterrupt so the user can still force-quit a
    hung process."""
    cancel = threading.Event()

    def handler(signum, frame):
        if not cancel.is_set():
            print(
                "\ncancel requested — finishing current file then stopping. "
                "Press Ctrl-C again to force-quit.",
                flush=True,
            )
            cancel.set()
        else:
            raise KeyboardInterrupt

    previous = signal.signal(signal.SIGINT, handler)
    return cancel, previous


def _restore_handler(previous: Any) -> None:
    signal.signal(signal.SIGINT, previous)


# ---------------------------------------------------------------------------
# unpack
# ---------------------------------------------------------------------------


def unpack_cmd(
    folder: str,
    *,
    recursive: bool = True,
    exclude: list[str] | None = None,
    db_path: str | None = None,
) -> int:
    """Run pst → mbox → msg unpack passes on ``folder`` with one progress
    bar per phase. The CLI displays a header before each phase, the bar
    advances per archive, and a summary line prints the per-phase counts
    and elapsed time at the end.

    ``db_path`` is only used to compile exclusion rules (so a
    ``.semanticignore`` that references the index dir applies consistently
    with what the indexer would do)."""
    from cli.progress import ProgressBar
    from server.exclusions import ExclusionRules
    from server.unpacker import run_unpack_passes

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        logger.error("folder not found: %s", folder)
        return 1

    try:
        exclusions = ExclusionRules.load(
            folder_path,
            extra_patterns=exclude or [],
            db_dir=db_path or os.environ.get("LANCEDB_PATH", "./lancedb"),
        )
    except ValueError as e:
        logger.error("invalid exclusion pattern: %s", e)
        return 1

    logger.info("Unpacking archives under %s", folder_path)
    if not recursive:
        logger.info("  (recursive=False — only top-level archives will be unpacked)")

    bars: dict[str, ProgressBar] = {}

    def on_event(phase: str, processed: int, total: int) -> None:
        bar = bars.get(phase)
        if bar is None:
            if total == 0:
                # Zero-archive phases still fire a final (0, 0) event; suppress
                # an empty bar but log a one-line note.
                logger.info("  %s: no archives found", phase)
                return
            bar = ProgressBar(total=total, desc=f"  {phase}", unit="archive")
            bars[phase] = bar
        bar.set_to(processed)
        if processed == total:
            bar.close()

    result = run_unpack_passes(
        folder_path,
        recursive=recursive,
        exclusions=exclusions,
        progress_callback=on_event,
    )

    for bar in bars.values():
        bar.close()

    logger.info("Unpack summary:")
    for phase in ("pst", "mbox", "msg"):
        stats = result[phase]
        logger.info(
            "  %-4s discovered=%d  failed=%d  elapsed=%.2fs",
            phase,
            stats["discovered"],
            stats["failed"],
            stats["elapsed_s"],
        )
    return 0


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def index_cmd(
    folder: str,
    *,
    db_path: str,
    recursive: bool = True,
    exclude: list[str] | None = None,
    unpack_first: bool = True,
    safe_flush: bool = False,
    file_types: list[str] | None = None,
) -> int:
    """Index ``folder`` with rich terminal output and SIGINT cancellation.

    Wraps ``server.indexer.index_folder`` with a tqdm progress bar that
    shows the current file and live chunk/buffer counts. A SIGINT trap
    triggers a graceful cancel: the loop stops at the next file
    boundary, the chunk buffer is flushed, and the process exits 130.
    Re-running picks up where it left off via the hash cache."""
    from cli.progress import ProgressBar
    from server.indexer import index_folder, ProgressEvent
    from server.jobs import JobRegistry

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        logger.error("folder not found: %s", folder)
        return 1

    busy = _check_no_running_job(db_path)
    if busy is not None:
        logger.error("%s", busy)
        return 1

    logger.info("Indexing %s", folder_path)
    logger.info("  db_path:        %s", db_path)
    logger.info("  recursive:      %s", recursive)
    if exclude:
        logger.info("  exclude:        %s", ", ".join(exclude))
    if file_types:
        logger.info("  file_types:     %s", ", ".join(file_types))
    if not unpack_first:
        logger.info("  unpack_first:   False (skipping pst/mbox/msg preprocessing)")
    if safe_flush:
        logger.info("  safe_flush:     True (every file persisted immediately)")

    # Register a job in the on-disk registry so `csemsearch status` (and an
    # MCP `get_index_status` call against the same index) shows CLI runs the
    # same way they show MCP background jobs. The registry is the contract
    # both sides observe to enforce single-writer semantics.
    jobs_path = db_path + ".jobs.json"
    cli_registry = JobRegistry(persist_path=jobs_path)
    job = cli_registry.create(str(folder_path), db_path)
    logger.debug("registered job %s", job.job_id)

    cancel, previous_handler = _install_sigint_handler()
    bar: ProgressBar | None = None
    last_event: ProgressEvent | None = None

    def on_event(event: ProgressEvent) -> None:
        nonlocal bar, last_event
        last_event = event
        if bar is None:
            bar = ProgressBar(total=event.total, desc="  indexing", unit="file")
        bar.set_to(event.processed, postfix=event.current_file or "")
        # Update job progress so `status` reflects live state. Mirrors what
        # the MCP _run_index_job does — JobRegistry throttles writes itself.
        cli_registry.update_progress(job.job_id, event.processed, event.total)

    try:
        try:
            result = index_folder(
                str(folder_path),
                file_types=file_types,
                recursive=recursive,
                db_path=db_path,
                exclude=exclude,
                unpack_first=unpack_first,
                progress_event_callback=on_event,
                cancel_event=cancel,
                flush_threshold=1 if safe_flush else None,
            )
        except Exception as e:
            cli_registry.mark_failed(job.job_id, str(e))
            raise
    finally:
        if bar is not None:
            bar.close()
        _restore_handler(previous_handler)

    if result.get("cancelled"):
        cli_registry.mark_failed(job.job_id, "cancelled by SIGINT")
    else:
        cli_registry.mark_completed(
            job.job_id, result, result.get("finalize_warnings", [])
        )

    cancelled = result.get("cancelled", False)
    total_files = last_event.total if last_event is not None else 0
    if cancelled:
        logger.warning(
            "Cancelled at %d/%d files. Re-run to resume.",
            result["files_indexed"] + result["files_skipped"],
            total_files,
        )
    logger.info("Done in %.2fs", result.get("duration_seconds", 0.0))
    logger.info("  indexed       %s files   %s chunks",
                f"{result['files_indexed']:,}",
                f"{last_event.chunks_written if last_event else 0:,}")
    logger.info("  skipped       %s files", f"{result['files_skipped']:,}")
    if result.get("files_size_skipped", 0):
        logger.info(
            "  oversized     %s files (over MAX_FILE_SIZE_MB cap)",
            f"{result['files_size_skipped']:,}",
        )
    if result.get("files_excluded_pruned", 0):
        logger.info(
            "  pruned        %s files (newly matched exclusion rules)",
            f"{result['files_excluded_pruned']:,}",
        )
    if result.get("files_deleted", 0):
        logger.info(
            "  deleted       %s files (no longer in folder)",
            f"{result['files_deleted']:,}",
        )
    if result.get("files_failed", 0):
        logger.warning("  errors        %s files", f"{result['files_failed']:,}")
    if result.get("descriptions_queued", 0):
        logger.info(
            "  desc. queued  %s spreadsheets (drain via MCP host)",
            f"{result['descriptions_queued']:,}",
        )
    logger.info("  total chunks  %s", f"{result['total_chunks']:,}")
    for warn in result.get("finalize_warnings", []):
        logger.warning("  finalize:     %s", warn)

    return 130 if cancelled else 0


# ---------------------------------------------------------------------------
# run = unpack + index
# ---------------------------------------------------------------------------


def run_cmd(
    folder: str,
    *,
    db_path: str,
    recursive: bool = True,
    exclude: list[str] | None = None,
    safe_flush: bool = False,
    file_types: list[str] | None = None,
) -> int:
    """Sugar for ``unpack`` then ``index``. Mirrors the one-shot behavior
    of the MCP ``index_folder`` tool, which always runs unpack inline."""
    rc = unpack_cmd(folder, recursive=recursive, exclude=exclude, db_path=db_path)
    if rc != 0:
        return rc
    return index_cmd(
        folder,
        db_path=db_path,
        recursive=recursive,
        exclude=exclude,
        unpack_first=False,   # already unpacked above
        safe_flush=safe_flush,
        file_types=file_types,
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search_cmd(
    query: str,
    *,
    db_path: str,
    mode: str = "vector",
    top_k: int = 10,
    folder: str | None = None,
) -> int:
    r"""Run ``semantic_search`` and print a ranked list with file paths,
    scores, and text snippets. Output goes to stdout so callers can pipe
    it (e.g. ``csemsearch search foo | grep '\.pdf'``)."""
    from server.search import semantic_search

    if not os.path.isdir(db_path):
        logger.error("index not found at %s. Run `csemsearch index` first.", db_path)
        return 1
    if mode not in ("vector", "hybrid"):
        logger.error("--mode must be 'vector' or 'hybrid', got %r", mode)
        return 1

    folder_abs = os.path.abspath(folder) if folder else None
    result = semantic_search(
        query=query,
        db_path=db_path,
        folder_path=folder_abs,
        top_k=top_k,
        mode=mode,
    )

    results = result.get("results", [])
    if not results:
        print(f"No results for: {query!r} (mode={mode}, top_k={top_k})")
        return 0

    print(f"Searching ({mode}, n={top_k})…")
    print()
    for i, row in enumerate(results, start=1):
        score = row.get("_distance") or row.get("score") or row.get("_relevance_score") or 0.0
        path = row.get("source_file", "(unknown)")
        snippet = (row.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:197] + "…"
        print(f"{i:>2}  {float(score):.2f}  {path}")
        if snippet:
            print(f"      …{snippet}…")
    return 0


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------


def reindex_cmd(file_path: str, *, db_path: str) -> int:
    """Force re-index one file. Delegates to ``server.indexer.reindex_one_file``
    so behavior matches the MCP ``reindex_file`` tool exactly."""
    from server.indexer import reindex_one_file

    target = Path(file_path).expanduser()
    if not target.exists():
        logger.error("file not found: %s", file_path)
        return 1

    result = reindex_one_file(str(target), db_path)
    status = result.get("status")
    if status == "rejected":
        logger.error("rejected: %s", result.get("message", "rejected"))
        return 1
    if status == "skipped":
        logger.warning("skipped: %s", result.get("reason"))
        return 0
    if status == "failed":
        logger.error("failed: %s", result.get("reason"))
        return 1
    if status == "queued":
        logger.info(
            "queued spreadsheet description (needs=%s) for %s",
            result.get("needs"),
            file_path,
        )
        return 0
    logger.info(
        "reindexed %s — %d chunks written",
        file_path,
        result.get("chunks_created", 0),
    )
    return 0
