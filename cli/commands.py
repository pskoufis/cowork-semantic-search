"""Subcommand implementations for ``csemsearch``.

One function per verb; ``cli.main.main()`` dispatches based on the
parsed argparse namespace. Each verb returns an integer process exit
code (``0`` success, ``1`` operational error, ``130`` SIGINT).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone


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
