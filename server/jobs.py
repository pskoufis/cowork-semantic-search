"""Registry of background indexing jobs, persisted to disk.

The registry is mirrored to a small JSON file next to the LanceDB directory so a
job's record survives the server process restarting. A job still marked
``running`` when the registry is loaded belonged to a process that is now gone,
so it is rewritten to ``interrupted`` — the user gets a clear signal instead of
a record that silently vanished.

This is *not* a resume mechanism: re-running ``index_folder`` after an
interruption is already cheap because Tier 2's mtime+size fast path skips every
unchanged file with a single stat() call.

The registry is accessed from two threads — the asyncio event loop (reads, for
``get_index_status``) and the worker thread running the indexing job (writes,
for progress updates) — so every access is guarded by a lock. Persistence is
best-effort: a failed read or write never breaks indexing.
"""

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from server.paths import default_db_dir

# Job states.
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
# A job found still 'running' when the registry is loaded from disk — the
# process that owned it stopped before it finished.
INTERRUPTED = "interrupted"

# Persist the registry no more than once per this many seconds while progress
# updates stream in (one update fires per file processed).
_SAVE_THROTTLE_SECONDS = 2.0


@dataclass
class IndexJob:
    """A single background indexing run and its progress."""

    job_id: str
    folder_path: str
    db_path: str
    state: str = RUNNING
    files_total: int = 0
    files_processed: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    result: dict | None = None
    finalize_warnings: list[str] = field(default_factory=list)
    # The asyncio.Task running this job. Held so the task is not garbage
    # collected mid-run; never serialized (see to_dict).
    task: Any = None

    def to_dict(self) -> dict:
        """JSON-safe view of the job. Omits ``task`` — an asyncio.Task is not
        JSON-serializable."""
        return {
            "job_id": self.job_id,
            "folder_path": self.folder_path,
            "db_path": self.db_path,
            "state": self.state,
            "files_total": self.files_total,
            "files_processed": self.files_processed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
            "finalize_warnings": self.finalize_warnings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IndexJob":
        """Rebuild a job from its to_dict() form. The asyncio Task is not
        restored — the process that owned it is gone."""
        return cls(
            job_id=d["job_id"],
            folder_path=d.get("folder_path", ""),
            db_path=d.get("db_path", ""),
            state=d.get("state", RUNNING),
            files_total=d.get("files_total", 0),
            files_processed=d.get("files_processed", 0),
            started_at=d.get("started_at", time.time()),
            finished_at=d.get("finished_at"),
            error=d.get("error"),
            result=d.get("result"),
            finalize_warnings=d.get("finalize_warnings") or [],
        )


class JobRegistry:
    """Thread-safe collection of indexing jobs, optionally mirrored to disk.

    With ``persist_path`` set, every mutation is written to that JSON file and
    the file is read back at construction. With ``persist_path`` None the
    registry is purely in-memory (used by unit tests).
    """

    def __init__(self, persist_path: str | None = None):
        self._lock = threading.Lock()
        self._jobs: dict[str, IndexJob] = {}
        self._order: list[str] = []  # insertion order, oldest first
        self._persist_path = persist_path
        self._last_save_at = 0.0
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        """Load persisted jobs at startup. Any job still 'running' is rewritten
        to 'interrupted'. Best-effort: a missing, partial or corrupt file
        yields an empty registry rather than crashing module import."""
        if self._persist_path is None or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, encoding="utf-8") as f:
                payload = json.load(f)
            for d in payload:
                job = IndexJob.from_dict(d)
                if job.state == RUNNING:
                    job.state = INTERRUPTED
                    job.error = (
                        "interrupted: the server stopped before this job finished"
                    )
                    if job.finished_at is None:
                        job.finished_at = time.time()
                self._jobs[job.job_id] = job
                self._order.append(job.job_id)
        except Exception:
            # Corrupt / partial / unknown-format file: start clean.
            self._jobs.clear()
            self._order.clear()

    def _save(self) -> None:
        """Write all jobs to disk via an atomic temp-file replace. Best-effort:
        a failure here must never fail an indexing job. Caller holds the lock."""
        if self._persist_path is None:
            return
        try:
            payload = [self._jobs[i].to_dict() for i in self._order]
            parent = os.path.dirname(self._persist_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = f"{self._persist_path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, self._persist_path)
            self._last_save_at = time.time()
        except Exception:
            pass  # persistence is a nicety; never break a job over it

    # -- job lifecycle -------------------------------------------------------

    def has_running(self) -> bool:
        """True if any job is still running."""
        with self._lock:
            return any(j.state == RUNNING for j in self._jobs.values())

    def create(self, folder_path: str, db_path: str) -> IndexJob:
        """Register a new running job and return it."""
        with self._lock:
            job = IndexJob(
                job_id=uuid.uuid4().hex,
                folder_path=folder_path,
                db_path=db_path,
            )
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._save()
            return job

    def get(self, job_id: str) -> IndexJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, processed: int, total: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.files_processed = processed
                job.files_total = total
                # Throttled: update_progress fires once per file, so do not
                # write the JSON on every call.
                if time.time() - self._last_save_at >= _SAVE_THROTTLE_SECONDS:
                    self._save()

    def mark_completed(
        self, job_id: str, result: dict, warnings: list[str] | None = None
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.state = COMPLETED
                job.result = result
                job.finalize_warnings = warnings or []
                job.finished_at = time.time()
                self._save()

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.state = FAILED
                job.error = error
                job.finished_at = time.time()
                self._save()

    def active(self) -> list[IndexJob]:
        """Running jobs, oldest first."""
        with self._lock:
            return [
                self._jobs[i] for i in self._order if self._jobs[i].state == RUNNING
            ]

    def recent(self, limit: int = 10) -> list[IndexJob]:
        """Finished jobs (completed, failed or interrupted), newest first,
        capped at ``limit``."""
        with self._lock:
            finished = [
                self._jobs[i] for i in self._order if self._jobs[i].state != RUNNING
            ]
            return finished[::-1][:limit]


def _default_jobs_path() -> str:
    """JSON file mirroring the registry — a sibling of the LanceDB directory."""
    return default_db_dir() + ".jobs.json"


# Process-wide singleton used by the MCP tools, persisted next to the index.
registry = JobRegistry(persist_path=_default_jobs_path())
