"""In-memory registry of background indexing jobs.

Job state lives in process memory only and does not survive a server restart.
That is acceptable for a stdio MCP server: re-running ``index_folder`` is cheap
because unchanged files are skipped by content hash.

The registry is accessed from two threads — the asyncio event loop (reads, for
``get_index_status``) and the worker thread running the indexing job (writes,
for progress updates) — so every access is guarded by a lock.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Job states.
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"


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


class JobRegistry:
    """Thread-safe collection of indexing jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, IndexJob] = {}
        self._order: list[str] = []  # insertion order, oldest first

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

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.state = FAILED
                job.error = error
                job.finished_at = time.time()

    def active(self) -> list[IndexJob]:
        """Running jobs, oldest first."""
        with self._lock:
            return [
                self._jobs[i] for i in self._order if self._jobs[i].state == RUNNING
            ]

    def recent(self, limit: int = 10) -> list[IndexJob]:
        """Finished (completed or failed) jobs, newest first, capped at ``limit``."""
        with self._lock:
            finished = [
                self._jobs[i] for i in self._order if self._jobs[i].state != RUNNING
            ]
            return finished[::-1][:limit]


# Process-wide singleton used by the MCP tools.
registry = JobRegistry()
