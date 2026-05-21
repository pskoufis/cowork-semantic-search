"""Tests for the in-memory background-indexing job registry."""

import pytest

from server.jobs import IndexJob, JobRegistry


@pytest.fixture
def registry():
    return JobRegistry()


def test_create_returns_running_job(registry):
    job = registry.create("/some/folder", "/some/db")
    assert isinstance(job, IndexJob)
    assert job.state == "running"
    assert job.folder_path == "/some/folder"
    assert job.db_path == "/some/db"
    assert job.job_id


def test_create_job_ids_are_unique(registry):
    a = registry.create("/f", "/db")
    b = registry.create("/f", "/db")
    assert a.job_id != b.job_id


def test_has_running_false_when_empty(registry):
    assert registry.has_running() is False


def test_has_running_true_while_job_running(registry):
    registry.create("/f", "/db")
    assert registry.has_running() is True


def test_has_running_false_after_completion(registry):
    job = registry.create("/f", "/db")
    registry.mark_completed(job.job_id, {"files_indexed": 3}, [])
    assert registry.has_running() is False


def test_has_running_false_after_failure(registry):
    job = registry.create("/f", "/db")
    registry.mark_failed(job.job_id, "boom")
    assert registry.has_running() is False


def test_get_returns_job_by_id(registry):
    job = registry.create("/f", "/db")
    assert registry.get(job.job_id) is job


def test_get_returns_none_for_unknown_id(registry):
    assert registry.get("nonexistent") is None


def test_update_progress(registry):
    job = registry.create("/f", "/db")
    registry.update_progress(job.job_id, processed=2, total=10)
    assert job.files_processed == 2
    assert job.files_total == 10


def test_mark_completed_records_result_and_warnings(registry):
    job = registry.create("/f", "/db")
    registry.mark_completed(job.job_id, {"files_indexed": 5}, ["index build skipped"])
    assert job.state == "completed"
    assert job.result == {"files_indexed": 5}
    assert job.finalize_warnings == ["index build skipped"]
    assert job.finished_at is not None


def test_mark_failed_records_error(registry):
    job = registry.create("/f", "/db")
    registry.mark_failed(job.job_id, "boom")
    assert job.state == "failed"
    assert job.error == "boom"
    assert job.finished_at is not None


def test_active_lists_only_running_jobs(registry):
    running = registry.create("/f1", "/db")
    done = registry.create("/f2", "/db")
    registry.mark_completed(done.job_id, {}, [])
    active = registry.active()
    assert [j.job_id for j in active] == [running.job_id]


def test_recent_lists_finished_jobs_newest_first(registry):
    first = registry.create("/f1", "/db")
    second = registry.create("/f2", "/db")
    registry.mark_completed(first.job_id, {}, [])
    registry.mark_failed(second.job_id, "err")
    recent = registry.recent()
    assert [j.job_id for j in recent] == [second.job_id, first.job_id]


def test_recent_excludes_running_jobs(registry):
    registry.create("/running", "/db")
    done = registry.create("/done", "/db")
    registry.mark_completed(done.job_id, {}, [])
    assert [j.job_id for j in registry.recent()] == [done.job_id]


def test_recent_respects_limit(registry):
    for i in range(5):
        job = registry.create(f"/f{i}", "/db")
        registry.mark_completed(job.job_id, {}, [])
    assert len(registry.recent(limit=3)) == 3


def test_to_dict_omits_task_handle(registry):
    job = registry.create("/f", "/db")
    job.task = object()  # stand-in for an asyncio.Task — not JSON-serializable
    d = job.to_dict()
    assert "task" not in d
    assert d["job_id"] == job.job_id
    assert d["state"] == "running"
