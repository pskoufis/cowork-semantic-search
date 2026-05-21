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


# -- Tier 3: persistence across server restarts (item 4.9) -------------------

def test_no_persistence_without_a_path():
    """With persist_path=None the registry is purely in-memory."""
    reg = JobRegistry()
    assert reg._persist_path is None
    job = reg.create("/corpus", "/db")  # must not raise
    assert reg.get(job.job_id) is job


def test_persisted_jobs_reload_in_a_fresh_registry(tmp_path):
    """A completed job written by one registry is read back by the next."""
    path = str(tmp_path / "jobs.json")
    reg = JobRegistry(persist_path=path)
    job = reg.create("/corpus", "/db")
    reg.mark_completed(job.job_id, {"files_indexed": 7}, ["a warning"])

    reloaded = JobRegistry(persist_path=path)
    got = reloaded.get(job.job_id)
    assert got is not None
    assert got.state == "completed"
    assert got.result == {"files_indexed": 7}
    assert got.finalize_warnings == ["a warning"]


def test_running_job_loads_as_interrupted(tmp_path):
    """A job still 'running' on disk belonged to a process that is now gone —
    it loads as 'interrupted'."""
    path = str(tmp_path / "jobs.json")
    reg = JobRegistry(persist_path=path)
    job = reg.create("/corpus", "/db")  # left running, as if the process died

    reloaded = JobRegistry(persist_path=path)
    got = reloaded.get(job.job_id)
    assert got.state == "interrupted"
    assert got.finished_at is not None
    assert got.error and "interrupted" in got.error
    # An interrupted job is finished: not running, shows up in recent().
    assert reloaded.has_running() is False
    assert [j.job_id for j in reloaded.recent()] == [job.job_id]
    assert reloaded.active() == []


def test_corrupt_jobs_file_yields_empty_registry(tmp_path):
    """A corrupt persistence file does not crash construction."""
    path = tmp_path / "jobs.json"
    path.write_text("{ this is not valid json")
    reg = JobRegistry(persist_path=str(path))
    assert reg.recent() == []
    assert reg.active() == []


def test_save_replaces_atomically_leaving_no_temp_file(tmp_path):
    """After a save the persistence file exists and no temp file lingers."""
    path = tmp_path / "jobs.json"
    reg = JobRegistry(persist_path=str(path))
    reg.create("/corpus", "/db")

    assert path.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "jobs.json"]
    assert leftovers == []
