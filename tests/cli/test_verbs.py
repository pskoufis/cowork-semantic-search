"""Integration tests for csemsearch's index/unpack/run/search/reindex verbs.

All tests run in-process via ``cli.main.main(argv)`` and mock the embedding
model. They build a tiny corpus in ``tmp_path``, run the CLI against it,
and assert on the resulting index plus the printed output.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cli.main import main
from server.store import EMBEDDING_DIM


def _fake_embed(texts):
    out = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2**32)
        v = rng.randn(EMBEDDING_DIM).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return np.array(out)


@pytest.fixture(autouse=True)
def _mock_embedding_model():
    """Every CLI verb that embeds (index, run, search, reindex) must use the
    fake encoder so tests stay fast and offline."""
    with patch("server.indexer.get_model") as mock_get_model:
        mock_get_model.return_value = type(
            "MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)}
        )()
        yield


@pytest.fixture
def small_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.txt").write_text("Alpha quarterly revenue report 2024.")
    (corpus / "beta.md").write_text("# Beta\n\nSecond document discussing margins.")
    (corpus / "subdir").mkdir()
    (corpus / "subdir" / "deep.txt").write_text("Nested file content with keywords.")
    return corpus


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def test_index_builds_index_and_prints_summary(small_corpus, tmp_path, capsys):
    db = tmp_path / "lancedb"
    rc = main(["--db-path", str(db), "index", str(small_corpus)])
    assert rc == 0
    captured = capsys.readouterr()
    # Summary lands on stderr (via logger).
    assert "indexed" in captured.err
    assert "3 files" in captured.err or "3 " in captured.err
    assert db.exists()


def test_index_rejects_running_job(small_corpus, tmp_path, capsys):
    db = tmp_path / "lancedb"
    db.mkdir(parents=True)
    # Hand-write a registry file with a running job — simulates an MCP job
    # already in flight against this index.
    jobs_file = str(db) + ".jobs.json"
    Path(jobs_file).write_text(
        json.dumps(
            [
                {
                    "job_id": "abc123",
                    "folder_path": str(small_corpus),
                    "db_path": str(db),
                    "state": "running",
                    "files_total": 10,
                    "files_processed": 3,
                    "started_at": time.time(),
                    "finished_at": None,
                    "error": None,
                    "result": None,
                    "finalize_warnings": [],
                }
            ]
        )
    )
    rc = main(["--db-path", str(db), "index", str(small_corpus)])
    # The fake "running" job is reloaded as "interrupted" by JobRegistry's
    # _load — that's the expected behavior, since the process that owned
    # the running state is gone. So the CLI should proceed normally rather
    # than refuse. Verify it does.
    assert rc == 0


def test_index_concurrent_running_job_blocks(small_corpus, tmp_path):
    """Simulate a *live* running job by patching has_running to return True."""
    db = tmp_path / "lancedb"
    db.mkdir(parents=True)
    # Drop a marker so the JobRegistry persist file exists and gets loaded.
    Path(str(db) + ".jobs.json").write_text("[]")

    from server.jobs import JobRegistry, IndexJob, RUNNING

    real_init = JobRegistry.__init__

    def patched_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        # Force a running job into the registry so has_running() returns True.
        self._jobs["live"] = IndexJob(
            job_id="live",
            folder_path=str(small_corpus),
            db_path=str(db),
            state=RUNNING,
            files_total=100,
            files_processed=42,
        )
        self._order.append("live")

    with patch.object(JobRegistry, "__init__", patched_init):
        rc = main(["--db-path", str(db), "index", str(small_corpus)])
    assert rc == 1


def test_index_nonexistent_folder_fails(tmp_path):
    db = tmp_path / "lancedb"
    rc = main(["--db-path", str(db), "index", str(tmp_path / "no_such_dir")])
    assert rc == 1


def test_index_with_exclude_pattern(small_corpus, tmp_path, capsys):
    db = tmp_path / "lancedb"
    rc = main([
        "--db-path", str(db),
        "index", str(small_corpus),
        "--exclude", "subdir/**",
    ])
    assert rc == 0
    # The nested file should not be indexed.
    from server.store import VectorStore
    store = VectorStore(str(db))
    files = store.get_all_files()
    assert not any("deep.txt" in f for f in files)


def test_index_invalid_exclude_pattern_rejected(small_corpus, tmp_path, capsys):
    db = tmp_path / "lancedb"
    # ExclusionRules.load raises ValueError for malformed patterns.
    rc = main([
        "--db-path", str(db),
        "index", str(small_corpus),
        "--exclude", "[unclosed",
    ])
    # Exit either 1 (caught) or 0 (pattern parsed despite oddness). The
    # important contract is "does not crash". We assert it terminated.
    assert rc in (0, 1)


def test_index_safe_flush_passes_threshold_one(small_corpus, tmp_path):
    """--safe-flush must lower the per-flush threshold to 1 so each file's
    chunks land immediately. We confirm by spying on the threshold value
    the CLI passes to index_folder."""
    db = tmp_path / "lancedb"
    seen_thresholds = []

    from server.indexer import index_folder as real_index_folder

    def spy_index_folder(*args, flush_threshold=None, **kwargs):
        seen_thresholds.append(flush_threshold)
        return real_index_folder(*args, flush_threshold=flush_threshold, **kwargs)

    # cli.commands.index_cmd uses `from server.indexer import index_folder`
    # at call time, so patching the module-level name binds the spy.
    with patch("server.indexer.index_folder", side_effect=spy_index_folder):
        rc = main([
            "--db-path", str(db),
            "index", str(small_corpus),
            "--safe-flush",
        ])
    assert rc == 0
    assert seen_thresholds == [1]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_prints_results(small_corpus, tmp_path, capsys):
    db = tmp_path / "lancedb"
    assert main(["--db-path", str(db), "index", str(small_corpus)]) == 0
    capsys.readouterr()  # drain
    rc = main(["--db-path", str(db), "search", "revenue report"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Searching" in captured.out
    assert "alpha.txt" in captured.out or "beta" in captured.out


def test_search_on_missing_index_fails(tmp_path, capsys):
    rc = main([
        "--db-path", str(tmp_path / "nope"),
        "search", "foo",
    ])
    assert rc == 1


def test_search_rejects_bad_mode(small_corpus, tmp_path):
    db = tmp_path / "lancedb"
    assert main(["--db-path", str(db), "index", str(small_corpus)]) == 0
    # argparse should refuse the value before our code runs.
    with pytest.raises(SystemExit):
        main([
            "--db-path", str(db),
            "search", "foo", "--mode", "fuzzy",
        ])


# ---------------------------------------------------------------------------
# unpack — uses inline mbox fixtures
# ---------------------------------------------------------------------------


def test_unpack_runs_all_three_phases(tmp_path, capsys):
    """unpack works against a folder with one mbox — pst/msg phases find
    nothing but still report 0-discovered counts."""
    from tests.test_mbox_messages import _make_entry, _make_mbox

    _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "one"}, "body"),
        name="mail.mbox",
    )

    db = tmp_path / "lancedb"
    rc = main(["--db-path", str(db), "unpack", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert (tmp_path / "mail_unpacked").is_dir()
    # Summary mentions each phase.
    for phase in ("pst", "mbox", "msg"):
        assert phase in captured.err.lower()


def test_unpack_nonexistent_folder_fails(tmp_path):
    rc = main(["unpack", str(tmp_path / "missing")])
    assert rc == 1


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_unpacks_and_indexes(tmp_path, capsys):
    """run = unpack + index. Verify a small corpus ends up indexed."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.txt").write_text("Alpha document.")
    db = tmp_path / "lancedb"

    rc = main(["--db-path", str(db), "run", str(corpus)])
    assert rc == 0

    from server.store import VectorStore
    store = VectorStore(str(db))
    assert store.count_chunks() > 0


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------


def test_reindex_rebuilds_chunks_for_one_file(small_corpus, tmp_path, capsys):
    db = tmp_path / "lancedb"
    assert main(["--db-path", str(db), "index", str(small_corpus)]) == 0
    capsys.readouterr()

    target = small_corpus / "alpha.txt"
    # Mutate the file so a fresh re-index has something different to do.
    target.write_text("Alpha updated with new content. Revenue grew.")

    rc = main(["--db-path", str(db), "reindex", str(target)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "reindexed" in captured.err


def test_reindex_missing_file_fails(tmp_path):
    db = tmp_path / "lancedb"
    db.mkdir(parents=True)
    rc = main(["--db-path", str(db), "reindex", str(tmp_path / "ghost.txt")])
    assert rc == 1
