"""Integration test for `csemsearch status` against a real (mocked-embed)
index.

This builds an index via the same ``index_folder`` path the MCP server
uses, then drives ``csemsearch status`` against it and asserts on the
printed output. Embedding is mocked to keep the test fast and offline.
"""

from __future__ import annotations

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


@pytest.fixture
def indexed_db(tmp_path, monkeypatch):
    """Build a small index and yield its db_path."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Alpha document. Revenue grew.")
    (corpus / "b.md").write_text("# Beta\n\nSome more text content.")
    db = tmp_path / "lancedb"

    with patch("server.indexer.get_model") as mock_get_model:
        mock_get_model.return_value = type(
            "MockModel", (), {"encode": lambda self, texts, **kw: _fake_embed(texts)}
        )()
        from server.indexer import index_folder

        result = index_folder(str(corpus), db_path=str(db))
        assert result["files_indexed"] == 2

    return db


def test_status_prints_size_chunks_files(indexed_db, capsys):
    rc = main(["--db-path", str(indexed_db), "status"])
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    assert str(indexed_db) in out
    assert "size on disk:" in out
    assert "total chunks:" in out
    assert "total files:" in out


def test_status_cold_start_does_not_crash(tmp_path, capsys):
    nonexistent = tmp_path / "no-index"
    rc = main(["--db-path", str(nonexistent), "status"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "csemsearch index" in captured.out
