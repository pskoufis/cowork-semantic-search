"""Integration tests for the spreadsheet → queue indexing path."""

import json
from pathlib import Path

import openpyxl

from server.indexer import index_folder, compute_file_hash
from server.paths import to_relative
from server.store import VectorStore


def _write_csv(folder: Path, name: str) -> Path:
    p = folder / name
    p.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    return p


def _write_xlsx(folder: Path, name: str, sheets: dict[str, list[list]]) -> Path:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for sheet_name, rows in sheets.items():
        ws = wb.create_sheet(sheet_name)
        for row in rows:
            ws.append(row)
    p = folder / name
    wb.save(p)
    return p


def test_index_folder_enqueues_csv(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    _write_csv(folder, "x.csv")
    db = str((tmp_path / "db").resolve())

    result = index_folder(str(folder), db_path=db)

    assert result["descriptions_queued"] == 1
    assert result["descriptions_sampled"] == 0
    store = VectorStore(db)
    assert store.count_chunks() == 0
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0]["needs"] == ["file"]
    preview = json.loads(pending[0]["preview_json"])
    assert preview["type"] == "csv"


def test_index_folder_enqueues_xlsx_per_sheet(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    _write_xlsx(folder, "biz.xlsx", {
        "Sales": [["a", "b"], [1, 2]],
        "Costs": [["c", "d"], [3, 4]],
    })
    db = str((tmp_path / "db").resolve())

    result = index_folder(str(folder), db_path=db)

    assert result["descriptions_queued"] == 1  # one file (with 3 needs)
    store = VectorStore(db)
    [pending] = store.list_pending()
    assert pending["needs"] == ["sheet:Sales", "sheet:Costs", "file"]


def test_index_folder_skips_dismissed_with_matching_hash(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    csv = _write_csv(folder, "x.csv")
    db = str((tmp_path / "db").resolve())

    h = compute_file_hash(csv)
    rel = to_relative(str(csv), db)
    store = VectorStore(db)
    store.dismiss(rel, h)

    result = index_folder(str(folder), db_path=db)
    assert result["descriptions_queued"] == 0
    assert store.pending_count() == 0


def test_index_folder_reenqueues_when_dismissed_hash_differs(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    csv = _write_csv(folder, "x.csv")
    db = str((tmp_path / "db").resolve())
    rel = to_relative(str(csv), db)
    store = VectorStore(db)
    store.dismiss(rel, "stale_hash_from_old_content")

    result = index_folder(str(folder), db_path=db)
    assert result["descriptions_queued"] == 1


def test_index_folder_evicts_legacy_csv_chunks_then_enqueues(tmp_path):
    """A pre-existing CSV indexed under the legacy raw-row scheme is evicted
    on the first run and replaced with a queue entry."""
    folder = tmp_path / "data"
    folder.mkdir()
    csv = _write_csv(folder, "x.csv")
    db = str((tmp_path / "db").resolve())
    rel = to_relative(str(csv), db)

    # Seed the store with a legacy "raw row text" chunk for this CSV.
    import numpy as np
    from server.store import EMBEDDING_DIM
    vec = np.random.RandomState(0).randn(EMBEDDING_DIM).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    store = VectorStore(db)
    store.add_chunks([{
        "id": "legacy_0",
        "text": "id,name\n1,Alice\n2,Bob",
        "source_file": rel,
        "file_name": "x.csv",
        "file_type": ".csv",
        "folder_path": str(folder.relative_to(folder.parent)),
        "chunk_index": 0,
        "content_hash": "ignored",
        "mtime_ns": 0,
        "file_size": 0,
        "vector": vec,
        # chunk_kind intentionally omitted → defaults to "text" via add_chunks
    }])
    assert store.count_chunks() == 1

    result = index_folder(str(folder), db_path=db)
    # Legacy chunk gone, CSV re-routed through the queue. Fresh handle —
    # LanceDB's cached table handle on `store` won't see writes from the
    # connection that `index_folder` opened internally.
    fresh = VectorStore(db)
    assert fresh.count_chunks() == 0
    assert result["descriptions_queued"] == 1
    assert fresh.pending_count() == 1
