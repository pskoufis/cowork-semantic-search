"""Tests for scripts/extract_zips_flat.py — run-log integration.

The flat-extract / collision / marker behaviour is covered by the script's own
logic; these tests pin only that the structured run-log mirrors the per-zip
outcome (ok / skip / fail). Output is written into the input tree, so the
run-log lands under ``<root>/_runlogs/``.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, data in members.items():
            zf.writestr(arcname, data)
    path.write_bytes(buf.getvalue())


def _runlog_items(root: Path, script: str = "extract_zips_flat") -> list[dict]:
    logs = sorted((root / "_runlogs").glob(f"{script}-*.jsonl"))
    assert logs, "no run-log written"
    items = []
    for line in logs[-1].read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["event"] == "item":
            items.append(rec)
    return items


def test_runlog_records_extraction(tmp_path: Path) -> None:
    from scripts.extract_zips_flat import main

    root = tmp_path / "root"
    _write_zip(root / "a.zip", {"sub/leaf.txt": b"x"})

    assert main([str(root)]) == 0
    items = _runlog_items(root)
    assert {it["input"] for it in items} == {"a.zip"}
    assert items[0]["status"] == "ok"


def test_rerun_records_skip(tmp_path: Path) -> None:
    from scripts.extract_zips_flat import main

    root = tmp_path / "root"
    _write_zip(root / "a.zip", {"leaf.txt": b"x"})

    assert main([str(root)]) == 0
    assert main([str(root)]) == 0  # marker makes the second run skip
    assert {it["status"] for it in _runlog_items(root)} == {"skip"}


def test_error_recorded_as_fail(tmp_path: Path) -> None:
    from scripts.extract_zips_flat import main

    root = tmp_path / "root"
    root.mkdir()
    (root / "broken.zip").write_bytes(b"not a real zip")

    assert main([str(root)]) == 1
    items = _runlog_items(root)
    assert items[0]["status"] == "fail"
    assert items[0]["error_type"]


def test_no_runlog_flag(tmp_path: Path) -> None:
    from scripts.extract_zips_flat import main

    root = tmp_path / "root"
    _write_zip(root / "a.zip", {"leaf.txt": b"x"})

    assert main([str(root), "--no-runlog"]) == 0
    assert not (root / "_runlogs").exists()


def test_dry_run_writes_no_runlog(tmp_path: Path) -> None:
    from scripts.extract_zips_flat import main

    root = tmp_path / "root"
    _write_zip(root / "a.zip", {"leaf.txt": b"x"})

    assert main([str(root), "--dry-run"]) == 0
    assert not (root / "_runlogs").exists()
