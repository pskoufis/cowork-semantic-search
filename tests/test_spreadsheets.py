"""Unit tests for spreadsheet preview extraction and prompt building."""

from pathlib import Path

import pytest

from server.spreadsheets import _extract_csv_preview


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_csv_preview_with_header(tmp_path: Path):
    csv = _write(
        tmp_path, "customers.csv",
        "id,name,amount\n1,Alice,10.5\n2,Bob,20\n3,Carol,33.7\n",
    )
    sheet = _extract_csv_preview(csv)
    assert sheet["name"] == "customers.csv"
    assert sheet["headers"] == ["id", "name", "amount"]
    assert sheet["first_row_as_data"] is False
    assert sheet["dtypes"] == ["number", "string", "number"]
    assert sheet["row_count"] == 3
    assert sheet["sample_rows"][0] == ["1", "Alice", "10.5"]
    assert len(sheet["sample_rows"]) == 3


def test_csv_preview_no_header_sniffer_falls_back(tmp_path: Path):
    # All-numeric rows: Sniffer.has_header returns False
    csv = _write(tmp_path, "nums.csv", "1,2,3\n4,5,6\n7,8,9\n")
    sheet = _extract_csv_preview(csv)
    assert sheet["headers"] is None
    assert sheet["first_row_as_data"] is True
    assert sheet["row_count"] == 3


def test_csv_preview_empty_file(tmp_path: Path):
    csv = _write(tmp_path, "empty.csv", "")
    sheet = _extract_csv_preview(csv)
    assert sheet["headers"] is None
    assert sheet["row_count"] == 0
    assert sheet["sample_rows"] == []


def test_csv_preview_caps_sample_rows(tmp_path: Path):
    # Multi-column with a clear string header so Sniffer reliably detects it.
    rows = "name,value\n" + "\n".join(f"item{i},{i}" for i in range(100)) + "\n"
    csv = _write(tmp_path, "many.csv", rows)
    sheet = _extract_csv_preview(csv)
    assert len(sheet["sample_rows"]) == 10
    assert sheet["row_count"] == 100
