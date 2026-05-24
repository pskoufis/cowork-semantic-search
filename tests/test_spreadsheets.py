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


# --- XLSX preview ----------------------------------------------------------

import openpyxl

from server.spreadsheets import (
    _extract_xlsx_preview,
    UnreadableSpreadsheetError,
)


def _build_xlsx(tmp_path: Path, name: str, sheets: dict[str, list[list]]) -> Path:
    wb = openpyxl.Workbook()
    # remove default Sheet so we control the sheet set fully
    default = wb.active
    wb.remove(default)
    for sheet_name, rows in sheets.items():
        ws = wb.create_sheet(sheet_name)
        for row in rows:
            ws.append(row)
    p = tmp_path / name
    wb.save(p)
    return p


def test_xlsx_preview_multi_sheet(tmp_path: Path):
    xlsx = _build_xlsx(tmp_path, "biz.xlsx", {
        "Sales": [
            ["region", "qty", "revenue"],
            ["EMEA", 10, 1000.0],
            ["APAC", 15, 1500.0],
        ],
        "Costs": [
            ["category", "amount"],
            ["rent", 5000],
            ["salaries", 25000],
        ],
    })
    sheets = _extract_xlsx_preview(xlsx)
    assert [s["name"] for s in sheets] == ["Sales", "Costs"]
    sales = sheets[0]
    assert sales["headers"] == ["region", "qty", "revenue"]
    assert sales["dtypes"] == ["string", "number", "number"]
    assert sales["row_count"] == 2
    # openpyxl's data_only normalises 1000.0 → 1000 when it fits an int
    assert sales["sample_rows"][0] == ["EMEA", "10", "1000"]


def test_xlsx_preview_header_only_sheet(tmp_path: Path):
    xlsx = _build_xlsx(tmp_path, "headers_only.xlsx", {
        "Empty": [["a", "b", "c"]],
    })
    sheets = _extract_xlsx_preview(xlsx)
    assert len(sheets) == 1
    assert sheets[0]["headers"] == ["a", "b", "c"]
    assert sheets[0]["row_count"] == 0
    assert sheets[0]["sample_rows"] == []


def test_xlsx_preview_empty_sheet(tmp_path: Path):
    xlsx = _build_xlsx(tmp_path, "empty_sheet.xlsx", {"Blank": []})
    sheets = _extract_xlsx_preview(xlsx)
    assert sheets[0]["headers"] is None
    assert sheets[0]["row_count"] == 0


def test_xlsx_preview_caps_sample_rows(tmp_path: Path):
    rows = [["h"]] + [[i] for i in range(100)]
    xlsx = _build_xlsx(tmp_path, "many.xlsx", {"S": rows})
    sheets = _extract_xlsx_preview(xlsx)
    assert len(sheets[0]["sample_rows"]) == 10
    assert sheets[0]["row_count"] == 100


def test_xlsx_preview_unreadable_file(tmp_path: Path):
    bad = tmp_path / "bogus.xlsx"
    bad.write_bytes(b"not actually a zip")
    with pytest.raises(UnreadableSpreadsheetError):
        _extract_xlsx_preview(bad)
