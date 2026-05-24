"""Preview extraction and prompt building for CSV/XLSX files.

A "preview" is a structured dict the host LLM uses to describe a spreadsheet
without seeing every row. The shape is the same for CSV and XLSX so the
queue and prompt-builders can be uniform.
"""

import csv
import io
from pathlib import Path

SAMPLE_ROW_LIMIT = 10
MAX_DESCRIPTION_BYTES = 4096
SPREADSHEET_EXTENSIONS = {".csv", ".xlsx"}


def _infer_dtype(value: str) -> str:
    """Coarse dtype guess for a cell string. Used for prompt context only —
    not stored, not searchable."""
    v = value.strip()
    if not v:
        return "empty"
    try:
        float(v)
        return "number"
    except ValueError:
        return "string"


def _column_dtypes(sample_rows: list[list[str]], n_cols: int) -> list[str]:
    """Per-column dtype from the first non-empty sample row, fallback 'string'."""
    if not sample_rows or n_cols == 0:
        return []
    dtypes = ["string"] * n_cols
    for col in range(n_cols):
        for row in sample_rows:
            if col < len(row) and row[col].strip():
                dtypes[col] = _infer_dtype(row[col])
                break
    return dtypes


def _extract_csv_preview(file_path: Path) -> dict:
    """Build the preview for a single CSV file (a CSV is one 'sheet')."""
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return {
            "name": file_path.name,
            "headers": None,
            "first_row_as_data": True,
            "dtypes": [],
            "row_count": 0,
            "sample_rows": [],
        }

    # Sniffer needs a non-empty sample; cap to first 64KB so huge files don't
    # hold the whole text in a second buffer just for detection.
    sample = raw[:65536]
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = False

    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return {
            "name": file_path.name,
            "headers": None,
            "first_row_as_data": True,
            "dtypes": [],
            "row_count": 0,
            "sample_rows": [],
        }

    if has_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        headers = None
        data_rows = rows

    sample_rows = data_rows[:SAMPLE_ROW_LIMIT]
    n_cols = len(headers) if headers else (len(sample_rows[0]) if sample_rows else 0)
    dtypes = _column_dtypes(sample_rows, n_cols)

    return {
        "name": file_path.name,
        "headers": headers,
        "first_row_as_data": headers is None,
        "dtypes": dtypes,
        "row_count": len(data_rows),
        "sample_rows": sample_rows,
    }
