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


class UnreadableSpreadsheetError(Exception):
    """Raised when a spreadsheet cannot be opened (corrupt, encrypted, etc).

    The caller treats this as a per-file failure: not enqueued, recorded in
    the file's error list. Encrypted/password-protected files are the
    common case but we deliberately don't try to distinguish — the LLM
    can't describe what we can't open, regardless of cause.
    """


def _cell_to_str(value) -> str:
    """Coerce a cell value to its preview-string form.

    openpyxl returns Python objects (int, float, datetime, bool, None);
    we stringify uniformly so the preview is JSON-encodable and consistent
    with the CSV path.
    """
    if value is None:
        return ""
    return str(value)


def _extract_xlsx_preview(file_path: Path) -> list[dict]:
    """Build per-sheet previews for a workbook.

    Uses openpyxl read_only mode + iter_rows so we never materialise the
    whole sheet. We pull one extra row beyond SAMPLE_ROW_LIMIT to compute
    row_count without iterating the rest (sheet.max_row from read-only mode
    is the workbook's recorded extent, which is reliable here).
    """
    import openpyxl
    from openpyxl.utils.exceptions import InvalidFileException
    import zipfile

    try:
        wb = openpyxl.load_workbook(
            filename=str(file_path), read_only=True, data_only=True,
        )
    except (InvalidFileException, zipfile.BadZipFile, KeyError) as exc:
        raise UnreadableSpreadsheetError(str(exc)) from exc

    previews: list[dict] = []
    try:
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows_iter = ws.iter_rows(values_only=True)
            collected: list[list[str]] = []
            for row in rows_iter:
                collected.append([_cell_to_str(c) for c in row])
                if len(collected) >= SAMPLE_ROW_LIMIT + 1:
                    break

            if not collected:
                previews.append({
                    "name": sheet_name,
                    "headers": None,
                    "first_row_as_data": True,
                    "dtypes": [],
                    "row_count": 0,
                    "sample_rows": [],
                })
                continue

            # Read-only mode: max_row is the workbook's stored extent. For
            # header-only sheets that's 1; for sheets with N data rows it's
            # N+1 (header + data).
            row_count_total = ws.max_row or len(collected)
            headers = collected[0]
            data_rows = collected[1:]
            sample = data_rows[:SAMPLE_ROW_LIMIT]
            data_count = max(row_count_total - 1, 0)
            n_cols = len(headers)
            dtypes = _column_dtypes(sample, n_cols)

            previews.append({
                "name": sheet_name,
                "headers": headers,
                "first_row_as_data": False,
                "dtypes": dtypes,
                "row_count": data_count,
                "sample_rows": sample,
            })
    finally:
        wb.close()

    return previews
