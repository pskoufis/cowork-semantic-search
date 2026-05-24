# Spreadsheet description-based indexing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw CSV/XLSX content indexing with LLM-generated descriptions sourced from the host MCP client via `ctx.sample()`, with a queue-and-drain fallback for clients lacking sampling support.

**Architecture:** `index_folder` always enqueues spreadsheets at file granularity (one queue row per file with a `needs` list of sub-items and a JSON-encoded preview). The async `_run_index_job` wrapper then auto-drains the queue via `ctx.sample()` when the client advertises sampling capability. Three new MCP tools (`list_pending_descriptions`, `submit_description`, `dismiss_pending_description`) let the host LLM drain manually when auto-drain is unavailable or fails. Both paths produce identical on-disk chunks with new `chunk_kind` (`sheet_description` / `file_description`) and nullable `sheet_name` fields.

**Tech Stack:** Python 3.11+, FastMCP, LanceDB, openpyxl (new), Python stdlib `csv` (existing).

**Spec:** `docs/superpowers/specs/2026-05-23-spreadsheet-descriptions-design.md`

---

## File Structure

- **Create:** `server/spreadsheets.py` — preview extraction (CSV + XLSX), prompt builders. Pure functions, no I/O beyond reading the file under inspection.
- **Create:** `tests/test_spreadsheets.py` — unit tests for the new module.
- **Create:** `tests/test_indexer_spreadsheets.py` — integration tests covering queue, auto-drain, dismissal.
- **Modify:** `pyproject.toml` — add `openpyxl` dependency.
- **Modify:** `server/parsers.py` — add `.xlsx` to `SUPPORTED_EXTENSIONS`; remove `_extract_csv` arm (indexer routes spreadsheets directly).
- **Modify:** `server/store.py` — schema additions (`chunk_kind`, `sheet_name`), `ensure_schema` migration, new `pending_descriptions` and `dismissed_files` tables, `evict_legacy_spreadsheet_chunks` helper.
- **Modify:** `server/indexer.py` — route spreadsheets to enqueue path, integrate eviction pass, add `descriptions_queued` / `descriptions_sampled` counters.
- **Modify:** `server/main.py` — three new MCP tools; auto-drainer logic in `_run_index_job`; `reindex_file` routes spreadsheets to enqueue.
- **Modify:** `tests/test_store.py` — schema migration + table tests.
- **Modify:** `tests/test_parsers.py` — assert `.xlsx` in `SUPPORTED_EXTENSIONS`, assert `.csv`/`.xlsx` no longer extractable via `extract_text`.
- **Modify:** `tests/test_mcp_tools.py` — smoke tests for the three new tools and `reindex_file` routing.

---

## Task 1: Add openpyxl dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add openpyxl to project dependencies**

Open `pyproject.toml` and add `"openpyxl>=3.1,<4"` to the `dependencies` array. (Find the existing array — it currently lists fastmcp, lancedb, sentence-transformers, etc. Add the entry in alphabetical order.)

- [ ] **Step 2: Sync the lockfile**

Run: `uv sync`
Expected: `uv.lock` updated; `openpyxl` and its transitive dep `et-xmlfile` resolved without error.

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "import openpyxl; print(openpyxl.__version__)"`
Expected: a version string `3.1.x` or similar prints.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add openpyxl for xlsx preview extraction"
```

---

## Task 2: spreadsheets module — CSV preview extraction

**Files:**
- Create: `server/spreadsheets.py`
- Create: `tests/test_spreadsheets.py`

- [ ] **Step 1: Write failing tests for `_extract_csv_preview`**

Create `tests/test_spreadsheets.py`:

```python
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
    rows = "h\n" + "\n".join(str(i) for i in range(100)) + "\n"
    csv = _write(tmp_path, "many.csv", rows)
    sheet = _extract_csv_preview(csv)
    assert len(sheet["sample_rows"]) == 10
    assert sheet["row_count"] == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_spreadsheets.py -v`
Expected: ImportError on `from server.spreadsheets import _extract_csv_preview`.

- [ ] **Step 3: Implement `_extract_csv_preview`**

Create `server/spreadsheets.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_spreadsheets.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/spreadsheets.py tests/test_spreadsheets.py
git commit -m "feat(spreadsheets): csv preview extraction with sniffer-based header detection"
```

---

## Task 3: spreadsheets module — XLSX preview extraction

**Files:**
- Modify: `server/spreadsheets.py`
- Modify: `tests/test_spreadsheets.py`

- [ ] **Step 1: Write failing tests for `_extract_xlsx_preview`**

Append to `tests/test_spreadsheets.py`:

```python
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
    assert sales["sample_rows"][0] == ["EMEA", "10", "1000.0"]


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_spreadsheets.py -v`
Expected: ImportError on `_extract_xlsx_preview` / `UnreadableSpreadsheetError`.

- [ ] **Step 3: Implement XLSX preview extraction**

Append to `server/spreadsheets.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_spreadsheets.py -v`
Expected: all tests passing (4 prior + 5 new = 9).

- [ ] **Step 5: Commit**

```bash
git add server/spreadsheets.py tests/test_spreadsheets.py
git commit -m "feat(spreadsheets): xlsx preview extraction via openpyxl read-only"
```

---

## Task 4: spreadsheets module — dispatcher + prompt builders

**Files:**
- Modify: `server/spreadsheets.py`
- Modify: `tests/test_spreadsheets.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_spreadsheets.py`:

```python
from server.spreadsheets import (
    extract_preview,
    build_sheet_prompt,
    build_file_prompt,
    needs_for_preview,
    SPREADSHEET_EXTENSIONS,
)


def test_extract_preview_csv(tmp_path: Path):
    csv = _write(tmp_path, "x.csv", "a,b\n1,2\n")
    pv = extract_preview(csv)
    assert pv["type"] == "csv"
    assert len(pv["sheets"]) == 1
    assert pv["sheets"][0]["headers"] == ["a", "b"]


def test_extract_preview_xlsx(tmp_path: Path):
    xlsx = _build_xlsx(tmp_path, "x.xlsx", {"S": [["a", "b"], [1, 2]]})
    pv = extract_preview(xlsx)
    assert pv["type"] == "xlsx"
    assert pv["sheets"][0]["name"] == "S"


def test_extract_preview_rejects_unknown(tmp_path: Path):
    p = _write(tmp_path, "x.txt", "hello")
    with pytest.raises(ValueError):
        extract_preview(p)


def test_needs_for_csv():
    pv = {"type": "csv", "sheets": [{"name": "x.csv"}]}
    assert needs_for_preview(pv) == ["file"]


def test_needs_for_xlsx():
    pv = {"type": "xlsx", "sheets": [{"name": "Sales"}, {"name": "Costs"}]}
    assert needs_for_preview(pv) == ["sheet:Sales", "sheet:Costs", "file"]


def test_build_sheet_prompt_includes_context():
    pv = {"type": "xlsx", "sheets": [{
        "name": "Sales",
        "headers": ["region", "qty"],
        "dtypes": ["string", "number"],
        "row_count": 1000,
        "sample_rows": [["EMEA", "10"]],
        "first_row_as_data": False,
    }]}
    prompt = build_sheet_prompt(pv, "Sales")
    assert "Sales" in prompt
    assert "region" in prompt
    assert "qty" in prompt
    assert "1000" in prompt
    assert "EMEA" in prompt


def test_build_file_prompt_includes_sheet_descriptions():
    pv = {"type": "xlsx", "sheets": [
        {"name": "Sales", "headers": ["a"], "dtypes": ["string"],
         "row_count": 1, "sample_rows": [], "first_row_as_data": False},
        {"name": "Costs", "headers": ["b"], "dtypes": ["string"],
         "row_count": 1, "sample_rows": [], "first_row_as_data": False},
    ]}
    descs = {"Sales": "Tracks sales by region.", "Costs": "Lists expenses."}
    prompt = build_file_prompt(pv, descs)
    assert "Tracks sales by region." in prompt
    assert "Lists expenses." in prompt
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `uv run pytest tests/test_spreadsheets.py -v`

- [ ] **Step 3: Implement dispatcher + prompt builders**

Append to `server/spreadsheets.py`:

```python
def extract_preview(file_path: Path) -> dict:
    """Build the full preview dict for a spreadsheet.

    Returns {"type": "csv"|"xlsx", "sheets": [sheet_preview, ...]}.
    CSV has exactly one sheet (using the file name). XLSX has one entry
    per worksheet.

    Raises ValueError for unsupported extensions; UnreadableSpreadsheetError
    for files that can't be opened.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return {"type": "csv", "sheets": [_extract_csv_preview(file_path)]}
    if suffix == ".xlsx":
        return {"type": "xlsx", "sheets": _extract_xlsx_preview(file_path)}
    raise ValueError(f"Not a spreadsheet: {suffix}")


def needs_for_preview(preview: dict) -> list[str]:
    """Derive the queue's `needs` list from a preview.

    CSV → one file-level description (the CSV *is* its single sheet, so we
    skip the redundant sheet-level description). XLSX → one description
    per sheet plus a file-level rollup.
    """
    if preview["type"] == "csv":
        return ["file"]
    return [f"sheet:{s['name']}" for s in preview["sheets"]] + ["file"]


def _format_sheet_block(sheet: dict) -> str:
    """Render a single sheet's preview as a markdown-ish block for prompts."""
    headers = sheet["headers"]
    header_line = (
        f"Headers: {headers}" if headers
        else "Headers: (none detected — first row is data)"
    )
    sample_lines = "\n".join(
        f"  {row}" for row in sheet["sample_rows"][:SAMPLE_ROW_LIMIT]
    ) or "  (no sample rows)"
    return (
        f"Sheet: {sheet['name']}\n"
        f"Rows: {sheet['row_count']}, Columns: {len(headers or sheet['sample_rows'][0]) if (headers or sheet['sample_rows']) else 0}\n"
        f"{header_line}\n"
        f"Dtypes: {sheet['dtypes']}\n"
        f"Sample:\n{sample_lines}"
    )


def build_sheet_prompt(preview: dict, sheet_name: str) -> str:
    """LLM prompt for a single sheet's description."""
    sheet = next(s for s in preview["sheets"] if s["name"] == sheet_name)
    return (
        "You are describing one sheet from a spreadsheet so that another "
        "system can later retrieve it by topic. Write a 1–3 sentence "
        "description of what this sheet contains and what it's used for. "
        "Focus on subject matter, not column-level mechanics. No preamble.\n\n"
        f"{_format_sheet_block(sheet)}"
    )


def build_file_prompt(preview: dict, sheet_descriptions: dict[str, str]) -> str:
    """LLM prompt for the file-level rollup description."""
    if preview["type"] == "csv":
        sheet = preview["sheets"][0]
        return (
            "You are describing a CSV file so that another system can later "
            "retrieve it by topic. Write a 1–3 sentence description of what "
            "this file contains and what it's used for. No preamble.\n\n"
            f"{_format_sheet_block(sheet)}"
        )
    desc_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in sheet_descriptions.items()
    )
    return (
        "You are writing the file-level summary for a multi-sheet workbook "
        "so that another system can retrieve it by topic. Write a 2–4 "
        "sentence description of the workbook's purpose and the relationship "
        "between its sheets. No preamble.\n\n"
        f"File: {preview['sheets'][0]['name'] if preview['sheets'] else '(unknown)'}\n"
        f"Sheets and their descriptions:\n{desc_lines}"
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_spreadsheets.py -v`
Expected: all tests passing.

- [ ] **Step 5: Commit**

```bash
git add server/spreadsheets.py tests/test_spreadsheets.py
git commit -m "feat(spreadsheets): preview dispatcher, needs derivation, prompt builders"
```

---

## Task 5: VectorStore — schema additions and migration

**Files:**
- Modify: `server/store.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: Write failing test for the migration**

Add to `tests/test_store.py` (use existing fixtures/patterns; if a `tmp_db` fixture exists, reuse it). New test function:

```python
import pyarrow as pa
import lancedb


def test_ensure_schema_adds_chunk_kind_and_sheet_name(tmp_path):
    """A pre-existing chunks table without the new columns gets them added
    with NULL defaults; existing rows are preserved."""
    db_path = str(tmp_path / "db")
    db = lancedb.connect(db_path)
    old_schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("file_type", pa.string()),
        pa.field("folder_path", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("content_hash", pa.string()),
        pa.field("mtime_ns", pa.int64()),
        pa.field("file_size", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 256)),
    ])
    table = db.create_table("chunks", schema=old_schema)
    table.add([{
        "id": "x_0", "text": "hi", "source_file": "a.txt",
        "file_name": "a.txt", "file_type": ".txt", "folder_path": ".",
        "chunk_index": 0, "content_hash": "deadbeef",
        "mtime_ns": 0, "file_size": 2,
        "vector": [0.0] * 256,
    }])

    from server.store import VectorStore
    store = VectorStore(db_path)
    store.ensure_schema()

    reopened = lancedb.connect(db_path).open_table("chunks")
    names = set(reopened.schema.names)
    assert "chunk_kind" in names
    assert "sheet_name" in names
    # Existing row preserved
    rows = reopened.search().limit(10).to_list()
    assert any(r["id"] == "x_0" for r in rows)
```

- [ ] **Step 2: Run the test — expect failure**

Run: `uv run pytest tests/test_store.py::test_ensure_schema_adds_chunk_kind_and_sheet_name -v`
Expected: AssertionError — `chunk_kind` / `sheet_name` missing from schema.

- [ ] **Step 3: Extend the SCHEMA and `ensure_schema`**

In `server/store.py`, update `SCHEMA`:

```python
SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("file_name", pa.string()),
    pa.field("file_type", pa.string()),
    pa.field("folder_path", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("content_hash", pa.string()),
    pa.field("mtime_ns", pa.int64()),
    pa.field("file_size", pa.int64()),
    pa.field("chunk_kind", pa.string()),
    pa.field("sheet_name", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
])
```

Extend `ensure_schema` to add the two new columns when missing. LanceDB
`add_columns` requires a per-column SQL expression for the default value;
both new columns default to NULL (cast to string), which the application
treats as "text" / "unset" downstream.

```python
def ensure_schema(self) -> None:
    table = self._get_table()
    if table is None:
        return
    existing = set(table.schema.names)
    missing = [
        field for field in (
            pa.field("mtime_ns", pa.int64()),
            pa.field("file_size", pa.int64()),
            pa.field("chunk_kind", pa.string()),
            pa.field("sheet_name", pa.string()),
        )
        if field.name not in existing
    ]
    if missing:
        # LanceDB add_columns takes {col_name: sql_expression}. NULL cast
        # to the column type is the cheapest backfill — application code
        # treats chunk_kind IS NULL as equivalent to "text".
        type_sql = {
            "mtime_ns": "CAST(NULL AS BIGINT)",
            "file_size": "CAST(NULL AS BIGINT)",
            "chunk_kind": "CAST(NULL AS VARCHAR)",
            "sheet_name": "CAST(NULL AS VARCHAR)",
        }
        table.add_columns({f.name: type_sql[f.name] for f in missing})
        self._table = self._db.open_table(TABLE_NAME)
```

Update `add_chunks` to accept the new fields (default to None / "text" if absent):

```python
def add_chunks(self, chunks: list[dict]) -> None:
    table = self._ensure_table()
    rows = []
    for c in chunks:
        rows.append({
            "id": c["id"],
            "text": c["text"],
            "source_file": c["source_file"],
            "file_name": c["file_name"],
            "file_type": c["file_type"],
            "folder_path": c["folder_path"],
            "chunk_index": c["chunk_index"],
            "content_hash": c["content_hash"],
            "mtime_ns": c["mtime_ns"],
            "file_size": c["file_size"],
            "chunk_kind": c.get("chunk_kind", "text"),
            "sheet_name": c.get("sheet_name"),
            "vector": c["vector"],
        })
    table.add(rows)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_store.py::test_ensure_schema_adds_chunk_kind_and_sheet_name -v`
Expected: pass.

- [ ] **Step 5: Run full store test suite to confirm no regressions**

Run: `uv run pytest tests/test_store.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add server/store.py tests/test_store.py
git commit -m "feat(store): add chunk_kind and sheet_name fields with in-place migration"
```

---

## Task 6: VectorStore — pending_descriptions table

**Files:**
- Modify: `server/store.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_store.py`:

```python
def test_pending_descriptions_enqueue_list_remove(tmp_path):
    from server.store import VectorStore
    store = VectorStore(str(tmp_path / "db"))
    store.enqueue_pending(
        file_path="folder/a.xlsx",
        needs=["sheet:Sales", "sheet:Costs", "file"],
        preview_json='{"type":"xlsx"}',
        content_hash="hashA",
    )
    store.enqueue_pending(
        file_path="folder/b.csv",
        needs=["file"],
        preview_json='{"type":"csv"}',
        content_hash="hashB",
    )
    pending = store.list_pending(limit=10)
    assert len(pending) == 2
    paths = {p["file_path"] for p in pending}
    assert paths == {"folder/a.xlsx", "folder/b.csv"}

    store.remove_pending("folder/a.xlsx")
    after = store.list_pending(limit=10)
    assert {p["file_path"] for p in after} == {"folder/b.csv"}
    assert store.pending_count() == 1


def test_pending_descriptions_update_needs(tmp_path):
    from server.store import VectorStore
    store = VectorStore(str(tmp_path / "db"))
    store.enqueue_pending(
        "x.xlsx", ["sheet:A", "sheet:B", "file"], "{}", "h",
    )
    store.update_pending_needs("x.xlsx", ["sheet:B", "file"])
    [entry] = store.list_pending()
    assert entry["needs"] == ["sheet:B", "file"]
```

- [ ] **Step 2: Run tests — expect AttributeError**

Run: `uv run pytest tests/test_store.py -k pending -v`

- [ ] **Step 3: Implement pending_descriptions table**

Add to `server/store.py` (near the top with the other constants):

```python
PENDING_TABLE_NAME = "pending_descriptions"

PENDING_SCHEMA = pa.schema([
    pa.field("file_path", pa.string()),
    pa.field("needs", pa.list_(pa.string())),
    pa.field("preview_json", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("enqueued_at_ns", pa.int64()),
])
```

Add methods to `VectorStore`:

```python
def _get_pending_table(self):
    try:
        return self._db.open_table(PENDING_TABLE_NAME)
    except Exception:
        return None


def _ensure_pending_table(self):
    table = self._get_pending_table()
    if table is None:
        table = self._db.create_table(
            PENDING_TABLE_NAME, schema=PENDING_SCHEMA
        )
    return table


def enqueue_pending(
    self,
    file_path: str,
    needs: list[str],
    preview_json: str,
    content_hash: str,
) -> None:
    import time
    table = self._ensure_pending_table()
    # Idempotent: replace any existing entry for this path
    table.delete(f"file_path = '{_escape(file_path)}'")
    table.add([{
        "file_path": file_path,
        "needs": needs,
        "preview_json": preview_json,
        "content_hash": content_hash,
        "enqueued_at_ns": time.time_ns(),
    }])


def list_pending(
    self, folder_path: str | None = None, limit: int = 20
) -> list[dict]:
    table = self._get_pending_table()
    if table is None or table.count_rows() == 0:
        return []
    query = table.search().limit(limit)
    if folder_path is not None:
        query = query.where(
            f"file_path LIKE '{_escape(folder_path)}%'", prefilter=True,
        )
    return [
        {
            "file_path": r["file_path"],
            "needs": list(r["needs"]),
            "preview_json": r["preview_json"],
            "content_hash": r["content_hash"],
        }
        for r in query.to_list()
    ]


def remove_pending(self, file_path: str) -> None:
    table = self._get_pending_table()
    if table is None:
        return
    table.delete(f"file_path = '{_escape(file_path)}'")


def update_pending_needs(self, file_path: str, new_needs: list[str]) -> None:
    """Rewrite an entry's `needs` list. LanceDB doesn't support updating a
    list-typed column in place, so we delete+reinsert preserving the rest."""
    table = self._get_pending_table()
    if table is None:
        return
    existing = (
        table.search()
        .where(f"file_path = '{_escape(file_path)}'", prefilter=True)
        .limit(1)
        .to_list()
    )
    if not existing:
        return
    row = existing[0]
    table.delete(f"file_path = '{_escape(file_path)}'")
    table.add([{
        "file_path": row["file_path"],
        "needs": new_needs,
        "preview_json": row["preview_json"],
        "content_hash": row["content_hash"],
        "enqueued_at_ns": row["enqueued_at_ns"],
    }])


def pending_count(self) -> int:
    table = self._get_pending_table()
    if table is None:
        return 0
    return table.count_rows()


def get_pending_entry(self, file_path: str) -> dict | None:
    table = self._get_pending_table()
    if table is None:
        return None
    rows = (
        table.search()
        .where(f"file_path = '{_escape(file_path)}'", prefilter=True)
        .limit(1)
        .to_list()
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "file_path": r["file_path"],
        "needs": list(r["needs"]),
        "preview_json": r["preview_json"],
        "content_hash": r["content_hash"],
    }
```

- [ ] **Step 4: Run pending tests**

Run: `uv run pytest tests/test_store.py -k pending -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add server/store.py tests/test_store.py
git commit -m "feat(store): pending_descriptions table with enqueue/list/remove/update"
```

---

## Task 7: VectorStore — dismissed_files table and one-shot eviction

**Files:**
- Modify: `server/store.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_store.py`:

```python
def test_dismissal_lifecycle(tmp_path):
    from server.store import VectorStore
    store = VectorStore(str(tmp_path / "db"))
    assert not store.is_dismissed("x.csv", "hashA")
    store.dismiss("x.csv", "hashA")
    assert store.is_dismissed("x.csv", "hashA")
    # different hash = not dismissed (file changed)
    assert not store.is_dismissed("x.csv", "hashB")
    store.clear_dismissal("x.csv")
    assert not store.is_dismissed("x.csv", "hashA")


def test_evict_legacy_spreadsheet_chunks(tmp_path):
    from server.store import VectorStore
    store = VectorStore(str(tmp_path / "db"))
    # Seed: 2 CSV text chunks, 1 PDF text chunk, 1 CSV file_description chunk
    store.add_chunks([
        {
            "id": "csv_0", "text": "row data", "source_file": "a.csv",
            "file_name": "a.csv", "file_type": ".csv", "folder_path": ".",
            "chunk_index": 0, "content_hash": "h", "mtime_ns": 0, "file_size": 1,
            "vector": [0.0] * 256,
        },
        {
            "id": "csv_1", "text": "more rows", "source_file": "b.csv",
            "file_name": "b.csv", "file_type": ".csv", "folder_path": ".",
            "chunk_index": 0, "content_hash": "h", "mtime_ns": 0, "file_size": 1,
            "vector": [0.0] * 256,
        },
        {
            "id": "pdf_0", "text": "pdf content", "source_file": "c.pdf",
            "file_name": "c.pdf", "file_type": ".pdf", "folder_path": ".",
            "chunk_index": 0, "content_hash": "h", "mtime_ns": 0, "file_size": 1,
            "vector": [0.0] * 256,
        },
        {
            "id": "csv_desc", "text": "a sales export", "source_file": "d.csv",
            "file_name": "d.csv", "file_type": ".csv", "folder_path": ".",
            "chunk_index": 0, "content_hash": "h", "mtime_ns": 0, "file_size": 1,
            "chunk_kind": "file_description",
            "vector": [0.0] * 256,
        },
    ])
    evicted = store.evict_legacy_spreadsheet_chunks()
    assert evicted == {"a.csv", "b.csv"}
    remaining_files = set(store.get_all_files())
    assert remaining_files == {"c.pdf", "d.csv"}
```

- [ ] **Step 2: Run tests — expect failure**

Run: `uv run pytest tests/test_store.py -k "dismissal or evict_legacy" -v`

- [ ] **Step 3: Implement dismissed_files + eviction**

Add to `server/store.py`:

```python
DISMISSED_TABLE_NAME = "dismissed_files"

DISMISSED_SCHEMA = pa.schema([
    pa.field("file_path", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("dismissed_at_ns", pa.int64()),
])
```

Add methods to `VectorStore`:

```python
def _get_dismissed_table(self):
    try:
        return self._db.open_table(DISMISSED_TABLE_NAME)
    except Exception:
        return None


def _ensure_dismissed_table(self):
    table = self._get_dismissed_table()
    if table is None:
        table = self._db.create_table(
            DISMISSED_TABLE_NAME, schema=DISMISSED_SCHEMA
        )
    return table


def dismiss(self, file_path: str, content_hash: str) -> None:
    import time
    table = self._ensure_dismissed_table()
    table.delete(f"file_path = '{_escape(file_path)}'")
    table.add([{
        "file_path": file_path,
        "content_hash": content_hash,
        "dismissed_at_ns": time.time_ns(),
    }])


def is_dismissed(self, file_path: str, content_hash: str) -> bool:
    table = self._get_dismissed_table()
    if table is None:
        return False
    rows = (
        table.search()
        .where(
            f"file_path = '{_escape(file_path)}' AND "
            f"content_hash = '{_escape(content_hash)}'",
            prefilter=True,
        )
        .limit(1)
        .to_list()
    )
    return bool(rows)


def clear_dismissal(self, file_path: str) -> None:
    table = self._get_dismissed_table()
    if table is None:
        return
    table.delete(f"file_path = '{_escape(file_path)}'")


def evict_legacy_spreadsheet_chunks(self) -> set[str]:
    """One-shot migration: delete any chunks for .csv/.xlsx files that were
    stored under the legacy raw-row scheme (chunk_kind IS NULL or 'text').

    Returns the set of source_file paths whose chunks were evicted, so the
    caller can force-reprocess them on the next pass even if their content
    hash hasn't changed.
    """
    table = self._get_table()
    if table is None:
        return set()
    n = table.count_rows()
    if n == 0:
        return set()
    rows = (
        table.search()
        .where(
            "(file_type = '.csv' OR file_type = '.xlsx') AND "
            "(chunk_kind IS NULL OR chunk_kind = 'text')",
            prefilter=True,
        )
        .select(["source_file"])
        .limit(n)
        .to_list()
    )
    affected = {r["source_file"] for r in rows}
    if affected:
        table.delete(
            "(file_type = '.csv' OR file_type = '.xlsx') AND "
            "(chunk_kind IS NULL OR chunk_kind = 'text')"
        )
    return affected
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_store.py -k "dismissal or evict_legacy" -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add server/store.py tests/test_store.py
git commit -m "feat(store): dismissed_files table + legacy spreadsheet chunk eviction"
```

---

## Task 8: Indexer — route spreadsheets to enqueue path

**Files:**
- Modify: `server/indexer.py`
- Create: `tests/test_indexer_spreadsheets.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_indexer_spreadsheets.py`:

```python
"""Integration tests for the spreadsheet → queue indexing path."""

import json
from pathlib import Path

import openpyxl
import pytest

from server.indexer import index_folder
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
    db = tmp_path / "db"

    result = index_folder(str(folder), db_path=str(db))

    assert result["descriptions_queued"] == 1
    assert result["descriptions_sampled"] == 0
    store = VectorStore(str(db))
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
    db = tmp_path / "db"

    result = index_folder(str(folder), db_path=str(db))

    assert result["descriptions_queued"] == 1  # one file
    store = VectorStore(str(db))
    [pending] = store.list_pending()
    assert pending["needs"] == ["sheet:Sales", "sheet:Costs", "file"]


def test_index_folder_skips_dismissed_with_matching_hash(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    csv = _write_csv(folder, "x.csv")
    db = tmp_path / "db"

    from server.indexer import compute_file_hash
    from server.paths import to_relative
    h = compute_file_hash(csv)
    rel = to_relative(str(csv), str(db.resolve()))
    store = VectorStore(str(db))
    store.dismiss(rel, h)

    result = index_folder(str(folder), db_path=str(db))
    assert result["descriptions_queued"] == 0
    assert store.pending_count() == 0


def test_index_folder_reenqueues_when_dismissed_hash_differs(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    csv = _write_csv(folder, "x.csv")
    db = tmp_path / "db"
    from server.paths import to_relative
    rel = to_relative(str(csv), str(db.resolve()))
    store = VectorStore(str(db))
    store.dismiss(rel, "stale_hash_from_old_content")

    result = index_folder(str(folder), db_path=str(db))
    assert result["descriptions_queued"] == 1
```

- [ ] **Step 2: Run tests — expect failure (no spreadsheet routing yet)**

Run: `uv run pytest tests/test_indexer_spreadsheets.py -v`
Expected: failures — `descriptions_queued` not in result, etc.

- [ ] **Step 3: Implement spreadsheet routing in `index_folder`**

Edit `server/indexer.py`. Add at the top of the file:

```python
from server.spreadsheets import (
    extract_preview, needs_for_preview, SPREADSHEET_EXTENSIONS,
    UnreadableSpreadsheetError,
)
import json as _json
```

Inside `index_folder`, after `store.ensure_schema()`, add the eviction
pass (initialise the forced-reprocess set up front so the main loop can
consult it):

```python
forced_reindex: set[str] = store.evict_legacy_spreadsheet_chunks()
```

Add the new counters near the existing counter init line:

```python
indexed, skipped, deleted, failed, stats_refreshed = 0, 0, 0, 0, 0
size_skipped = 0
descriptions_queued = 0
descriptions_sampled = 0  # filled in by the async auto-drainer post-call
```

Inside the per-file `try:` block, branch on suffix *before* the
`extract_text` call:

```python
suffix = file_path.suffix.lower()

if suffix in SPREADSHEET_EXTENSIONS:
    # Fast path & dismissal checks for spreadsheets follow the same
    # mtime/hash logic as text files, but routing diverges: we never
    # embed raw cells.
    record = file_index.get(source_rel)
    if (record
            and source_rel not in forced_reindex
            and record["mtime_ns"] == mtime_ns
            and record["file_size"] == size):
        skipped += 1
        continue

    file_hash = compute_file_hash(file_path)

    if (record
            and source_rel not in forced_reindex
            and record["content_hash"] == file_hash):
        store.update_file_stat(source_rel, mtime_ns, size)
        stats_refreshed += 1
        skipped += 1
        continue

    if store.is_dismissed(source_rel, file_hash):
        skipped += 1
        continue

    # New / changed: enqueue. Drop any old chunks first so the file
    # isn't half-old half-new.
    store.delete_by_file(source_rel)
    try:
        preview = extract_preview(file_path)
    except UnreadableSpreadsheetError as exc:
        failed += 1
        errors.append({"file": str(file_path), "error": f"unreadable spreadsheet: {exc}"})
        continue

    needs = needs_for_preview(preview)
    store.enqueue_pending(
        file_path=source_rel,
        needs=needs,
        preview_json=_json.dumps(preview),
        content_hash=file_hash,
    )
    descriptions_queued += 1
    indexed += 1  # count as indexed work for finalize trigger
    forced_reindex.discard(source_rel)
    continue
```

Update the return dict — add to the existing dict literal:

```python
"descriptions_queued": descriptions_queued,
"descriptions_sampled": descriptions_sampled,
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_indexer_spreadsheets.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full indexer test suite to confirm no regressions**

Run: `uv run pytest tests/test_indexer.py tests/test_indexer_spreadsheets.py -v`
Expected: all pass. (The legacy CSV test in `test_indexer.py` may need adjustment if it asserts on raw-row chunks — see Task 14.)

- [ ] **Step 6: Commit**

```bash
git add server/indexer.py tests/test_indexer_spreadsheets.py
git commit -m "feat(indexer): route csv/xlsx to enqueue path, evict legacy chunks on first run"
```

---

## Task 9: MCP tool — list_pending_descriptions

**Files:**
- Modify: `server/main.py`
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_mcp_tools.py` (use the existing test client setup pattern; if there's a fixture for an indexed folder, reuse it):

```python
def test_list_pending_descriptions_returns_enqueued(tmp_path):
    from server.store import VectorStore
    from server.main import list_pending_descriptions

    db = tmp_path / "db"
    store = VectorStore(str(db))
    store.enqueue_pending(
        "folder/a.xlsx",
        ["sheet:Sales", "file"],
        '{"type":"xlsx","sheets":[{"name":"Sales"}]}',
        "h1",
    )
    result = list_pending_descriptions.fn(
        folder_path=None, limit=10, db_path=str(db),
    )
    assert result["total_remaining"] == 1
    assert len(result["pending"]) == 1
    assert result["pending"][0]["file_path"] == "folder/a.xlsx"
    assert result["pending"][0]["needs"] == ["sheet:Sales", "file"]
    assert result["pending"][0]["preview"]["type"] == "xlsx"


def test_list_pending_descriptions_auto_evicts_deleted_files(tmp_path):
    from server.store import VectorStore
    from server.main import list_pending_descriptions
    from server.paths import to_absolute

    db = tmp_path / "db"
    db_dir = str(db.resolve())
    fake_rel = "missing/x.csv"
    store = VectorStore(db_dir)
    store.enqueue_pending(fake_rel, ["file"], '{"type":"csv"}', "h")
    # File doesn't exist on disk
    result = list_pending_descriptions.fn(
        folder_path=None, limit=10, db_path=db_dir,
    )
    assert result["pending"] == []
    assert store.pending_count() == 0  # evicted from the table
```

(Note: `.fn` accesses the underlying function of a FastMCP tool — matches the existing test pattern in `test_mcp_tools.py`. If that file uses a different access path, follow its convention.)

- [ ] **Step 2: Run test — expect ImportError**

Run: `uv run pytest tests/test_mcp_tools.py -k list_pending -v`

- [ ] **Step 3: Implement `list_pending_descriptions`**

Add to `server/main.py` (after the existing tool definitions):

```python
@mcp.tool(annotations={"readOnlyHint": True})
def list_pending_descriptions(
    folder_path: Annotated[
        str | None,
        Field(
            description="Filter to entries whose file_path begins with this "
                        "folder. If omitted, returns entries across all folders.",
            default=None,
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Max entries to return per call", default=20, ge=1, le=100),
    ] = 20,
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """List spreadsheets awaiting LLM-generated descriptions.

    For each pending file, returns the preview (sheet names, headers, sample
    rows) the host LLM needs to produce a description, plus the list of
    items still needed (per-sheet descriptions and/or the file-level rollup).

    Files that have been deleted from disk since they were enqueued are
    auto-evicted from the queue and not surfaced.
    """
    import json as _json
    from pathlib import Path as _Path
    from server.store import VectorStore
    from server.paths import to_absolute

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    store = VectorStore(db_dir)
    raw = store.list_pending(folder_path=folder_path, limit=limit * 2)

    pending: list[dict] = []
    for entry in raw:
        abs_path = _Path(to_absolute(entry["file_path"], db_dir))
        if not abs_path.exists():
            store.remove_pending(entry["file_path"])
            continue
        pending.append({
            "file_path": entry["file_path"],
            "needs": entry["needs"],
            "preview": _json.loads(entry["preview_json"]),
        })
        if len(pending) >= limit:
            break

    return {
        "pending": pending,
        "total_remaining": store.pending_count(),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mcp_tools.py -k list_pending -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_mcp_tools.py
git commit -m "feat(mcp): list_pending_descriptions tool"
```

---

## Task 10: MCP tool — submit_description

**Files:**
- Modify: `server/main.py`
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_mcp_tools.py`:

```python
def test_submit_description_csv_completes_file(tmp_path):
    from server.store import VectorStore
    from server.main import submit_description
    from server.paths import to_relative

    folder = tmp_path / "data"
    folder.mkdir()
    csv = folder / "x.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    db = tmp_path / "db"
    db_dir = str(db.resolve())
    rel = to_relative(str(csv), db_dir)

    store = VectorStore(db_dir)
    store.enqueue_pending(
        rel, ["file"], '{"type":"csv","sheets":[{"name":"x.csv"}]}', "h",
    )

    result = submit_description.fn(
        file_path=str(csv),
        sheet_name=None,
        description="This is a tiny customer-name list.",
        db_path=db_dir,
    )
    assert result["status"] == "stored"
    assert result["file_complete"] is True
    assert store.count_chunks() == 1
    assert store.pending_count() == 0


def test_submit_description_xlsx_incremental(tmp_path):
    from server.store import VectorStore
    from server.main import submit_description
    from server.paths import to_relative
    import openpyxl

    folder = tmp_path / "data"
    folder.mkdir()
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    wb.create_sheet("Sales").append(["a", "b"])
    wb.create_sheet("Costs").append(["c", "d"])
    xlsx = folder / "biz.xlsx"
    wb.save(xlsx)
    db = tmp_path / "db"
    db_dir = str(db.resolve())
    rel = to_relative(str(xlsx), db_dir)

    store = VectorStore(db_dir)
    store.enqueue_pending(
        rel,
        ["sheet:Sales", "sheet:Costs", "file"],
        '{"type":"xlsx","sheets":[{"name":"Sales"},{"name":"Costs"}]}',
        "h",
    )

    r1 = submit_description.fn(
        file_path=str(xlsx), sheet_name="Sales",
        description="Sales by region", db_path=db_dir,
    )
    assert r1["file_complete"] is False
    assert store.count_chunks() == 1

    r2 = submit_description.fn(
        file_path=str(xlsx), sheet_name="Costs",
        description="Costs by category", db_path=db_dir,
    )
    assert r2["file_complete"] is False

    r3 = submit_description.fn(
        file_path=str(xlsx), sheet_name=None,
        description="Sales and costs ledger.", db_path=db_dir,
    )
    assert r3["file_complete"] is True
    assert store.count_chunks() == 3
    assert store.pending_count() == 0


def test_submit_description_rejects_unknown_sheet(tmp_path):
    from server.store import VectorStore
    from server.main import submit_description

    folder = tmp_path / "data"
    folder.mkdir()
    csv = folder / "x.csv"
    csv.write_text("a,b\n", encoding="utf-8")
    db = tmp_path / "db"
    db_dir = str(db.resolve())
    from server.paths import to_relative
    rel = to_relative(str(csv), db_dir)

    store = VectorStore(db_dir)
    store.enqueue_pending(rel, ["file"], '{"type":"csv"}', "h")

    result = submit_description.fn(
        file_path=str(csv),
        sheet_name="NotInFile",
        description="x",
        db_path=db_dir,
    )
    assert result["status"] == "rejected"
    assert store.pending_count() == 1
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `uv run pytest tests/test_mcp_tools.py -k submit_description -v`

- [ ] **Step 3: Implement the shared submission helper and the tool**

Add to `server/main.py`:

```python
def _submit_one_description(
    *,
    db_dir: str,
    file_path: str,
    sheet_name: str | None,
    description: str,
) -> dict:
    """Embed and store one description chunk. Shared between the
    submit_description tool and the auto-drainer.

    Returns {status, file_complete, reason?}. Does NOT raise on
    expected rejection paths (validation, missing file) — surfaces them
    via the status field so tools can return them cleanly.
    """
    from pathlib import Path
    from server.store import VectorStore
    from server.paths import to_relative
    from server.indexer import embed_chunks, compute_file_hash
    from server.chunker import _short_hash
    from server.spreadsheets import MAX_DESCRIPTION_BYTES

    path = Path(file_path)
    if not path.exists():
        return {"status": "rejected", "reason": "file_not_found"}

    if not description or not description.strip():
        return {"status": "rejected", "reason": "empty_description"}
    if len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
        return {"status": "rejected", "reason": "description_too_large"}

    source_rel = to_relative(str(path), db_dir)
    store = VectorStore(db_dir)
    store.ensure_schema()

    entry = store.get_pending_entry(source_rel)
    if entry is None:
        return {"status": "rejected", "reason": "not_pending"}

    needed_item = "file" if sheet_name is None else f"sheet:{sheet_name}"
    if needed_item not in entry["needs"]:
        return {"status": "rejected", "reason": "item_not_needed"}

    # Original enqueue-time `needs` defines chunk_index slots so re-orderings
    # don't collide. We compute the slot from preview ordering, which is the
    # canonical order.
    import json as _json
    preview = _json.loads(entry["preview_json"])
    from server.spreadsheets import needs_for_preview
    full_needs = needs_for_preview(preview)
    slot = full_needs.index(needed_item)

    stat = path.stat()
    file_hash = compute_file_hash(path)
    chunk_kind = "file_description" if sheet_name is None else "sheet_description"

    chunk = {
        "id": f"{_short_hash(source_rel)}_{slot}",
        "text": description.strip(),
        "source_file": source_rel,
        "file_name": path.name,
        "file_type": path.suffix.lower(),
        "folder_path": os.path.dirname(source_rel) or ".",
        "chunk_index": slot,
        "content_hash": file_hash,
        "mtime_ns": stat.st_mtime_ns,
        "file_size": stat.st_size,
        "chunk_kind": chunk_kind,
        "sheet_name": sheet_name,
    }
    [chunk] = embed_chunks([chunk])
    store.add_chunks([chunk])

    remaining = [n for n in entry["needs"] if n != needed_item]
    if not remaining:
        store.remove_pending(source_rel)
        return {"status": "stored", "file_complete": True}
    store.update_pending_needs(source_rel, remaining)
    return {"status": "stored", "file_complete": False}


@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False}
)
def submit_description(
    file_path: Annotated[str, Field(description="Absolute path to the spreadsheet")],
    sheet_name: Annotated[
        str | None,
        Field(
            description="Name of the sheet being described. Pass null for "
                        "the file-level rollup. For CSV, always pass null.",
            default=None,
        ),
    ] = None,
    description: Annotated[
        str,
        Field(description="The LLM-generated description text"),
    ] = "",
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Store a description for a queued spreadsheet sheet or file-level rollup.

    Embeds the description and writes a chunk with chunk_kind set to
    sheet_description or file_description (and sheet_name set for sheet
    descriptions). Removes the corresponding entry from the pending queue;
    when the file's last needed description lands, the file's queue entry
    is removed entirely.
    """
    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)
    return _submit_one_description(
        db_dir=db_dir,
        file_path=file_path,
        sheet_name=sheet_name,
        description=description,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mcp_tools.py -k submit_description -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_mcp_tools.py
git commit -m "feat(mcp): submit_description tool with shared submission helper"
```

---

## Task 11: MCP tool — dismiss_pending_description

**Files:**
- Modify: `server/main.py`
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_mcp_tools.py`:

```python
def test_dismiss_removes_from_queue_and_records_hash(tmp_path):
    from server.store import VectorStore
    from server.main import dismiss_pending_description
    from server.paths import to_relative
    from server.indexer import compute_file_hash

    folder = tmp_path / "data"
    folder.mkdir()
    csv = folder / "junk.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    db = tmp_path / "db"
    db_dir = str(db.resolve())
    rel = to_relative(str(csv), db_dir)
    h = compute_file_hash(csv)

    store = VectorStore(db_dir)
    store.enqueue_pending(rel, ["file"], '{"type":"csv"}', h)

    result = dismiss_pending_description.fn(
        file_path=str(csv), db_path=db_dir,
    )
    assert result["status"] == "dismissed"
    assert store.pending_count() == 0
    assert store.is_dismissed(rel, h)


def test_dismiss_unknown_file_returns_not_pending(tmp_path):
    from server.main import dismiss_pending_description

    db = tmp_path / "db"
    result = dismiss_pending_description.fn(
        file_path=str(tmp_path / "nope.csv"), db_path=str(db),
    )
    assert result["status"] == "rejected"
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `uv run pytest tests/test_mcp_tools.py -k dismiss -v`

- [ ] **Step 3: Implement the tool**

Add to `server/main.py`:

```python
@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}
)
def dismiss_pending_description(
    file_path: Annotated[str, Field(description="Absolute path to the spreadsheet to dismiss")],
    db_path: Annotated[
        str | None,
        Field(
            description="Path to the LanceDB database. Uses LANCEDB_PATH env var if omitted.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Remove a file from the pending-descriptions queue and record a
    dismissal keyed by the file's current content hash.

    On future index_folder runs the file is skipped silently as long as the
    hash matches the dismissed hash. If the file's content changes, the
    dismissal becomes stale and the file is re-enqueued automatically.
    """
    from pathlib import Path
    from server.store import VectorStore
    from server.paths import to_relative
    from server.indexer import compute_file_hash

    path = Path(file_path)
    if not path.exists():
        return {"status": "rejected", "reason": "file_not_found"}

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)
    source_rel = to_relative(str(path), db_dir)

    store = VectorStore(db_dir)
    entry = store.get_pending_entry(source_rel)
    if entry is None:
        return {"status": "rejected", "reason": "not_pending"}

    file_hash = compute_file_hash(path)
    store.remove_pending(source_rel)
    store.dismiss(source_rel, file_hash)
    return {"status": "dismissed", "file_path": file_path}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mcp_tools.py -k dismiss -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_mcp_tools.py
git commit -m "feat(mcp): dismiss_pending_description tool"
```

---

## Task 12: Auto-drainer in `_run_index_job`

**Files:**
- Modify: `server/main.py`
- Modify: `tests/test_indexer_spreadsheets.py`

**API reference:** FastMCP exposes the host LLM sampling via the Context's
`ctx.sample(...)` async method. Inside `_run_index_job` we don't have a
Context directly (the run is kicked off by `index_folder`, which doesn't
receive ctx). Solution: thread an *optional* Context into the wrapper at
job creation time — `index_folder` MCP tool already has access to `ctx`
via FastMCP injection if we declare it as a parameter. The current
signature doesn't take ctx; this task adds it.

- [ ] **Step 1: Write failing test (auto-drain via mock ctx)**

Add to `tests/test_indexer_spreadsheets.py`:

```python
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock


def test_auto_drainer_writes_chunks_via_mock_sampling(tmp_path):
    """When ctx.sample is available, the auto-drainer turns queued entries
    into description chunks and clears the queue."""
    from server.main import _run_index_job
    from server.jobs import registry
    from server.store import VectorStore

    folder = tmp_path / "data"
    folder.mkdir()
    xlsx = _write_xlsx(folder, "biz.xlsx", {
        "Sales": [["region", "qty"], ["EMEA", 10]],
        "Costs": [["category", "amount"], ["rent", 5000]],
    })
    db = str((tmp_path / "db").resolve())

    # Mock ctx with an async sample() returning canned descriptions
    ctx = MagicMock()
    canned = iter([
        MagicMock(text="Sales by region."),
        MagicMock(text="Cost breakdown by category."),
        MagicMock(text="Workbook tracking sales vs costs."),
    ])
    ctx.sample = AsyncMock(side_effect=lambda *a, **kw: next(canned))
    # Indicate the client supports sampling
    ctx.session = MagicMock()
    ctx.session.check_client_capability = MagicMock(return_value=True)

    job = registry.create(str(folder), db)
    asyncio.run(_run_index_job(
        job, None, True, None, ctx=ctx,
    ))

    store = VectorStore(db)
    assert store.count_chunks() == 3
    assert store.pending_count() == 0
    final = job.to_dict()
    assert final["result"]["descriptions_sampled"] == 3


def test_auto_drainer_leaves_entry_when_sampling_fails(tmp_path):
    from server.main import _run_index_job
    from server.jobs import registry
    from server.store import VectorStore

    folder = tmp_path / "data"
    folder.mkdir()
    xlsx = _write_xlsx(folder, "biz.xlsx", {
        "Sales": [["a"], [1]],
        "Costs": [["b"], [2]],
    })
    db = str((tmp_path / "db").resolve())

    # First sheet sampling succeeds, second sheet fails twice → file goes back to queue
    calls = []
    async def flaky_sample(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            return MagicMock(text="Sales by region.")
        raise RuntimeError("sampling boom")
    ctx = MagicMock()
    ctx.sample = flaky_sample
    ctx.session = MagicMock()
    ctx.session.check_client_capability = MagicMock(return_value=True)

    job = registry.create(str(folder), db)
    asyncio.run(_run_index_job(job, None, True, None, ctx=ctx))

    store = VectorStore(db)
    # Atomic commit: nothing should have landed for this file
    assert store.count_chunks() == 0
    assert store.pending_count() == 1
```

- [ ] **Step 2: Run tests — expect failure (no ctx param, no auto-drainer)**

Run: `uv run pytest tests/test_indexer_spreadsheets.py -k auto_drain -v`

- [ ] **Step 3: Modify `index_folder` MCP tool + `_run_index_job` to accept ctx**

In `server/main.py`, update the `index_folder` tool signature to accept a
Context. Per FastMCP, declare a `ctx: Context` parameter and it is
injected automatically.

```python
from fastmcp import Context

@mcp.tool(...)
async def index_folder(
    folder_path: Annotated[str, Field(...)],
    # ... existing params ...
    ctx: Context = None,
) -> dict:
    # ... existing body ...
    job.task = asyncio.create_task(
        _run_index_job(job, file_types, recursive, exclude, ctx=ctx)
    )
    # ...
```

Update `_run_index_job` to accept and use ctx:

```python
async def _run_index_job(job, file_types, recursive, exclude, ctx=None) -> None:
    from server.jobs import registry
    from server.indexer import index_folder as _index_folder

    def progress(processed: int, total: int) -> None:
        registry.update_progress(job.job_id, processed, total)

    try:
        result = await asyncio.to_thread(
            _index_folder,
            job.folder_path,
            file_types, recursive, job.db_path,
            progress, exclude,
        )
        # Auto-drain: if any spreadsheets were queued AND the client
        # supports sampling, try to describe them now. Failures leave the
        # entry in the queue for the LLM to drain manually later.
        if result.get("descriptions_queued", 0) > 0 and _ctx_supports_sampling(ctx):
            sampled = await _auto_drain_descriptions(job.db_path, ctx)
            result["descriptions_sampled"] = sampled
        registry.mark_completed(
            job.job_id, result, result.get("finalize_warnings", []),
        )
    except asyncio.CancelledError:
        registry.mark_failed(job.job_id, "cancelled (server shutting down)")
        raise
    except Exception as e:
        registry.mark_failed(job.job_id, str(e))
```

Add the drainer helpers in `server/main.py`:

```python
def _ctx_supports_sampling(ctx) -> bool:
    """Best-effort check that the connected client advertised sampling.

    FastMCP exposes capability via ctx.session.check_client_capability
    (or similar — adjust if the installed version names it differently).
    A failure to introspect is treated as 'unsupported' so we don't raise
    against a non-MCP test ctx.
    """
    if ctx is None:
        return False
    try:
        from mcp.types import SamplingCapability
        return bool(ctx.session.check_client_capability(
            type("ClientCapabilities", (), {"sampling": SamplingCapability()})()
        ))
    except Exception:
        # Test paths may pass a mock with check_client_capability returning True
        try:
            return bool(ctx.session.check_client_capability(None))
        except Exception:
            return False


async def _sample_with_retry(ctx, prompt: str) -> str | None:
    """One retry with 2s backoff. Returns the description text on success,
    None on failure (caller treats None as 'skip this file's commit')."""
    import asyncio
    from server.spreadsheets import MAX_DESCRIPTION_BYTES

    for attempt in range(2):
        try:
            result = await ctx.sample(
                messages=prompt,  # FastMCP accepts a string or list[Message]
                max_tokens=400,
            )
            text = getattr(result, "text", None) or str(result)
            if not text or not text.strip():
                raise ValueError("empty sampling response")
            if len(text.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
                raise ValueError("sampling response over size cap")
            return text.strip()
        except Exception:
            if attempt == 0:
                await asyncio.sleep(2)
                continue
            return None
    return None


async def _auto_drain_descriptions(db_dir: str, ctx) -> int:
    """Drain the pending_descriptions queue using ctx.sample().

    Per-file atomic commit: all of a file's descriptions are buffered and
    written together once the file-level rollup also succeeds. Any failure
    inside a file leaves that file's queue entry intact.

    Returns the number of description chunks written (not files).
    """
    import json as _json
    from pathlib import Path
    from server.store import VectorStore
    from server.paths import to_absolute, to_relative
    from server.indexer import embed_chunks, compute_file_hash
    from server.chunker import _short_hash
    from server.spreadsheets import (
        build_sheet_prompt, build_file_prompt, needs_for_preview,
    )

    store = VectorStore(db_dir)
    written = 0
    # Drain in batches of 20 until empty or all remaining have failed once.
    failed_paths: set[str] = set()
    while True:
        batch = [
            e for e in store.list_pending(limit=20)
            if e["file_path"] not in failed_paths
        ]
        if not batch:
            break
        for entry in batch:
            preview = _json.loads(entry["preview_json"])
            abs_path = Path(to_absolute(entry["file_path"], db_dir))
            if not abs_path.exists():
                store.remove_pending(entry["file_path"])
                continue

            full_needs = needs_for_preview(preview)
            sheet_descriptions: dict[str, str] = {}
            chunks: list[dict] = []
            file_aborted = False

            # Per-sheet first
            for slot, item in enumerate(full_needs):
                if item == "file":
                    continue
                sheet_name = item.removeprefix("sheet:")
                prompt = build_sheet_prompt(preview, sheet_name)
                desc = await _sample_with_retry(ctx, prompt)
                if desc is None:
                    file_aborted = True
                    break
                sheet_descriptions[sheet_name] = desc
                chunks.append(_build_description_chunk(
                    abs_path, entry, slot, desc,
                    chunk_kind="sheet_description", sheet_name=sheet_name,
                ))

            if file_aborted:
                failed_paths.add(entry["file_path"])
                continue

            # File-level rollup
            file_slot = full_needs.index("file")
            file_prompt = build_file_prompt(preview, sheet_descriptions)
            file_desc = await _sample_with_retry(ctx, file_prompt)
            if file_desc is None:
                failed_paths.add(entry["file_path"])
                continue
            chunks.append(_build_description_chunk(
                abs_path, entry, file_slot, file_desc,
                chunk_kind="file_description", sheet_name=None,
            ))

            # Atomic commit
            chunks = embed_chunks(chunks)
            store.add_chunks(chunks)
            store.remove_pending(entry["file_path"])
            written += len(chunks)

    return written


def _build_description_chunk(
    abs_path,
    entry: dict,
    slot: int,
    description: str,
    *,
    chunk_kind: str,
    sheet_name: str | None,
) -> dict:
    import os
    from server.paths import to_relative
    from server.indexer import compute_file_hash
    from server.chunker import _short_hash

    db_dir = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_dir)
    source_rel = entry["file_path"]
    stat = abs_path.stat()
    return {
        "id": f"{_short_hash(source_rel)}_{slot}",
        "text": description,
        "source_file": source_rel,
        "file_name": abs_path.name,
        "file_type": abs_path.suffix.lower(),
        "folder_path": os.path.dirname(source_rel) or ".",
        "chunk_index": slot,
        "content_hash": entry["content_hash"],
        "mtime_ns": stat.st_mtime_ns,
        "file_size": stat.st_size,
        "chunk_kind": chunk_kind,
        "sheet_name": sheet_name,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_indexer_spreadsheets.py -v`
Expected: pass (including the two new auto-drain tests).

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_indexer_spreadsheets.py
git commit -m "feat(mcp): auto-drain pending descriptions via ctx.sample() with atomic per-file commit"
```

---

## Task 13: reindex_file routes spreadsheets to enqueue

**Files:**
- Modify: `server/main.py`
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_mcp_tools.py`:

```python
def test_reindex_file_routes_csv_to_queue(tmp_path):
    from server.main import reindex_file
    from server.store import VectorStore
    from server.paths import to_relative

    folder = tmp_path / "data"
    folder.mkdir()
    csv = folder / "x.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    db = str((tmp_path / "db").resolve())

    result = reindex_file.fn(file_path=str(csv), db_path=db)
    assert result["status"] == "queued"
    store = VectorStore(db)
    assert store.pending_count() == 1
    rel = to_relative(str(csv), db)
    entry = store.get_pending_entry(rel)
    assert entry["needs"] == ["file"]
```

- [ ] **Step 2: Run test — expect failure (current reindex_file extracts text)**

Run: `uv run pytest tests/test_mcp_tools.py -k reindex_file_routes_csv -v`

- [ ] **Step 3: Add spreadsheet routing to `reindex_file`**

In `server/main.py`, modify `reindex_file`. Just before the existing
`parts = extract_text(path)` call, add:

```python
from server.spreadsheets import (
    SPREADSHEET_EXTENSIONS, extract_preview, needs_for_preview,
    UnreadableSpreadsheetError,
)
import json as _json

if path.suffix.lower() in SPREADSHEET_EXTENSIONS:
    try:
        preview = extract_preview(path)
    except UnreadableSpreadsheetError as exc:
        return {
            "status": "failed",
            "file_path": file_path,
            "reason": f"unreadable spreadsheet: {exc}",
        }
    file_hash = compute_file_hash(path)
    store.enqueue_pending(
        file_path=source_rel,
        needs=needs_for_preview(preview),
        preview_json=_json.dumps(preview),
        content_hash=file_hash,
    )
    return {
        "status": "queued",
        "file_path": file_path,
        "needs": needs_for_preview(preview),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mcp_tools.py -v`
Expected: pass (no regressions in other reindex_file tests).

- [ ] **Step 5: Commit**

```bash
git add server/main.py tests/test_mcp_tools.py
git commit -m "feat(mcp): reindex_file routes csv/xlsx to enqueue path"
```

---

## Task 14: Cleanup — remove `_extract_csv`, add `.xlsx` to SUPPORTED_EXTENSIONS

**Files:**
- Modify: `server/parsers.py`
- Modify: `tests/test_parsers.py`

After the spreadsheet path is in place, `_extract_csv` is unreachable
through the indexer or reindex_file. Remove it so the parsers module
doesn't carry dead code.

- [ ] **Step 1: Update test_parsers.py**

Modify `tests/test_parsers.py`:
1. Find the existing test that uses `extract_text` on a CSV. Replace it with a test that asserts `extract_text` raises `ValueError` for `.csv` and `.xlsx`.
2. Add an assertion that `SUPPORTED_EXTENSIONS` contains both `.csv` and `.xlsx`.

Concrete patch:

```python
def test_extract_text_rejects_spreadsheet_extensions(tmp_path):
    """CSV/XLSX no longer route through extract_text — they're handled by
    server.spreadsheets via the queue path. extract_text should raise."""
    from server.parsers import extract_text, SUPPORTED_EXTENSIONS

    assert ".csv" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS

    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        extract_text(csv)

    xlsx = tmp_path / "x.xlsx"
    xlsx.write_text("not a real xlsx")
    with pytest.raises(ValueError):
        extract_text(xlsx)
```

Remove any prior test in `test_parsers.py` that asserted `extract_text`
produces text for a CSV.

- [ ] **Step 2: Run the new test — expect failure (CSV still routes through extract_text)**

Run: `uv run pytest tests/test_parsers.py::test_extract_text_rejects_spreadsheet_extensions -v`

- [ ] **Step 3: Update `server/parsers.py`**

In `SUPPORTED_EXTENSIONS`, add `.xlsx`:

```python
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".pptx", ".csv", ".xlsx", ".pst"}
```

Remove the `.csv` arm from the `match` statement in `extract_text`:

```python
match suffix:
    case ".txt" | ".md":
        ...
    case ".pdf":
        ...
    case ".docx":
        ...
    case ".pptx":
        ...
    case ".pst":
        ...
    case _:
        raise ValueError(...)
```

Delete the `_extract_csv` function entirely.

- [ ] **Step 4: Run the parsers tests**

Run: `uv run pytest tests/test_parsers.py -v`
Expected: all pass.

- [ ] **Step 5: Run the entire test suite for one final regression check**

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add server/parsers.py tests/test_parsers.py
git commit -m "chore(parsers): drop legacy _extract_csv (spreadsheets now route via queue path)"
```

---

## Self-Review

Spec sections cross-referenced against tasks:

| Spec section | Covered by |
|---|---|
| Schema changes (chunk_kind, sheet_name) | Task 5 |
| pending_descriptions table | Task 6 |
| dismissed_files table | Task 7 |
| One-shot eviction of legacy CSV chunks | Task 7 (helper) + Task 8 (integration) |
| XLSX preview extraction | Task 3 |
| CSV preview extraction | Task 2 |
| Indexing flow XLSX/CSV (replace, not augment) | Task 8 (enqueue), Task 12 (auto-drain) |
| Sampling failure routing (capability + per-call + atomic commit) | Task 12 |
| `list_pending_descriptions` MCP tool | Task 9 |
| `submit_description` MCP tool | Task 10 |
| `dismiss_pending_description` MCP tool | Task 11 |
| `index_folder` unchanged signature + new counters | Task 8 (counters), Task 12 (`ctx` is FastMCP-injected and doesn't change the public signature) |
| Auto-evict deleted-file queue entries | Task 9 |
| Edge case: encrypted XLSX | Task 3 (UnreadableSpreadsheetError) |
| Edge case: header-only sheet | Task 3 |
| Edge case: huge sheet | Task 3 (read_only mode + bounded sample) |
| Reindex_file routes spreadsheets | Task 13 |
| Cleanup: drop `_extract_csv` | Task 14 |

No spec gaps detected.

**Type consistency:** `needs_for_preview` returns `list[str]` with items
`"sheet:<name>"` or `"file"`. Used consistently in the indexer enqueue,
the queue table, `submit_description`'s `needed_item` lookup, and the
auto-drainer. `chunk_kind` values `"sheet_description"` / `"file_description"`
are used consistently. `MAX_DESCRIPTION_BYTES` defined once in
`server/spreadsheets.py` and imported by `_submit_one_description` and
`_sample_with_retry`.

**One known caveat to verify at implementation time:** FastMCP's
`ctx.sample()` API surface (parameter names for prompts, capability check
introspection). The plan codes against a plausible shape; the implementer
should verify against the installed FastMCP version's `fastmcp.Context`
class and adjust the call sites in Task 12 if names differ. The
sampling-supported test uses a `MagicMock` so it's agnostic to the exact
real API; only the real-MCP smoke (not in this plan) exercises the actual
signature.

No placeholders detected.
