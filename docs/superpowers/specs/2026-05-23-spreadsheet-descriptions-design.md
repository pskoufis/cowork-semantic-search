# Spreadsheet description-based indexing

**Status:** draft — pending implementation plan
**Branch:** `feat/spreadsheet-descriptions`
**Date:** 2026-05-23

## Problem

CSV files are currently indexed by serializing every row into one text blob
(`server/parsers.py:79`). XLSX is not supported at all. Both approaches scale
badly: large CSVs blow up embedding cost for low-value chunks ("row 47238 of
the transactions export" is rarely what anyone searches for), and naive row
text embeds poorly anyway.

We want spreadsheets to be retrievable by **what they're about**, not by
their raw contents — "find the workbook tracking customer churn" rather than
"find rows containing churn." The descriptions must be produced by the LLM
that uses the MCP (the host LLM in the user's chat session), not by a model
embedded inside the server.

## Goals

- Replace raw spreadsheet indexing with LLM-generated descriptions, embedded
  and stored as ordinary chunks.
- Per-sheet descriptions for XLSX (one per sheet), plus a file-level rollup.
  One file-level description for CSV.
- Two execution modes:
  - **Sampling (default):** server requests the host LLM's description via
    MCP `ctx.sample()` during the index job — single call, zero follow-up.
  - **Queue (fallback):** for MCP clients that don't support sampling, or
    when sampling fails, the server enqueues work and the host LLM drains
    via new MCP tools.
- Both modes produce identical on-disk chunks. `semantic_search` is
  unchanged.
- Existing already-indexed CSV chunks are auto-evicted and re-processed on
  the next `index_folder` run.

## Non-goals (v1)

- Filtering `semantic_search` by `chunk_kind` (chunk kind is stored but not
  exposed as a search parameter).
- Auto-rollup on the server side in queue mode — the LLM produces the
  file-level description from the sheet descriptions it just submitted.
- Augmenting descriptions with raw-row indexing as a second chunk kind —
  raw cells are never embedded for CSV or XLSX.
- Configurable retry policy for sampling — hard-coded to one retry with 2s
  backoff.

## Architecture

### Schema changes

Extend the `chunks` schema in `server/store.py:15` with two fields:

- `chunk_kind: string` — one of `"text"`, `"sheet_description"`,
  `"file_description"`. Default `"text"`.
- `sheet_name: string` (nullable) — set only on `sheet_description` chunks.

`VectorStore.ensure_schema` already migrates in place; extend it to add the
two columns with defaults via LanceDB `add_columns`. Existing chunks
backfill to `chunk_kind="text"`, `sheet_name=null`.

### Two new metadata tables

- `pending_descriptions(file_path, item_kind, sheet_name, preview_json,
  enqueued_at)` — one row per outstanding description (per sheet + one
  file-level row per file).
- `dismissed_files(file_path, content_hash, dismissed_at)` — files the LLM
  has explicitly declined to describe, keyed by the content hash that was
  current at dismissal time.

Both live in the same LanceDB instance as `chunks`. No vectors, pure
metadata.

### Indexing flow — XLSX

1. Open with `openpyxl` in `read_only=True` mode.
2. For each sheet, extract a **preview**: sheet name, row/col count,
   headers (row 1), inferred column dtypes from row 2 (with a couple of
   later-row samples for type robustness), and up to 10 sample rows from
   the top.
3. For each sheet, request a description via `ctx.sample()`. On success,
   embed the returned text and write one chunk with
   `chunk_kind="sheet_description"`, `sheet_name=<name>`.
4. After all sheets, one final `ctx.sample()` with the per-sheet
   descriptions as input → file-level description. Embed and write with
   `chunk_kind="file_description"`, `sheet_name=null`.
5. **Replace, not augment:** raw cell values are not indexed.

### Indexing flow — CSV

1. Read with `csv.Sniffer` to detect headers. Build the same preview shape
   (`headers` or `first_row_as_data: true`, dtypes, up to 10 sample rows,
   row count).
2. One `ctx.sample()` call → file-level description. Store with
   `chunk_kind="file_description"`.
3. **Replace, not augment:** raw rows are never indexed. (CSV is symmetric
   with XLSX in this v1.)

### Sampling failure routing

- Client advertises no `sampling` capability → at job start, all
  spreadsheet work routes directly to the queue. One log line, not per-file
  noise.
- Per-call sampling failure (timeout, refusal, network, empty response,
  response > 4KB) → retry once with 2s backoff. Still failing → the entire
  file routes to the queue; the job continues.
- **Atomic per-file commit:** sheet descriptions produced via sampling are
  buffered in memory, not written to the chunks table immediately. The
  buffer commits only after the file-level description also succeeds, at
  which point all N+1 chunks land in one write. If any step fails along
  the way, the buffer is discarded and the entire file (all sheets + the
  file-level rollup) is enqueued. A workbook is therefore always either
  fully sampled or fully queued — never half-and-half. This keeps queue
  previews self-contained (the LLM doing the file-level rollup has every
  sheet description in the same queue entry).

### Queue mode (fallback)

Three new MCP tools:

```
list_pending_descriptions(
    folder_path: str | None = None,
    limit: int = 20,
) -> {
    "pending": [
        {
            "file_path": str,
            "needs": ["sheet:Sales", "sheet:Costs", "file"],
            "preview": {
                "sheets": [
                    {"name": "Sales", "headers": [...], "dtypes": [...],
                     "row_count": int, "sample_rows": [[...], ...]},
                    ...
                ]
            }
        },
        ...
    ],
    "total_remaining": int
}
```

```
submit_description(
    file_path: str,
    sheet_name: str | None,            # null = file-level description
    description: str,
) -> {"status": "stored" | "rejected",
      "file_complete": bool, ...}
```

```
dismiss_pending_description(
    file_path: str,
) -> {"status": "dismissed", ...}
```

`submit_description` embeds the description, writes the chunk, removes the
matching row from `pending_descriptions`, and reports `file_complete=true`
once all sheets + the file-level rollup have been submitted. The host LLM
is responsible for producing the file-level description from the sheet
descriptions it just wrote.

`dismiss_pending_description` removes the queue entry and writes a row to
`dismissed_files` with the file's current `content_hash`. On the next
`index_folder` run: if the file's hash matches the dismissed hash, the
file is skipped silently; if the hash differs, the dismissal is dropped
and the file is re-enqueued.

`submit_description` is permitted while an `index_folder` job is running.
The existing "one indexer at a time" rule continues to apply to
`index_folder` only.

### `index_folder` — unchanged signature

No new parameters. Sampling capability is auto-detected; routing is
automatic. The result dict gains two new counters:

- `descriptions_sampled` — chunks written via the sampling path.
- `descriptions_queued` — entries written to `pending_descriptions`.

### One-shot eviction of existing CSV chunks

On the first `index_folder` run after upgrade, the indexer:

1. Queries `chunks` for any row with `file_type in {.csv, .xlsx}` and
   `chunk_kind = "text"`.
2. Groups them by `source_file` and deletes.
3. Forces re-processing of those files (bypasses the `content_hash`
   short-circuit exactly once).
4. After that pass, normal hash-based change detection resumes.

This flips already-indexed CSVs from raw-rows to descriptions without any
user intervention.

## Edge cases

- **0-row / header-only sheet:** still described (headers are informative).
- **0-col sheet:** skipped with `{reason: "empty_sheet"}` on the file
  result.
- **Encrypted / password-protected workbook:** fails per-file with
  `{reason: "encrypted"}`. Not enqueued — the LLM cannot describe what
  cannot be opened.
- **Huge sheet:** openpyxl read-only + bounded preview (10 rows) keep
  memory flat. `MAX_FILE_SIZE_BYTES` already gates files before open.
- **Queued file deleted from disk:** auto-evicted from the queue on the
  next `list_pending_descriptions` call. Not surfaced.
- **`submit_description` for a deleted or missing file:** rejected with a
  clear reason.
- **CSV with no detectable header:** preview carries
  `headers: null, first_row_as_data: true` so the LLM can decide what to
  do.

## Testing

### Unit — `tests/test_parsers.py` (extended)

- XLSX preview: programmatically built multi-sheet fixture. Assert sheet
  names, headers, dtypes, bounded sample size, row/col counts. Cover
  header-only and empty sheets.
- CSV preview: with-header and no-header fixtures; assert Sniffer-based
  detection produces the right preview shape.
- Encrypted XLSX: assert the `{reason: "encrypted"}` per-file result.

### Unit — `tests/test_store.py` (extended)

- Schema migration: build a DB on the old schema, call `ensure_schema`,
  assert columns added with correct defaults.
- One-shot eviction: seed an old-schema DB with CSV `text` chunks and a
  PDF `text` chunk; run the eviction pass; assert CSV chunks deleted, PDF
  chunks untouched.

### Integration — `tests/test_indexer_spreadsheets.py` (new)

- Sampling path: monkeypatch the `ctx.sample()` call site to return canned
  descriptions; index a fixture XLSX; assert N+1 chunks with correct
  `chunk_kind` / `sheet_name`.
- Queue path: simulate "no sampling capability"; index the same fixture;
  assert zero chunks, N+1 queue rows. Then call `list_pending_descriptions`,
  `submit_description` N+1 times, assert chunks land and queue empties.
- Mid-file sampling failure: canned sample that succeeds for sheet 1, fails
  for sheet 2 → assert *all* of the file's work routes to queue.
- Dismissal: enqueue → dismiss → reindex same file → assert it stays out.
  Modify file content → reindex → assert it's re-enqueued.
- Deleted-file auto-evict: enqueue a file, delete it from disk, call
  `list_pending_descriptions`, assert it's no longer surfaced.

### Integration — `tests/test_main.py` (extended)

Thin smoke tests for `list_pending_descriptions`, `submit_description`,
`dismiss_pending_description` through the FastMCP tool boundary:
argument validation, missing-file rejection, missing-DB rejection.

### Out of scope for v1 tests

- Real MCP sampling round-trip (covered by manual smoke).
- LanceDB performance regression gating.

## Dependencies

- `openpyxl` — add to `pyproject.toml` as a required dependency (XLSX
  parsing). Pinned to a current 3.x.
- No new dependency for sampling — `fastmcp` already exposes
  `ctx.sample()`.

## Open questions

None — all design decisions resolved during brainstorming. Ready for
implementation plan.
