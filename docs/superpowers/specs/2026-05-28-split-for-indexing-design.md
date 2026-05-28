# Split-for-indexing utility — design

A new prep script that takes a flat dump of files and sorts them into
buckets suited to the downstream indexer. Files within size limits are
copied into 10K-file zips; out-of-spec files, images, and unknown
extensions land in dedicated sibling folders.

## Goals

- Single command that takes a flat input folder and produces a sorted
  output tree without modifying the input.
- Streaming behaviour — no upfront directory materialization. Files
  are classified and dispatched as they are discovered.
- Live progress feedback while running.
- Deterministic categorization rules so the same input always produces
  the same layout.

## Non-goals

- Recursive walking. Input is assumed flat; subdirectories are ignored.
- Filename de-duplication. The flat input is assumed to have unique
  basenames already.
- Actually converting images to PDF. The `TO_PDF/` bucket is just a
  staging area for a separate later step.
- Resume / idempotency. A re-run requires a fresh, empty output dir.
- Manifest files. The bucket layout itself is the record of truth.

## CLI

```
python scripts/split_for_indexing.py INPUT_DIR OUTPUT_DIR [--batch-size 10000]
```

- `INPUT_DIR` — flat folder of files to sort. Must exist and be a directory.
- `OUTPUT_DIR` — destination. Must not exist, or must exist and be empty.
- `--batch-size` — files per zip (default 10000).

Exit codes:
- `0` — clean run.
- `2` — argument / pre-flight failure (bad paths, non-empty output).
- `130` — interrupted by SIGINT (Ctrl-C).

## Output layout

```
OUTPUT_DIR/
  BATCHES/
    batch_001.zip
    batch_002.zip
    ...
  TOO_LARGE/
    <loose files>
  TO_PDF/
    <loose files>
  OTHER_EXTENSIONS/
    <loose files>
```

All four subdirs are created up-front. Empty ones are left in place
after the run; downstream tooling can rely on their existence.

## Categorization

Extension matching is case-insensitive. The trailing dot's case
("`.PDF`" vs "`.pdf`") does not affect classification.

| Group | Extensions | Size rule | Destination |
| --- | --- | --- | --- |
| Text-like | `.txt`, `.htm`, `.html`, `.rtf` | size ≤ 20 × 1024² bytes | `BATCHES/` |
| Text-like (oversize) | same | size > 20 × 1024² bytes | `TOO_LARGE/` |
| Document-like | `.pdf`, `.doc`, `.docx`, `.eml`, `.msg`, `.wpd` | size ≤ 500 × 1024² bytes | `BATCHES/` |
| Document-like (oversize) | same | size > 500 × 1024² bytes | `TOO_LARGE/` |
| Image | `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.heic`, `.heif`, `.webp`, `.jp2` | no size cap | `TO_PDF/` |
| Everything else (incl. no extension) | * | n/a | `OTHER_EXTENSIONS/` |

Notes:

- Size limits are binary (`MiB`): 20 × 1024 × 1024 and 500 × 1024 × 1024.
- The bound is inclusive (`<=`).
- Dotfiles (basename starting with `.`, e.g. `.DS_Store`) are skipped
  entirely. They are not copied and are not counted in any bucket.
- Subdirectory entries and symbolic links to directories in the input
  are silently skipped.

## Streaming flow

1. Validate `INPUT_DIR` exists and is a directory.
2. Validate `OUTPUT_DIR` either doesn't exist or is empty; create the
   four subdirs.
3. Iterate `os.scandir(INPUT_DIR)` so directory entries are streamed,
   not materialized.
4. For each entry:
   - Skip if not a regular file, or if its basename starts with a dot.
   - Classify via lowercase extension + `entry.stat().st_size`.
   - If the destination is `BATCHES/`: open `batch_001.zip` lazily on
     the first such file; `zf.write(entry.path, arcname=entry.name)`;
     increment a counter; when counter == `--batch-size`, close the
     zip and roll to `batch_002.zip`.
   - Otherwise: `shutil.copy2(entry.path, dest_subdir / entry.name)`.
5. After the loop, close any open zip writer.
6. Print a final summary with per-bucket file counts and total bytes
   copied per bucket.

`ZIP_DEFLATED` compression is used so the zips are useful as transport
artefacts. Each zip member's `arcname` is the file's basename (no
directory prefix inside the zip).

## Progress

Uses `tqdm` if available (already an optional `cli` dependency in
`pyproject.toml`). The bar runs in unknown-total mode (`total=None`,
`unit="file"`) and updates a postfix dict each iteration:

```
{batches: N, too_large: N, to_pdf: N, other: N, cur_zip: batch_003}
```

If `tqdm` is not importable, the script falls back to a plain
"`processed N files`" line printed every 500 files so the run still
shows life on minimal installs.

## Error handling

- `PermissionError` / `FileNotFoundError` (file disappeared mid-walk)
  on a single entry: log a warning to stderr, skip, continue.
- `OSError` writing the destination (disk full, etc.): let it
  propagate. The output zip is likely corrupt; the user should fix
  the underlying problem and re-run into a fresh `OUTPUT_DIR`.
- `KeyboardInterrupt` at the top level: close the current zip writer
  cleanly, print a partial summary, exit `130`.

## Testing

`tests/test_split_for_indexing.py` covers:

- Each category boundary: text-like at and above 20 MiB; document-like
  at and above 500 MiB; an image; an unknown extension; a file with no
  extension at all.
- Mixed-case extension (`.PDF`, `.JPG`) classifies the same as lower.
- Dotfile (`.DS_Store`) is silently skipped.
- Subdirectory in the input is silently skipped.
- Batch rolling at the configured size: with `--batch-size 3` and 7
  in-spec files we get `batch_001.zip`, `batch_002.zip`,
  `batch_003.zip` with 3/3/1 members respectively.
- Pre-existing non-empty `OUTPUT_DIR` is rejected with exit code 2.
- Pre-existing empty `OUTPUT_DIR` is accepted.
- Zip contents: `arcname` matches the input basename (no path prefix).

Size-bound tests use a small `--batch-size` and small synthetic
thresholds where possible (the script's limit constants are module-level
so a test can monkeypatch them down to a few KiB rather than create a
500 MiB fixture on disk).
