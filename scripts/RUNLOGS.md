# Run-logs: analyzing and re-running the batch scripts

The unpack / convert / copy scripts under `scripts/` write a **structured JSONL
run-log** in addition to their human-readable stderr output. One artifact serves
three purposes:

1. **Analyze a run afterwards** — one JSON object per line, the same schema
   across every script, so `jq` can answer "which inputs failed, with what error
   class, and where are they?" without grepping stderr.
2. **Re-execute only what's left** — the log doubles as an idempotency ledger:
   re-running the same command skips inputs already done and reprocesses the
   ones that failed or are missing.
3. **Surface failures** — each failed item carries the exception type and
   message.

The shared implementation is `scripts/_runlog.py`.

## Where the log lives

```
<output>/_runlogs/<script>-<run_id>.jsonl
```

- One file per run (`run_id` = UTC microsecond timestamp + short random suffix),
  so history accumulates and no prior file is ever rewritten.
- `<output>` is the script's output folder. For `extract_zips.py` /
  `extract_zips_flat.py`, which extract into the input tree, `<output>` is the
  scanned root.
- Override the location with `--log-dir DIR`; disable entirely with
  `--no-runlog`.

## Schema (`schema: 1`)

```jsonc
// first line
{"event":"run_start","schema":1,"ts":"2026-05-30T12:00:00Z","run_id":"…",
 "script":"unpack_mbox_folder","argv":["in/","out/"],
 "input_root":"/abs/in","output_root":"/abs/out"}

// one per discovered input
{"event":"item","ts":"…","run_id":"…","script":"…",
 "input":"sub/a.mbox","input_abs":"/abs/in/sub/a.mbox",
 "input_mtime":1748600000.0,"input_size":12345,
 "kind":"mbox","output":"/abs/out/a","status":"ok|skip|fail",
 "error_type":null,"error":null,"duration_s":1.83}

// last line
{"event":"run_end","ts":"…","run_id":"…","script":"…",
 "totals":{"ok":40,"skip":7,"fail":2},"elapsed_s":91.4,"exit_code":1}
```

- `status`: `ok` (processed), `skip` (already done — idempotency hit), `fail`.
- `kind`: archive/file type, or the bucket name for `split_for_indexing.py`.

## Analysis recipes (`jq`)

List every failure with its cause:

```bash
jq -c 'select(.event=="item" and .status=="fail")
       | {input, error_type, error}' \
  out/_runlogs/extract_archives_folder-*.jsonl
```

Count outcomes for the latest run:

```bash
jq -c 'select(.event=="run_end") | .totals' \
  "$(ls -t out/_runlogs/extract_archives_folder-*.jsonl | head -1)"
```

Group failures by error class:

```bash
jq -r 'select(.event=="item" and .status=="fail") | .error_type' \
  out/_runlogs/*.jsonl | sort | uniq -c | sort -rn
```

Bucket distribution from `split_for_indexing.py`:

```bash
jq -r 'select(.event=="item") | .kind' \
  out/_runlogs/split_for_indexing-*.jsonl | sort | uniq -c
```

Trace one input's full history across all runs:

```bash
jq -c 'select(.input=="sub/a.mbox") | {run_id, status, error}' \
  out/_runlogs/unpack_mbox_folder-*.jsonl
```

## Re-running after fixing an issue

The workflow is: run → inspect failures in the run-log → fix the cause (install a
codec, repair a file, free disk) → **run the same command again**. Inputs already
done are skipped; only the previously-failed or now-changed inputs are
reprocessed.

An input counts as *done* when its last record was `ok`, its `(mtime, size)` is
unchanged, and its recorded output still exists. Editing an input invalidates the
skip (it gets reprocessed); deleting its output does too.

Flags:

- `--force` — reprocess everything, ignoring the ledger (still overwrites
  outputs in place rather than duplicating). Not offered by the marker-based
  scripts (`extract_zips*`, which use `--overwrite`) or `split_for_indexing.py`.
- `--log-dir DIR`, `--no-runlog` — as above.

### Idempotency per script

| Script | Re-run behaviour |
|---|---|
| `flatten_copy.py` | Ledger skip — re-run no longer duplicates as `_1/_2`. |
| `images_to_pdf.py` | Ledger skip; re-run allowed into its own prior output. |
| `unpack_mbox_folder.py` | Ledger skip + stable target reuse (no `-N` dupes). |
| `unpack_msg_folder.py`, `unpack_pst_folder.py` | Ledger skip (paths were already deterministic). |
| `extract_archives_folder.py` | Ledger skip across the recursion, including nested archives. |
| `extract_zips.py`, `extract_zips_flat.py` | Marker-file idempotency (pre-existing); run-log records the outcome. |
| `split_for_indexing.py` | One-shot (output must be empty); run-log records classification only. |

## Known limitations (by design / out of scope)

- **Sub-item failures aren't itemized.** When a single message inside an
  `.mbox` / `.pst` / `.msg` fails to parse, the handler libraries log a warning
  to stderr but the script records the *whole file* as `ok`/`fail`. Surfacing
  per-message failures would require changing the handler libraries and was kept
  out of scope.
- **SIGINT handling is uneven.** `images_to_pdf.py` and `split_for_indexing.py`
  trap Ctrl-C and exit `130`; the other scripts do not install a handler.
- **A path-traversal-unsafe zip/rar member aborts the whole archive** (a
  deliberate security stance) rather than skipping just that member.
- **`--copy-others` files** copied by `extract_archives_folder.py` are not
  individually recorded in the run-log (they already have mtime-based
  idempotency and per-file stderr errors).

## Indexing note

`_runlogs/` lives inside the output tree (consistent with
`extract_zips_flat.py`'s audit log). If you index that tree, exclude
`_runlogs/` so the JSONL logs aren't treated as content.
