# Unified archive extractor — design

**Date:** 2026-05-28
**Script:** `scripts/extract_archives_folder.py`

## Purpose

One unified orchestrator that scans an input folder (recursively) for `.mbox`,
`.pst`, `.msg`, and `.zip`, extracts/unpacks each into an output folder that
**mirrors the input's structure**, and **recurses fully** — archives revealed
inside an extracted zip are processed too, until nothing new appears.

## CLI

```
python scripts/extract_archives_folder.py <input-folder> <output-folder> \
    [--dry-run] [--max-depth N] [--copy-others]
```

- `input_folder`, `output_folder`: positional, like the sibling batch scripts.
- `--dry-run`: print planned work, write nothing.
- `--max-depth N`: recursion backstop (default 10).
- `--copy-others`: also copy every **non-archive** file from the input tree
  into the output, mirroring its layout — making the output a complete
  index-ready mirror (extracted archives + all other docs). A single pass over
  the *input* tree only (files revealed inside extracted zips already live in
  the output). A destination is overwritten only when the source is newer
  (mtime), so re-runs are idempotent. Archive originals are never copied — they
  are extracted, not duplicated.

## Reuse, not reinvention

Imports `ensure_unpacked` from `mbox_handling`, `msg_handling`, `pst_handling`;
uses stdlib `zipfile` for zips and the optional `rarfile` package for rars. No
duplicated extraction logic.

## Per-type output (mirrors input layout)

| Type    | Target                                                              |
|---------|--------------------------------------------------------------------|
| `.msg`  | `<out>/<rel_parent>/<stem>.txt` + `attachments/<stem>__*`          |
| `.mbox` | `<out>/<rel_parent>/<stem>_unpacked/` (thread tree)                |
| `.pst`  | `<out>/<rel_parent>/<stem>_unpacked/` (folder tree)                |
| `.zip`  | `<out>/<rel_parent>/<stem>/` (internal structure preserved)        |
| `.rar`  | `<out>/<rel_parent>/<stem>/` (internal structure preserved)        |

Deliberate deviation: the standalone `unpack_mbox_folder.py` *flattens* mbox to
`<out>/<stem>/`; this unified script *mirrors* (`<out>/<rel>/<stem>_unpacked/`).

## Recursion loop (cycle-safe by construction)

- Worklist with `processed: set[Path]` of resolved archive paths.
- Pass 1 scans the **input** tree. Each later pass scans the **output** tree for
  the four types, processing only paths not in `processed`, marking each as it
  goes.
- Terminate when a pass yields zero new archives. The `processed` set is what
  makes **zips** terminate (zip extraction has no mtime idempotency, unlike the
  email types).
- Hard `--max-depth` cap as a backstop against a self-similar/quine archive.

## Relative-path base rule (one rule, applied consistently)

- Top-level files mirror from the **input root** → `<out>/<rel>/...`.
- Anything discovered **inside the output tree** (revealed by a zip) is processed
  **in place within the output tree** — its own parent dir is its target.
- Worked example: `<input>/a/foo.zip` containing `bar.msg` → `<out>/a/foo/bar.txt`.

## Invariants & guards

- **Input tree is never modified** — all writes go to output.
- Reject `output inside input` — with output-tree scanning that is a real
  feedback loop, not just untidy.
- Path-traversal guard on zip **and rar** entries (reject entries escaping the
  target dir).
- **macOS metadata skip:** AppleDouble `._*` sidecars (created on exFAT/FAT
  volumes), `.DS_Store`, and anything under a `__MACOSX/` folder are skipped in
  both archive discovery and `--copy-others`. Without this, a `._foo.zip`
  sidecar is mistaken for a real zip and fails with "File is not a zip file".
- **Multi-volume rar:** only the first volume of a set is processed. Discovery
  skips `foo.partN.rar` where N > 1 (the first volume pulls in the whole set;
  opening a later volume directly would error and inflate the failure count).
- Known asymmetry (documented, not "fixed"): top-level email *sources* stay in
  input (only derived `.txt` lands in output); zip-nested email sources get
  copied into output by extraction, so they sit beside their `.txt`. Fine for
  indexing — no deletion logic added.

## Errors & exit

Per-file isolation (log to stderr, continue); final summary tally per type;
exit non-zero if any file failed. Matches existing batch-script convention.

## Tests

`tests/test_extract_archives_folder.py`, focused on the invented part:

1. A zip-containing-a-`.msg` fully unpacks to the right mirrored path.
2. A nested zip (zip-in-zip) extracts recursively, structure preserved.
3. An already-processed zip in the output tree is not re-extracted (loop
   terminates; each distinct zip extracted exactly once).
4. Output-inside-input is rejected.
5. Max-depth backstop caps recursion.
6. A real `.mbox` in a subdir unpacks to the mirrored `<stem>_unpacked/`.
7. `_target_for` convention for all four kinds, both passes (the cheap
   discriminator for the `.pst` branch — see caveat below).
8. A `.msg` that fails to parse is counted as a failure, not a silent
   success (mirrors `unpack_msg_folder.py`'s guard).
9. `--copy-others` mirrors non-archive files into the output while archives
   are still extracted; without the flag they are left behind; archive
   originals are not copied.
10. A re-run with `--copy-others` does not re-copy an up-to-date destination.
11. `.rar` is a recognised kind and `_extract_rar` raises a friendly
    `pip install ...[rar]` hint when `rarfile` is absent.
12. Discovery skips macOS metadata (`._*`, `.DS_Store`, `__MACOSX/`).
13. Discovery keeps `a.part1.rar`/`b.part01.rar` but skips secondary volumes.
14. `--copy-others` does not copy macOS metadata junk.

Per-type unpacking internals are already covered by existing tests.

**pst caveat:** the `.pst` extraction path is structurally identical to
`.mbox` (same `ensure_unpacked(target=...)` call, same `<stem>_unpacked`
target convention, verified by test 7), but it is **not exercised
end-to-end here** because `pypff`/`libratom` are not installed in this
environment.

**rar caveat:** `.rar` needs the optional `rarfile` package plus a system
backend (`unar`/`unrar`; the system `bsdtar` also works for many archives).
`_extract_rar` is a verbatim port of the end-to-end-tested `_extract_zip`
(same path-traversal guard), but a real rar round-trip is **not exercised
here** — creating a `.rar` requires the proprietary `rar` compressor, which
is not installed (same unit-only strategy as `.pst`). Install for real use
with `pip install 'cowork-semantic-search[rar]'` + `brew install unar`.
