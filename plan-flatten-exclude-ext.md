# Plan: `--exclude-ext` for `flatten_copy.py`

## Goal
Let `flatten_copy.py` skip files by extension (e.g. `.ics`) so they're never
copied into the flat target. Generic, not `.ics`-specific.

## CLI
```
uv run python scripts/flatten_copy.py SRC DST --exclude-ext .ics
# repeatable and comma-separated, both supported:
  --exclude-ext .ics --exclude-ext .tmp
  --exclude-ext .ics,.tmp
```

## Behavior
- New arg: `--exclude-ext` with `action="append"` (default `None`).
- Normalize each value: split on commas, strip whitespace, lowercase, ensure a
  leading `.`, drop empties. Collect into a `set[str]`.
- Matching is case-insensitive on the file's last suffix
  (`Path(name).suffix.lower()`).
- Excluded files are **filtered out before the copy loop** (in `_iter_files`),
  so:
  - the `Found N regular file(s)` count reflects only files that will be copied,
  - the run-log never records them (they aren't "skipped", they're out of scope).
- No exclusions given → behaviour is identical to today.

## Changes
1. `_parse_args` — add the `--exclude-ext` argument.
2. New helper `_normalize_exts(values) -> set[str]` (comma-split + dot-normalize).
3. `_iter_files(root, exclude_exts)` — skip files whose suffix is in the set.
4. `main` — build the set, pass it to `_iter_files`, and print a one-line note to
   stderr when exclusions are active (e.g. `Excluding extensions: .ics`).
5. Update the module docstring usage block.

## Test
Add `tests/test_flatten_copy_exclude_ext.py` (or extend an existing flatten
test): a source tree with `a.txt`, `b.ics`, `sub/c.ICS` → run with
`--exclude-ext .ics` → target contains only `a.txt`; assert `.ics`/`.ICS`
absent and case-insensitive matching works. Also assert comma-separated form
`--exclude-ext .ics,.tmp` excludes both.

## Out of scope
Glob patterns, include-only filters, size filters.
