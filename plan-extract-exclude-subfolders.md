# Plan: `--exclude` flag for extract_archives_folder.py

## Goal
Add a repeatable `--exclude <DIR>` flag to `scripts/extract_archives_folder.py`
so a run can extract every `.zip/.rar/.pst/.mbox/.msg` under the input folder
**except** files living inside one or more named subfolders. Covers the user's
"extract everything except 2 subfolders" requirement; the type-handling and
"zips/rars first, then mail" ordering are already provided by the existing tool.

## CLI
```
python scripts/extract_archives_folder.py <input> <output> \
    --exclude SubA --exclude path/to/SubB [--dry-run] [--copy-others]
```
- `--exclude` is repeatable (`action="append"`, default `[]`).
- Each value is a path **relative to `<input>`** (e.g. `Archive/Old`). A bare
  name (`SubA`) matches a top-level subfolder. Leading/trailing slashes tolerated.
- Matching is by path prefix: a file is excluded if its path, relative to the
  input root, is inside any excluded directory.

## Behavior / implementation
1. Parse `--exclude` into a set of normalized relative `Path`s (resolved against
   `src`, then made relative to `src`; warn + ignore any that don't exist or
   escape the input tree).
2. Add one helper:
   ```python
   def _is_excluded(path: Path, root: Path, excludes: set[Path]) -> bool:
       """True if path (under root) lives inside any excluded relative dir."""
   ```
   Compare `path.relative_to(root)` against each excluded dir via `is_relative_to`.
3. Apply the filter at the three input-tree walks (all gated on the excludes set
   being non-empty, so behavior is unchanged when the flag is absent):
   - `_discover` (line ~417) — the main archive scan. **Most important.**
   - `_copy_other_files` (line ~384) — so `--copy-others` also honors it.
   - `_warn_orphan_secondary_rars` (line ~240) — don't warn about rar sets in
     excluded folders.
   These three take `src` as `root`. Pass 1 only scans `src`, so exclusion is
   applied where it matters.
4. Pass 2+ scans the **output** tree (`dst`). Excluded input subfolders never get
   extracted into `dst`, so nothing from them can reappear there — no extra
   filtering needed in later passes. (A short comment will note this.)
5. Threading: `_discover` is also called by `_remaining()`. Either thread
   `excludes` through both, or capture them where called. Plan: add an
   `excludes` parameter (default empty set) to `_discover`, `_copy_other_files`,
   `_warn_orphan_secondary_rars`, and `_remaining`, threaded from `_run`.

## Logging
- At startup, when excludes are set, print one stderr line:
  `Excluding subfolders: SubA, path/to/SubB`.
- Any `--exclude` value that doesn't resolve to an existing dir under input:
  `warning: --exclude <val> not found under input; ignored`.

## Out of scope
- Glob/wildcard excludes (only literal directory paths).
- Excluding by file extension (that's `flatten_copy.py --exclude-ext`).
- Excluding inside revealed-in-zip output (pass 2+) — not needed, see step 4.

## Testing
- `--dry-run` on a fixture tree with an excluded subfolder containing a `.zip`
  and a `.pst`: confirm neither appears in the planned output, and files outside
  the excluded folders still do.
- Re-run without `--exclude`: output identical to current behavior (regression).

## Files touched
- `scripts/extract_archives_folder.py` only. Stdlib only; no new deps.
