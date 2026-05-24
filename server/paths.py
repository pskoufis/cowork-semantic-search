"""Path portability: convert between absolute paths and stored paths.

When the corpus and LanceDB index live on the same volume, paths are stored
relative to the index directory — the index is then portable across drive
remounts and across Macs. When they live on different volumes, paths are
stored absolute — the index works but is no longer portable across remounts
of either volume. The form is chosen automatically per call based on st_dev.

macOS-only: os.path.relpath already yields '/' separators.
"""

import os


def _db_dir(db_path: str) -> str:
    """db_path normalized to an absolute directory string."""
    return os.path.abspath(db_path)


def default_db_dir() -> str:
    """The LanceDB directory the server uses when no db_path is given:
    the LANCEDB_PATH env var, or ./lancedb, normalized to an absolute path."""
    return os.path.abspath(os.environ.get("LANCEDB_PATH", "./lancedb"))


def to_relative(abs_path: str, db_path: str, check_volume: bool = True) -> str:
    """Convert an absolute path to its canonical storage form.

    Returns a path relative to the LanceDB directory when both live on the
    same volume (the portable case — result typically contains '../' segments
    since corpus and index are siblings on a drive). When they live on
    different volumes, returns the absolute path unchanged — the index then
    works but is no longer portable across remounts. The volume detection is
    skipped when check_volume is False or either path is missing on disk, in
    which case the relative form is computed unconditionally.
    """
    db_dir = _db_dir(db_path)
    abs_norm = os.path.abspath(abs_path)
    if check_volume and os.path.exists(abs_norm) and os.path.exists(db_dir):
        if os.stat(abs_norm).st_dev != os.stat(db_dir).st_dev:
            return abs_norm
    return os.path.normpath(os.path.relpath(abs_norm, start=db_dir))


def to_absolute(rel_path: str, db_path: str) -> str:
    """Resolve a stored path back to absolute. Accepts either the relative
    form (same-volume entries, joined against the current index location) or
    an already-absolute form (cross-volume entries, returned unchanged)."""
    if os.path.isabs(rel_path):
        return os.path.normpath(rel_path)
    return os.path.normpath(os.path.join(_db_dir(db_path), rel_path))
