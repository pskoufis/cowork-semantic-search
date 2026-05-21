"""Path portability: convert between absolute paths and paths relative to the
LanceDB directory (the portable anchor).

The LanceDB directory and the indexed corpus must live on the same volume.
Relative paths are resolved against the *current* LanceDB location at runtime,
so the index survives the drive being re-mounted or moved between Macs.
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
    """Convert an absolute path to one relative to the LanceDB directory.

    Corpus and index are typically siblings on a drive, so the result usually
    contains '../' segments (os.path.relpath can express that; Path.relative_to
    cannot). Raises ValueError if abs_path is on a different volume than the
    index (portability only works within one volume); the check is skipped when
    check_volume is False or either path is missing on disk.
    """
    db_dir = _db_dir(db_path)
    abs_norm = os.path.abspath(abs_path)
    if check_volume and os.path.exists(abs_norm) and os.path.exists(db_dir):
        if os.stat(abs_norm).st_dev != os.stat(db_dir).st_dev:
            raise ValueError(
                f"{abs_norm} is on a different volume than the index "
                f"directory {db_dir}; the index and corpus must share one "
                f"volume to be portable."
            )
    return os.path.normpath(os.path.relpath(abs_norm, start=db_dir))


def to_absolute(rel_path: str, db_path: str) -> str:
    """Resolve a stored relative path back to absolute against the current
    LanceDB directory location."""
    return os.path.normpath(os.path.join(_db_dir(db_path), rel_path))
