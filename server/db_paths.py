"""Resolve which LanceDB index directories an operation targets.

Reads are fan-out: `resolve_db_dirs` returns every configured index.
Writes are single-target: `resolve_write_dir` collapses to one.

Precedence (reads): explicit arg → LANCEDB_PATHS (os.pathsep list) →
LANCEDB_PATH (singular) → ./lancedb.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _from_paths_env() -> list[str]:
    raw = os.environ.get("LANCEDB_PATHS")
    if not raw:
        return []
    seen: dict[str, None] = {}
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        seen.setdefault(os.path.abspath(part), None)
    return list(seen)


def resolve_db_dirs(explicit: str | None) -> list[str]:
    """The list of index dirs a read (search/status) should span."""
    if explicit is not None:
        return [os.path.abspath(explicit)]
    paths = _from_paths_env()
    if paths:
        return paths
    singular = os.environ.get("LANCEDB_PATH")
    if singular:
        return [os.path.abspath(singular)]
    return [os.path.abspath("./lancedb")]


def resolve_write_dir(explicit: str | None) -> str:
    """The single index dir a write (index/reindex) should target."""
    if explicit is not None:
        return os.path.abspath(explicit)
    singular = os.environ.get("LANCEDB_PATH")
    if singular:
        return os.path.abspath(singular)
    paths = _from_paths_env()
    if paths:
        if len(paths) > 1:
            logger.info("indexing into %s (first of LANCEDB_PATHS)", paths[0])
        return paths[0]
    return os.path.abspath("./lancedb")
