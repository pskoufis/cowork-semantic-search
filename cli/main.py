"""Entry point for the ``csemsearch`` CLI.

This module exposes ``main(argv=None) -> int`` so:

* the console-script entry in ``pyproject.toml`` can call it directly, and
* tests can drive subcommands by passing argv lists rather than spawning
  subprocesses.

``main`` does no real work itself — it parses argv, configures logging,
resolves the index path, and dispatches to a function in
``cli.commands``. Each subcommand function returns an integer exit code.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser plus subparsers.

    Subparsers are added even when their command implementation is in a
    later wave so ``csemsearch --help`` shows the full surface from day
    one. Each subcommand's actual handler is imported lazily in
    ``_dispatch`` to keep cold-start cheap.
    """
    parser = argparse.ArgumentParser(
        prog="csemsearch",
        description=(
            "Command-line driver for the cowork-semantic-search indexer. "
            "Wraps the same library code path the MCP server uses, with "
            "richer terminal output."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=(
            "Path to the LanceDB directory. Defaults to the LANCEDB_PATH "
            "env var, then ./lancedb."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DEBUG-level log messages.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress INFO log messages; show WARNING and above.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p_status = sub.add_parser(
        "status",
        help="Show index contents and recent job history.",
        description=(
            "Print the size on disk, total chunks, total files, and the last "
            "ten background jobs known to the on-disk job registry."
        ),
    )
    p_status.set_defaults(handler=_dispatch_status)

    p_unpack = sub.add_parser(
        "unpack",
        help="Unpack .pst/.mbox/.msg archives in a folder.",
        description=(
            "Run only the pst → mbox → msg preprocessing passes. Each archive "
            "gets a sibling <stem>_unpacked/ tree of .txt files + attachments. "
            "Idempotent — re-runs skip up-to-date archives by mtime."
        ),
    )
    p_unpack.add_argument("folder", help="Folder to scan for archives.")
    p_unpack.add_argument(
        "--no-recursive", action="store_true",
        help="Only unpack archives at the top of the folder.",
    )
    p_unpack.add_argument(
        "--exclude", action="append", default=[], metavar="PAT",
        help="gitignore-style pattern. Repeatable.",
    )
    p_unpack.set_defaults(handler=_dispatch_unpack)

    p_index = sub.add_parser(
        "index",
        help="Index documents in a folder.",
        description=(
            "Discover supported files, parse, chunk, embed, and store. "
            "Re-runs are incremental — unchanged files are skipped via the "
            "hash cache. Ctrl-C cancels gracefully at the next file boundary."
        ),
    )
    p_index.add_argument("folder", help="Folder to index.")
    p_index.add_argument(
        "--no-recursive", action="store_true",
        help="Only index files at the top of the folder.",
    )
    p_index.add_argument(
        "--no-unpack", action="store_true",
        help="Skip the pst/mbox/msg preprocessing passes.",
    )
    p_index.add_argument(
        "--exclude", action="append", default=[], metavar="PAT",
        help="gitignore-style pattern. Repeatable.",
    )
    p_index.add_argument(
        "--types", action="append", default=[], metavar="EXT",
        help="Limit to these file extensions (e.g. .pdf .md). Repeatable.",
    )
    p_index.add_argument(
        "--safe-flush", action="store_true",
        help=(
            "Persist each file's chunks immediately (flush_threshold=1). "
            "Slower on many-small-files corpora; bounds work-loss on a crash."
        ),
    )
    p_index.set_defaults(handler=_dispatch_index)

    p_run = sub.add_parser(
        "run",
        help="Unpack and then index in one go.",
        description=(
            "Sugar for `unpack` followed by `index --no-unpack`. Mirrors the "
            "MCP `index_folder` tool's one-shot behavior."
        ),
    )
    p_run.add_argument("folder", help="Folder to unpack and index.")
    p_run.add_argument(
        "--no-recursive", action="store_true",
        help="Only act on files at the top of the folder.",
    )
    p_run.add_argument(
        "--exclude", action="append", default=[], metavar="PAT",
        help="gitignore-style pattern. Repeatable.",
    )
    p_run.add_argument(
        "--types", action="append", default=[], metavar="EXT",
        help="Limit indexing to these extensions. Repeatable.",
    )
    p_run.add_argument(
        "--safe-flush", action="store_true",
        help="Per-file flush during indexing (see `index --safe-flush`).",
    )
    p_run.set_defaults(handler=_dispatch_run)

    p_search = sub.add_parser(
        "search",
        help="Search the index.",
        description=(
            "Embed the query, retrieve top matches, and print results. "
            "Vector mode is pure semantic; hybrid combines vector with BM25 "
            "full-text search via reciprocal rank fusion."
        ),
    )
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument(
        "--mode", choices=("vector", "hybrid"), default="vector",
        help="Search mode (default: vector).",
    )
    p_search.add_argument(
        "-n", "--top-k", type=int, default=10,
        help="Number of results (default: 10).",
    )
    p_search.add_argument(
        "--folder", default=None,
        help="Limit results to files under this folder.",
    )
    p_search.set_defaults(handler=_dispatch_search)

    p_reindex = sub.add_parser(
        "reindex",
        help="Force re-index a single file.",
        description=(
            "Bypass the hash cache and rebuild chunks for one file. Rejected "
            "if a background indexing job is currently running."
        ),
    )
    p_reindex.add_argument("file_path", help="Absolute or relative path to the file.")
    p_reindex.set_defaults(handler=_dispatch_reindex)

    return parser


# ---------------------------------------------------------------------------
# Dispatchers (kept thin — they import their handler lazily, then call it)
# ---------------------------------------------------------------------------


def _dispatch_status(args: argparse.Namespace, db_path: str) -> int:
    from cli.commands import status_cmd

    return status_cmd(db_path)


def _dispatch_unpack(args: argparse.Namespace, db_path: str) -> int:
    from cli.commands import unpack_cmd

    return unpack_cmd(
        args.folder,
        recursive=not args.no_recursive,
        exclude=args.exclude or None,
        db_path=db_path,
    )


def _dispatch_index(args: argparse.Namespace, db_path: str) -> int:
    from cli.commands import index_cmd

    return index_cmd(
        args.folder,
        db_path=db_path,
        recursive=not args.no_recursive,
        exclude=args.exclude or None,
        unpack_first=not args.no_unpack,
        safe_flush=args.safe_flush,
        file_types=args.types or None,
    )


def _dispatch_run(args: argparse.Namespace, db_path: str) -> int:
    from cli.commands import run_cmd

    return run_cmd(
        args.folder,
        db_path=db_path,
        recursive=not args.no_recursive,
        exclude=args.exclude or None,
        safe_flush=args.safe_flush,
        file_types=args.types or None,
    )


def _dispatch_search(args: argparse.Namespace, db_path: str) -> int:
    from cli.commands import search_cmd

    return search_cmd(
        args.query,
        db_path=db_path,
        mode=args.mode,
        top_k=args.top_k,
        folder=args.folder,
    )


def _dispatch_reindex(args: argparse.Namespace, db_path: str) -> int:
    from cli.commands import reindex_cmd

    return reindex_cmd(args.file_path, db_path=db_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_db_path(explicit: str | None) -> str:
    if explicit is not None:
        return os.path.abspath(explicit)
    return os.path.abspath(os.environ.get("LANCEDB_PATH", "./lancedb"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose and args.quiet:
        parser.error("--verbose and --quiet are mutually exclusive")

    from cli.log import configure as configure_logging

    configure_logging(verbose=args.verbose, quiet=args.quiet)

    db_path = _resolve_db_path(args.db_path)

    handler = getattr(args, "handler", None)
    if handler is None:
        # argparse with required=True should never let us reach here, but be
        # defensive in case a subcommand gets added without set_defaults.
        parser.error(f"unknown command: {args.command}")
    try:
        return handler(args, db_path)
    except KeyboardInterrupt:
        logging.getLogger("cowork_semantic_search").info(
            "interrupted by user (SIGINT); exiting"
        )
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
