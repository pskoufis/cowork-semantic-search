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

    return parser


# ---------------------------------------------------------------------------
# Dispatchers (kept thin — they import their handler lazily, then call it)
# ---------------------------------------------------------------------------


def _dispatch_status(args: argparse.Namespace, db_path: str) -> int:
    from cli.commands import status_cmd

    return status_cmd(db_path)


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
