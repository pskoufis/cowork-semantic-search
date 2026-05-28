"""Logging setup for the ``csemsearch`` CLI.

The library (``server.*``) attaches no handlers to its loggers — that keeps
the MCP server's stdio JSON-RPC channel clean. The CLI installs a single
stderr ``StreamHandler`` on the ``cowork_semantic_search`` logger so phase
events, file counters and end-of-run summaries land in the user's terminal.

``configure`` is idempotent: calling it twice (e.g. in tests that import
multiple CLI subcommand modules) does not stack handlers.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "cowork_semantic_search"


def configure(*, verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Install a stderr handler on the library logger and set the level.

    ``verbose=True`` and ``quiet=True`` together is a programmer error and
    must be caught at the argparse layer before this is called.
    """
    if verbose and quiet:
        raise ValueError("verbose and quiet are mutually exclusive")

    logger = logging.getLogger(LOGGER_NAME)
    if quiet:
        logger.setLevel(logging.WARNING)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not any(getattr(h, "_csemsearch_owned", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        # Tag so we don't add it twice on a re-configure.
        handler._csemsearch_owned = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    # Don't propagate up to root — would otherwise print everything twice
    # if a parent application (e.g. a test runner) has its own root handler.
    logger.propagate = False
    return logger
