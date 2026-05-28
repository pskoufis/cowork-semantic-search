"""``csemsearch`` — command-line driver for cowork-semantic-search.

The CLI shares the indexer's library code path (``server.indexer``,
``server.search``, ``server.unpacker``) with the MCP server, but adds rich
terminal output and SIGINT-driven cancellation for long-running runs over
large corpora. The MCP tool remains the preferred entry point for ad-hoc
agent calls; the CLI is the preferred entry point for hand-driven indexing.
"""
