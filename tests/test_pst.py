"""Tests for the public .pst routing in server/parsers.py.

The PST extraction logic — previously inline in server/parsers.py — now
lives in pst_handling/. The tests for the walker, body extraction,
mail-only filter, attachment handling, and HTML/RTF stripping all moved
to tests/test_pst_unpack.py alongside the unpacker they exercise.

What remains here is the negative-dispatch contract: ``.pst`` must no
longer be in ``SUPPORTED_EXTENSIONS`` and ``extract_text`` must refuse
it, because PST is now handled by the preprocessing pass in
``server.unpacker.run_unpack_passes`` rather than directly.
"""

from pathlib import Path

import pytest

from server.parsers import extract_text, SUPPORTED_EXTENSIONS


def test_pst_is_not_in_supported_extensions():
    assert ".pst" not in SUPPORTED_EXTENSIONS


def test_extract_text_rejects_pst_with_helpful_error():
    """A stray caller still passing a .pst into extract_text should get
    a clear error pointing at the preprocessing pass, not silently
    fall through."""
    with pytest.raises(ValueError, match=r"Unsupported file type: \.pst"):
        extract_text(Path("/some/archive.pst"))
