"""Tests for cli.progress — the tqdm wrapper and its plain-counter fallback.

The wrapper has two important contracts:

1. With or without tqdm installed, ``ProgressBar(total=N)`` accepts ``update``
   calls and advances its internal ``n`` counter correctly.
2. The fallback path prints something useful to stderr (so the user sees
   progress even on a barebones install).
"""

from __future__ import annotations

import io
import sys

import pytest

import cli.progress as progress_mod
from cli.progress import ProgressBar, _truncate_middle


# ---------------------------------------------------------------------------
# Core behavior — works regardless of tqdm presence
# ---------------------------------------------------------------------------


def test_progress_bar_tracks_internal_counter():
    bar = ProgressBar(total=10)
    bar.update(3)
    bar.update(2)
    assert bar.n == 5
    bar.close()


def test_progress_bar_set_to_advances_to_absolute():
    bar = ProgressBar(total=10)
    bar.set_to(4)
    assert bar.n == 4
    bar.set_to(7)
    assert bar.n == 7
    bar.close()


def test_progress_bar_context_manager_closes_cleanly():
    with ProgressBar(total=5) as bar:
        bar.update(2)
        assert bar.n == 2
    # No exception raised on close — that's the contract.


# ---------------------------------------------------------------------------
# Plain-counter fallback (tqdm absent)
# ---------------------------------------------------------------------------


def test_plain_counter_writes_to_stderr_when_tqdm_missing(
    monkeypatch, capsys
):
    monkeypatch.setattr(progress_mod, "HAS_TQDM", False)
    monkeypatch.setattr(progress_mod, "tqdm", None)

    with ProgressBar(total=3, desc="indexing") as bar:
        bar.update(1, postfix="file-a.txt")
        bar.update(1, postfix="file-b.txt")

    captured = capsys.readouterr()
    assert "indexing" in captured.err
    assert "[1/3]" in captured.err
    assert "[2/3]" in captured.err
    assert "file-a.txt" in captured.err


# ---------------------------------------------------------------------------
# Path truncation
# ---------------------------------------------------------------------------


def test_truncate_middle_short_strings_pass_through():
    assert _truncate_middle("short", 80) == "short"


def test_truncate_middle_keeps_tail_visible():
    """The tail (filename + extension) is the most important part of a path;
    keep more of it visible than the head."""
    long = "a/very/deep/nested/path/to/important-file.txt"
    truncated = _truncate_middle(long, 25)
    assert len(truncated) <= 25
    assert truncated.endswith("important-file.txt") or truncated.endswith(
        "ortant-file.txt"
    )


def test_truncate_middle_includes_ellipsis_when_truncated():
    long = "x" * 200
    truncated = _truncate_middle(long, 20)
    assert "…" in truncated
    assert len(truncated) == 20
