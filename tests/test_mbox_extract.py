"""Tests for server/parsers.py:_extract_mbox — indexer-side mbox extraction."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_mbox_messages import _make_entry, _make_mbox


def test_extract_mbox_returns_list_of_dicts_with_text_and_metadata(
    tmp_path: Path,
) -> None:
    from server.parsers import _extract_mbox

    mbox = _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "one", "Message-ID": "<m1@x>"}, "body1"),
        _make_entry({"From": "b@x", "Subject": "two", "Message-ID": "<m2@x>"}, "body2"),
    )
    parts = _extract_mbox(mbox)
    assert len(parts) == 2
    for part in parts:
        assert isinstance(part["text"], str)
        assert isinstance(part["metadata"], dict)


def test_extract_mbox_folds_header_into_text(tmp_path: Path) -> None:
    from server.parsers import _extract_mbox

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {
                "From": "alice@x",
                "Subject": "hello",
                "Message-ID": "<m@x>",
                "Date": "Mon, 1 Jan 2024 00:00:00 +0000",
            },
            "body content",
        ),
    )
    [part] = _extract_mbox(mbox)
    head = part["text"].splitlines()[:4]
    assert head[0] == "Subject: hello"
    assert head[1] == "From: alice@x"
    assert head[2].startswith("Date: ")
    assert head[3].startswith("Thread: ")


def test_extract_mbox_metadata_includes_thread_id(tmp_path: Path) -> None:
    from server.parsers import _extract_mbox

    mbox = _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "s", "Message-ID": "<m@x>"}, "body"),
    )
    [part] = _extract_mbox(mbox)
    tid = part["metadata"]["thread_id"]
    assert re.fullmatch(r"[0-9a-f]{16}", tid)


def test_extract_mbox_metadata_includes_in_reply_to_and_references(
    tmp_path: Path,
) -> None:
    from server.parsers import _extract_mbox

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {"From": "a@x", "Subject": "root", "Message-ID": "<root@x>"},
            "rb",
        ),
        _make_entry(
            {
                "From": "b@x",
                "Subject": "Re: root",
                "Message-ID": "<reply@x>",
                "In-Reply-To": "<root@x>",
                "References": "<root@x>",
            },
            "rep",
        ),
    )
    parts = _extract_mbox(mbox)
    reply = parts[1]
    assert reply["metadata"]["in_reply_to"] == "<root@x>"
    assert reply["metadata"]["references"] == "<root@x>"


def test_extract_mbox_empty_mbox_returns_empty_list(tmp_path: Path) -> None:
    from server.parsers import _extract_mbox

    mbox = tmp_path / "empty.mbox"
    mbox.write_bytes(b"")
    assert _extract_mbox(mbox) == []


def test_extract_mbox_keeps_header_when_body_empty(tmp_path: Path) -> None:
    """Mirrors PST's _pst_message_to_part — header alone keeps message attributable."""
    from server.parsers import _extract_mbox

    mbox = _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "bare", "Message-ID": "<m@x>"}, ""),
    )
    [part] = _extract_mbox(mbox)
    assert "Subject: bare" in part["text"]


def test_extract_text_dispatches_mbox_extension(tmp_path: Path) -> None:
    from server.parsers import extract_text

    mbox = _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "x", "Message-ID": "<m@x>"}, "body"),
        name="archive.mbox",
    )
    parts = extract_text(mbox)
    assert len(parts) == 1
    assert "thread_id" in parts[0]["metadata"]


def test_supported_extensions_includes_mbox() -> None:
    from server.parsers import SUPPORTED_EXTENSIONS

    assert ".mbox" in SUPPORTED_EXTENSIONS
