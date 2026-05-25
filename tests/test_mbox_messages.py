"""Tests for mbox_handling.messages — the shared parsed-message iterator.

The mbox format is plain text, so fixtures are built inline as bytes — no
need for the duck-typed fake hierarchy that tests/test_pst.py needs for
pypff. Every test writes a synthetic .mbox to tmp_path and asserts on the
ParsedMessage stream that comes out.
"""

from __future__ import annotations

import base64
from pathlib import Path
from textwrap import dedent

import pytest


def _make_entry(headers: dict[str, str], body: str) -> bytes:
    """Build one mbox entry: ``From `` separator + headers + blank + body + trailing blank."""
    lines = ["From mboxrd@z Thu Jan  1 00:00:00 1970"]
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    lines.append("")  # header/body separator
    lines.append(body)
    lines.append("")  # trailing blank between messages
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_mbox(tmp_path: Path, *entries: bytes, name: str = "test.mbox") -> Path:
    """Concatenate entries into an mbox file at ``tmp_path / name``."""
    path = tmp_path / name
    path.write_bytes(b"".join(entries))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_iter_messages_yields_one_per_entry(tmp_path: Path) -> None:
    from mbox_handling.messages import iter_messages

    mbox = _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "one"}, "first body"),
        _make_entry({"From": "b@x", "Subject": "two"}, "second body"),
    )
    messages = list(iter_messages(mbox))
    assert len(messages) == 2
    assert messages[0].index == 1
    assert messages[1].index == 2
    assert messages[0].subject == "one"
    assert messages[1].subject == "two"


def test_iter_messages_extracts_threading_headers(tmp_path: Path) -> None:
    from mbox_handling.messages import iter_messages

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {
                "From": "alice@x",
                "Subject": "thread root",
                "Message-ID": "<root@x>",
                "In-Reply-To": "<parent@x>",
                "References": "<grand@x> <parent@x>",
            },
            "body",
        ),
    )
    [msg] = list(iter_messages(mbox))
    assert msg.message_id == "<root@x>"
    assert msg.in_reply_to == "<parent@x>"
    assert msg.references == ("<grand@x>", "<parent@x>")


def test_iter_messages_missing_threading_headers_returns_none_and_empty_tuple(
    tmp_path: Path,
) -> None:
    from mbox_handling.messages import iter_messages

    mbox = _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "bare"}, "body"),
    )
    [msg] = list(iter_messages(mbox))
    assert msg.message_id is None
    assert msg.in_reply_to is None
    assert msg.references == ()


def test_iter_messages_prefers_plain_text_over_html(tmp_path: Path) -> None:
    from mbox_handling.messages import iter_messages

    # Build a multipart/alternative message with both plain and HTML parts.
    boundary = "BOUNDARY"
    body = dedent(
        f"""\
        --{boundary}
        Content-Type: text/plain; charset=utf-8

        plain text wins

        --{boundary}
        Content-Type: text/html; charset=utf-8

        <p>html loses</p>

        --{boundary}--
        """
    )
    entry = _make_entry(
        {
            "From": "a@x",
            "Subject": "alt",
            "MIME-Version": "1.0",
            "Content-Type": f'multipart/alternative; boundary="{boundary}"',
        },
        body,
    )
    mbox = _make_mbox(tmp_path, entry)
    [msg] = list(iter_messages(mbox))
    assert "plain text wins" in msg.body
    assert "<p>" not in msg.body
    assert "html loses" not in msg.body


def test_iter_messages_falls_back_to_html_when_plain_absent(tmp_path: Path) -> None:
    from mbox_handling.messages import iter_messages

    entry = _make_entry(
        {"From": "a@x", "Subject": "html", "Content-Type": "text/html; charset=utf-8"},
        "<p>only html here</p>",
    )
    mbox = _make_mbox(tmp_path, entry)
    [msg] = list(iter_messages(mbox))
    assert "only html here" in msg.body
    assert "<p>" not in msg.body


def test_iter_messages_extracts_attachments(tmp_path: Path) -> None:
    from mbox_handling.messages import iter_messages

    gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    gif_b64 = base64.b64encode(gif_bytes).decode("ascii")
    boundary = "BOUNDARY"
    body = dedent(
        f"""\
        --{boundary}
        Content-Type: text/plain; charset=utf-8

        body before attachment

        --{boundary}
        Content-Type: image/gif; name="pixel.gif"
        Content-Transfer-Encoding: base64
        Content-Disposition: attachment; filename="pixel.gif"
        Content-ID: <cid-123@x>

        {gif_b64}

        --{boundary}--
        """
    )
    entry = _make_entry(
        {
            "From": "a@x",
            "Subject": "with attach",
            "MIME-Version": "1.0",
            "Content-Type": f'multipart/mixed; boundary="{boundary}"',
        },
        body,
    )
    mbox = _make_mbox(tmp_path, entry)
    [msg] = list(iter_messages(mbox))
    assert len(msg.attachments) == 1
    att = msg.attachments[0]
    assert att.filename == "pixel.gif"
    assert att.content_type == "image/gif"
    assert att.content_id == "<cid-123@x>"
    assert att.data == gif_bytes


def test_iter_messages_sanitizes_unsafe_attachment_filenames(tmp_path: Path) -> None:
    from mbox_handling.messages import iter_messages

    payload_b64 = base64.b64encode(b"x").decode("ascii")
    boundary = "BOUNDARY"
    body = dedent(
        f"""\
        --{boundary}
        Content-Type: application/octet-stream; name="../etc/passwd"
        Content-Transfer-Encoding: base64
        Content-Disposition: attachment; filename="../etc/passwd"

        {payload_b64}

        --{boundary}--
        """
    )
    entry = _make_entry(
        {
            "From": "a@x",
            "Subject": "evil",
            "MIME-Version": "1.0",
            "Content-Type": f'multipart/mixed; boundary="{boundary}"',
        },
        body,
    )
    mbox = _make_mbox(tmp_path, entry)
    [msg] = list(iter_messages(mbox))
    assert len(msg.attachments) == 1
    name = msg.attachments[0].filename
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name or name == "..etcpasswd" or name.endswith("passwd")


def test_iter_messages_handles_charset_errors_gracefully(tmp_path: Path) -> None:
    """A part declaring a charset Python doesn't know should still yield a body."""
    from mbox_handling.messages import iter_messages

    entry = _make_entry(
        {
            "From": "a@x",
            "Subject": "bad charset",
            "Content-Type": "text/plain; charset=zzz-not-a-real-charset",
            "Content-Transfer-Encoding": "8bit",
        },
        "actual content here",
    )
    mbox = _make_mbox(tmp_path, entry)
    [msg] = list(iter_messages(mbox))
    assert msg.error is None
    assert "actual content here" in msg.body


def test_iter_messages_isolates_per_message_errors(monkeypatch, tmp_path: Path) -> None:
    """One broken message must not abort the iterator; broken record has error set."""
    from mbox_handling import messages as messages_mod

    mbox = _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "first"}, "first body"),
        _make_entry({"From": "b@x", "Subject": "second"}, "second body"),
    )

    real_extract = messages_mod._extract_body
    call_count = {"n": 0}

    def flaky(msg):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("synthetic failure")
        return real_extract(msg)

    monkeypatch.setattr(messages_mod, "_extract_body", flaky)

    out = list(messages_mod.iter_messages(mbox))
    assert len(out) == 2
    assert out[0].error is not None
    assert "synthetic failure" in out[0].error
    assert out[1].error is None
    assert out[1].body.strip() == "second body"


def test_iter_messages_decodes_rfc2047_headers(tmp_path: Path) -> None:
    from mbox_handling.messages import iter_messages

    mbox = _make_mbox(
        tmp_path,
        _make_entry({"From": "a@x", "Subject": "=?utf-8?b?VGVzdA==?="}, "body"),
    )
    [msg] = list(iter_messages(mbox))
    assert msg.subject == "Test"
