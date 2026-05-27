"""Tests for mbox_handling.unpack — thread-grouped on-disk unpacker."""

from __future__ import annotations

import base64
from pathlib import Path
from textwrap import dedent

import pytest


# Reuse the inline-mbox helpers from test_mbox_messages to keep fixture
# construction in one place.
from tests.test_mbox_messages import _make_entry, _make_mbox


def _list_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_unpack_defaults_output_dir_to_sibling_of_mbox(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {"From": "a@x", "Subject": "one", "Message-ID": "<m@x>"},
            "body",
        ),
    )
    expected = mbox.parent / f"{mbox.stem}_unpacked"

    rc = main([str(mbox)])
    assert rc == 0
    assert expected.is_dir()
    assert any(expected.rglob("msg-*.txt"))


def test_unpack_default_dir_suffix_on_non_empty_collision(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {"From": "a@x", "Subject": "one", "Message-ID": "<m@x>"},
            "body",
        ),
    )
    base = mbox.parent / f"{mbox.stem}_unpacked"
    base.mkdir()
    (base / "prior.txt").write_text("keep me")

    rc = main([str(mbox)])
    assert rc == 0
    # Original target untouched, suffix variant created.
    assert (base / "prior.txt").read_text() == "keep me"
    suffixed = mbox.parent / f"{mbox.stem}_unpacked-1"
    assert suffixed.is_dir()
    assert any(suffixed.rglob("msg-*.txt"))


def test_unpack_default_dir_advances_to_next_suffix(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {"From": "a@x", "Subject": "one", "Message-ID": "<m@x>"},
            "body",
        ),
    )
    base = mbox.parent / f"{mbox.stem}_unpacked"
    base.mkdir()
    (base / "a.txt").write_text("x")
    first = mbox.parent / f"{mbox.stem}_unpacked-1"
    first.mkdir()
    (first / "b.txt").write_text("y")

    rc = main([str(mbox)])
    assert rc == 0
    second = mbox.parent / f"{mbox.stem}_unpacked-2"
    assert second.is_dir()
    assert any(second.rglob("msg-*.txt"))
    # Earlier collisions left alone.
    assert (base / "a.txt").exists()
    assert (first / "b.txt").exists()


def test_unpack_default_dir_reuses_existing_empty_sibling(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {"From": "a@x", "Subject": "one", "Message-ID": "<m@x>"},
            "body",
        ),
    )
    base = mbox.parent / f"{mbox.stem}_unpacked"
    base.mkdir()

    rc = main([str(mbox)])
    assert rc == 0
    assert base.is_dir()
    assert any(base.rglob("msg-*.txt"))
    # No suffix variant created when the original was empty.
    assert not (mbox.parent / f"{mbox.stem}_unpacked-1").exists()


def test_unpack_creates_thread_directory_per_thread(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {
                "From": "alice@x",
                "Subject": "Thread one",
                "Message-ID": "<root@x>",
                "Date": "Mon, 1 Jan 2024 00:00:00 +0000",
            },
            "root body",
        ),
        _make_entry(
            {
                "From": "bob@x",
                "Subject": "Re: Thread one",
                "Message-ID": "<reply@x>",
                "In-Reply-To": "<root@x>",
                "Date": "Mon, 1 Jan 2024 00:01:00 +0000",
            },
            "reply body",
        ),
        _make_entry(
            {
                "From": "carol@x",
                "Subject": "Standalone",
                "Message-ID": "<solo@x>",
                "Date": "Mon, 1 Jan 2024 00:02:00 +0000",
            },
            "solo body",
        ),
    )
    out = tmp_path / "out"
    rc = main([str(mbox), "--output-dir", str(out)])
    assert rc == 0

    entries = sorted(p.name for p in out.iterdir() if p.is_dir())
    # One thread dir (root + reply) and one _unthreaded dir (solo).
    assert any(e.startswith("thread-0001_") for e in entries)
    assert "_unthreaded" in entries


def test_unpack_writes_message_txt_with_full_header_block(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {
                "From": "alice@example.com",
                "To": "bob@example.com",
                "Cc": "carol@example.com",
                "Subject": "Hello",
                "Message-ID": "<msg@x>",
                "Date": "Mon, 1 Jan 2024 00:00:00 +0000",
            },
            "Hello world.",
        ),
    )
    out = tmp_path / "out"
    main([str(mbox), "--output-dir", str(out)])

    txt_files = list((out / "_unthreaded").glob("msg-*.txt"))
    assert len(txt_files) == 1
    content = txt_files[0].read_text()
    assert "From: alice@example.com" in content
    assert "To: bob@example.com" in content
    assert "Cc: carol@example.com" in content
    assert "Subject: Hello" in content
    assert "Date: Mon, 1 Jan 2024 00:00:00 +0000" in content
    assert "Message-ID: <msg@x>" in content
    assert "In-Reply-To: " in content  # present even if empty
    assert "References: " in content
    assert "Hello world." in content


def test_unpack_writes_attachments_under_msg_subdir(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    gif_b64 = base64.b64encode(gif_bytes).decode("ascii")
    boundary = "BOUNDARY"
    body = dedent(
        f"""\
        --{boundary}
        Content-Type: text/plain; charset=utf-8

        body

        --{boundary}
        Content-Type: image/gif; name="pixel.gif"
        Content-Transfer-Encoding: base64
        Content-Disposition: attachment; filename="pixel.gif"

        {gif_b64}

        --{boundary}--
        """
    )
    entry = _make_entry(
        {
            "From": "a@x",
            "Subject": "with attach",
            "Message-ID": "<m1@x>",
            "Date": "Mon, 1 Jan 2024 00:00:00 +0000",
            "MIME-Version": "1.0",
            "Content-Type": f'multipart/mixed; boundary="{boundary}"',
        },
        body,
    )
    mbox = _make_mbox(tmp_path, entry)
    out = tmp_path / "out"
    main([str(mbox), "--output-dir", str(out)])

    gif_files = list(out.rglob("pixel.gif"))
    assert len(gif_files) == 1
    gif_path = gif_files[0]
    assert "attachments" in gif_path.parts
    assert gif_path.parent.name == "msg-0001"
    assert gif_path.read_bytes() == gif_bytes


def test_unpack_message_txt_references_its_attachments(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    payload_b64 = base64.b64encode(b"data").decode("ascii")
    boundary = "BOUNDARY"
    body = dedent(
        f"""\
        --{boundary}
        Content-Type: text/plain; charset=utf-8

        body

        --{boundary}
        Content-Type: application/pdf; name="report.pdf"
        Content-Transfer-Encoding: base64
        Content-Disposition: attachment; filename="report.pdf"

        {payload_b64}

        --{boundary}--
        """
    )
    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {
                "From": "a@x",
                "Subject": "withattach",
                "Message-ID": "<m@x>",
                "Date": "Mon, 1 Jan 2024 00:00:00 +0000",
                "MIME-Version": "1.0",
                "Content-Type": f'multipart/mixed; boundary="{boundary}"',
            },
            body,
        ),
    )
    out = tmp_path / "out"
    main([str(mbox), "--output-dir", str(out)])

    txt_files = list(out.rglob("msg-*.txt"))
    content = txt_files[0].read_text()
    assert "Attachments:" in content
    assert "attachments/msg-0001/report.pdf" in content


def test_unpack_orphans_go_to_unthreaded_bucket(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {
                "From": "a@x",
                "Subject": "orphan",
                "Message-ID": "<solo@x>",
                "Date": "Mon, 1 Jan 2024 00:00:00 +0000",
            },
            "body",
        ),
    )
    out = tmp_path / "out"
    main([str(mbox), "--output-dir", str(out)])

    assert (out / "_unthreaded").is_dir()
    txt_files = list((out / "_unthreaded").glob("msg-*.txt"))
    assert len(txt_files) == 1
    # No thread-NNNN_* dirs.
    assert not any(p.name.startswith("thread-") for p in out.iterdir())


def test_unpack_truncates_existing_output_dir(tmp_path: Path, capsys) -> None:
    from mbox_handling.unpack import main

    out = tmp_path / "out"
    out.mkdir()
    junk = out / "junk.txt"
    junk.write_text("garbage")

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {"From": "a@x", "Subject": "one", "Message-ID": "<m@x>"},
            "body",
        ),
    )
    main([str(mbox), "--output-dir", str(out)])
    assert not junk.exists()
    captured = capsys.readouterr()
    assert "clear" in captured.err.lower() or "notice" in captured.err.lower()


def test_unpack_sanitizes_subject_and_from_in_filenames(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {
                "From": "weird/name@x",
                "Subject": "a/b\\c unicode-é",
                "Message-ID": "<m@x>",
            },
            "body",
        ),
    )
    out = tmp_path / "out"
    main([str(mbox), "--output-dir", str(out)])

    for path in out.rglob("*.txt"):
        assert "/" not in path.name
        assert "\\" not in path.name
    for path in out.rglob("*"):
        if path.is_dir():
            assert "/" not in path.name
            assert "\\" not in path.name


def test_unpack_exit_code_zero_for_empty_mbox(tmp_path: Path) -> None:
    from mbox_handling.unpack import main

    mbox = tmp_path / "empty.mbox"
    mbox.write_bytes(b"")
    out = tmp_path / "out"
    rc = main([str(mbox), "--output-dir", str(out)])
    assert rc == 0
    assert out.is_dir()


def test_unpack_per_message_error_does_not_abort(
    monkeypatch, tmp_path: Path
) -> None:
    from mbox_handling import messages as messages_mod
    from mbox_handling.unpack import main

    mbox = _make_mbox(
        tmp_path,
        _make_entry(
            {"From": "a@x", "Subject": "good", "Message-ID": "<a@x>"},
            "first",
        ),
        _make_entry(
            {"From": "b@x", "Subject": "good", "Message-ID": "<b@x>"},
            "second",
        ),
    )

    real_extract = messages_mod._extract_body
    call_count = {"n": 0}

    def flaky(msg):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("kaboom")
        return real_extract(msg)

    monkeypatch.setattr(messages_mod, "_extract_body", flaky)

    out = tmp_path / "out"
    main([str(mbox), "--output-dir", str(out)])

    txts = list(out.rglob("msg-*.txt"))
    assert len(txts) == 2
    error_files = [p for p in txts if "error" in p.read_text().lower()]
    assert len(error_files) >= 1
    assert any("kaboom" in p.read_text() for p in error_files)
