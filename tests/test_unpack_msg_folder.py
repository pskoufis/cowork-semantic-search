"""Tests for scripts/unpack_msg_folder.py — batch .msg extractor that
preserves subdirectory layout.

Like test_msg_unpack.py, these tests inject a fake ``read_message`` so we
never need binary .msg fixtures. The unit under test is the script's
traversal + target-layout behavior; the parsing is exercised by the
in-place unit tests.
"""

from __future__ import annotations

import time
import os
from pathlib import Path


def _msg(
    *,
    from_addr: str = "alice@example.com",
    to: str = "bob@example.com",
    cc: str = "",
    subject: str = "Hello",
    date: str = "Mon, 1 Jan 2024 00:00:00 +0000",
    message_id: str = "<a@x>",
    body: str = "body",
    attachments: tuple = (),
):
    from msg_handling.messages import ParsedMessage
    return ParsedMessage(
        from_addr=from_addr, to=to, cc=cc, subject=subject, date=date,
        message_id=message_id, body=body, attachments=attachments,
    )


def _att(filename: str, data: bytes = b"x", is_msg: bool = False):
    from msg_handling.messages import ParsedAttachment
    return ParsedAttachment(filename=filename, data=data, is_msg=is_msg)


def _install_reader(monkeypatch, reader):
    from msg_handling import unpack
    monkeypatch.setattr(unpack, "read_message", reader)


def _drop_msg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")


def test_extracts_each_msg_into_target_with_subdir_layout(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.unpack_msg_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_msg(src / "a" / "foo.msg")
    _drop_msg(src / "b" / "deep" / "bar.msg")

    def reader(p):
        p = Path(p)
        if p.name == "foo.msg":
            return _msg(subject="Foo", body="foo body",
                        attachments=(_att("alpha.pdf", b"AAA"),))
        if p.name == "bar.msg":
            return _msg(subject="Bar", body="bar body",
                        attachments=(_att("beta.pdf", b"BBB"),))
        raise FileNotFoundError(f"no fake for {p}")
    _install_reader(monkeypatch, reader)

    rc = main([str(src), str(dst)])
    assert rc == 0

    # Target mirrors source's subdir structure.
    assert (dst / "a" / "foo.txt").is_file()
    assert (dst / "a" / "attachments" / "foo__alpha.pdf").read_bytes() == b"AAA"
    assert (dst / "b" / "deep" / "bar.txt").is_file()
    assert (dst / "b" / "deep" / "attachments" / "bar__beta.pdf").read_bytes() == b"BBB"

    # Source is untouched.
    assert not (src / "a" / "foo.txt").exists()
    assert not (src / "a" / "attachments").exists()
    assert (src / "a" / "foo.msg").is_file()


def test_idempotent_on_second_run(tmp_path: Path, monkeypatch) -> None:
    from scripts.unpack_msg_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_msg(src / "foo.msg")
    _install_reader(monkeypatch, lambda p: _msg(subject="A", body="b"))

    assert main([str(src), str(dst)]) == 0
    txt = dst / "foo.txt"
    first_mtime = txt.stat().st_mtime_ns

    # Make the target .txt strictly newer than the source .msg, then verify
    # the second run does not re-parse anything (tripwire).
    future = int(time.time()) + 60
    os.utime(txt, (future, future))

    def tripwire(_p):
        raise AssertionError("read_message should not run on idempotent skip")
    from msg_handling import unpack
    monkeypatch.setattr(unpack, "read_message", tripwire)

    assert main([str(src), str(dst)]) == 0
    assert txt.stat().st_mtime_ns >= first_mtime


def test_output_inside_input_is_refused(tmp_path: Path, monkeypatch) -> None:
    from scripts.unpack_msg_folder import main

    src = tmp_path / "src"
    src.mkdir()
    _drop_msg(src / "foo.msg")
    inside = src / "out"

    rc = main([str(src), str(inside)])
    assert rc != 0
    assert not inside.exists() or not any(inside.iterdir())


def test_broken_msg_does_not_abort_batch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scripts.unpack_msg_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_msg(src / "good.msg")
    _drop_msg(src / "broken.msg")

    def reader(p):
        p = Path(p)
        if p.name == "broken.msg":
            raise ValueError("intentional test failure")
        return _msg(subject="ok", body="hi")
    _install_reader(monkeypatch, reader)

    rc = main([str(src), str(dst)])
    assert rc != 0  # at least one failure → non-zero exit

    # The good one extracted; the broken one logged a warning.
    assert (dst / "good.txt").is_file()
    assert not (dst / "broken.txt").exists()
    err = capsys.readouterr().err
    assert "broken.msg" in err


def test_recursive_discovery_case_insensitive(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.unpack_msg_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_msg(src / "FOO.MSG")
    _drop_msg(src / "nested" / "BAR.Msg")
    _install_reader(monkeypatch, lambda p: _msg(subject="x", body="y"))

    assert main([str(src), str(dst)]) == 0
    assert (dst / "FOO.txt").is_file()
    assert (dst / "nested" / "BAR.txt").is_file()


def test_input_must_be_an_existing_directory(tmp_path: Path) -> None:
    from scripts.unpack_msg_folder import main

    rc = main([str(tmp_path / "nope"), str(tmp_path / "out")])
    assert rc != 0
