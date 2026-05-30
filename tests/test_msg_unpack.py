"""Tests for msg_handling.unpack — per-message .msg unpacker.

extract-msg has no programmatic writer for .msg files, so these tests inject
a fake ``read_message`` instead of crafting real binary fixtures. The unit
under test is the file-layout / idempotency / cleanup / recursion logic, all
of which is observable through filesystem inspection.

One end-to-end test against a real extract-msg parse lives in
test_indexer.py once the indexer wiring lands.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


def _msg(
    *,
    from_addr: str = "alice@example.com",
    to: str = "bob@example.com",
    cc: str = "",
    subject: str = "Hello",
    date: str = "Mon, 1 Jan 2024 00:00:00 +0000",
    message_id: str = "<a@x>",
    body: str = "body text",
    attachments: tuple = (),
):
    from msg_handling.messages import ParsedMessage
    return ParsedMessage(
        from_addr=from_addr,
        to=to,
        cc=cc,
        subject=subject,
        date=date,
        message_id=message_id,
        body=body,
        attachments=attachments,
    )


def _att(filename: str, data: bytes = b"x", is_msg: bool = False):
    from msg_handling.messages import ParsedAttachment
    return ParsedAttachment(filename=filename, data=data, is_msg=is_msg)


def _install_reader(monkeypatch, mapping_factory):
    """Replace ``msg_handling.unpack.read_message`` with ``mapping_factory(path)``.

    Lazy resolution lets a single fake serve both outer and (post-write)
    inner .msg paths in the recursion test.
    """
    from msg_handling import unpack

    def fake(path):
        result = mapping_factory(Path(path))
        if result is None:
            raise FileNotFoundError(f"no fake message configured for {path}")
        return result

    monkeypatch.setattr(unpack, "read_message", fake)


def _touch_msg(path: Path, contents: bytes = b"\x00") -> None:
    path.write_bytes(contents)


def _bump_mtime(path: Path) -> None:
    future = int(time.time()) + 60
    os.utime(path, (future, future))


def test_unpack_writes_sibling_txt_and_attachments(tmp_path: Path, monkeypatch) -> None:
    from msg_handling.unpack import ensure_unpacked

    msg_path = tmp_path / "foo.msg"
    _touch_msg(msg_path)
    _install_reader(monkeypatch, lambda p: (
        _msg(
            subject="Greetings",
            body="hello world",
            attachments=(_att("report.pdf", b"PDF-DATA"),),
        ) if p.name == "foo.msg" else None
    ))

    ensure_unpacked(msg_path)

    txt = tmp_path / "foo.txt"
    assert txt.is_file()
    content = txt.read_text(encoding="utf-8")
    assert "Subject: Greetings" in content
    assert "hello world" in content
    assert "From: alice@example.com" in content

    attachment = tmp_path / "attachments" / "foo__report.pdf"
    assert attachment.is_file()
    assert attachment.read_bytes() == b"PDF-DATA"


def test_unpack_is_idempotent_when_txt_fresher(tmp_path: Path, monkeypatch) -> None:
    from msg_handling.unpack import ensure_unpacked
    from msg_handling import unpack

    msg_path = tmp_path / "foo.msg"
    _touch_msg(msg_path)
    _install_reader(monkeypatch, lambda p: _msg(subject="A", body="b"))

    ensure_unpacked(msg_path)
    txt = tmp_path / "foo.txt"
    assert txt.is_file()
    first_mtime_ns = txt.stat().st_mtime_ns

    # Force the .txt to be unambiguously newer than the .msg.
    _bump_mtime(txt)

    # Replacing the reader with a tripwire proves the skip branch ran.
    def tripwire(_path):
        raise AssertionError("read_message should not be called on idempotent skip")
    monkeypatch.setattr(unpack, "read_message", tripwire)

    ensure_unpacked(msg_path)
    assert txt.stat().st_mtime_ns >= first_mtime_ns


def test_unpack_reextracts_when_msg_newer(tmp_path: Path, monkeypatch) -> None:
    from msg_handling.unpack import ensure_unpacked

    msg_path = tmp_path / "foo.msg"
    _touch_msg(msg_path)

    state = {
        "body": "first",
        "attachment_name": "old.pdf",
        "attachment_data": b"OLD",
    }
    _install_reader(monkeypatch, lambda p: _msg(
        subject="A",
        body=state["body"],
        attachments=(_att(state["attachment_name"], state["attachment_data"]),),
    ))

    ensure_unpacked(msg_path)
    assert (tmp_path / "attachments" / "foo__old.pdf").is_file()

    state["body"] = "second"
    state["attachment_name"] = "new.pdf"
    state["attachment_data"] = b"NEW"
    _bump_mtime(msg_path)

    ensure_unpacked(msg_path)
    assert (tmp_path / "attachments" / "foo__new.pdf").read_bytes() == b"NEW"
    assert not (tmp_path / "attachments" / "foo__old.pdf").exists()
    assert "second" in (tmp_path / "foo.txt").read_text(encoding="utf-8")


def test_long_attachment_name_is_capped(tmp_path: Path, monkeypatch) -> None:
    """An over-long attachment name (which would blow past NAME_MAX and raise
    ENAMETOOLONG when written) is capped to a filesystem-safe length, keeping
    the extension, so the unpack succeeds instead of aborting the .msg."""
    from msg_handling.unpack import ensure_unpacked

    msg = tmp_path / "m.msg"
    _touch_msg(msg)
    long_name = "x" * 400 + ".pdf"
    _install_reader(
        monkeypatch,
        lambda p: _msg(body="hi", attachments=(_att(long_name, b"DATA"),)),
    )

    ensure_unpacked(msg)  # must not raise OSError(ENAMETOOLONG)

    att_dir = tmp_path / "attachments"
    files = [p for p in att_dir.iterdir() if p.is_file()]
    assert len(files) == 1
    name = files[0].name
    assert len(name.encode("utf-8")) <= 255
    assert name.endswith(".pdf")
    assert files[0].read_bytes() == b"DATA"
    # The .txt was still produced and references the capped attachment name.
    txt = (tmp_path / "m.txt").read_text(encoding="utf-8")
    assert name in txt


def test_distinct_long_names_do_not_collide(tmp_path: Path, monkeypatch) -> None:
    """Two different over-long attachment names must cap to distinct on-disk
    names (a uniqueness hash), not silently overwrite each other."""
    from msg_handling.unpack import ensure_unpacked

    msg = tmp_path / "m.msg"
    _touch_msg(msg)
    a = "a" * 400 + ".pdf"
    b = "b" * 400 + ".pdf"
    _install_reader(
        monkeypatch,
        lambda p: _msg(attachments=(_att(a, b"AAA"), _att(b, b"BBB"))),
    )

    ensure_unpacked(msg)

    att_dir = tmp_path / "attachments"
    payloads = sorted(p.read_bytes() for p in att_dir.iterdir() if p.is_file())
    assert payloads == [b"AAA", b"BBB"]  # both survived, no clobber


def test_two_msgs_share_one_attachments_folder(tmp_path: Path, monkeypatch) -> None:
    from msg_handling.unpack import ensure_unpacked

    foo = tmp_path / "foo.msg"
    bar = tmp_path / "bar.msg"
    _touch_msg(foo)
    _touch_msg(bar)

    def reader(p: Path):
        if p.name == "foo.msg":
            return _msg(attachments=(_att("report.pdf", b"FOO"),))
        if p.name == "bar.msg":
            return _msg(attachments=(_att("report.pdf", b"BAR"),))
        return None
    _install_reader(monkeypatch, reader)

    ensure_unpacked(foo)
    ensure_unpacked(bar)

    assert (tmp_path / "attachments" / "foo__report.pdf").read_bytes() == b"FOO"
    assert (tmp_path / "attachments" / "bar__report.pdf").read_bytes() == b"BAR"


def test_re_extract_does_not_clobber_sibling_msg_attachments(
    tmp_path: Path, monkeypatch
) -> None:
    from msg_handling.unpack import ensure_unpacked

    foo = tmp_path / "foo.msg"
    bar = tmp_path / "bar.msg"
    _touch_msg(foo)
    _touch_msg(bar)

    foo_attachment = {"name": "a.pdf", "data": b"FOO-A"}

    def reader(p: Path):
        if p.name == "foo.msg":
            return _msg(attachments=(_att(foo_attachment["name"], foo_attachment["data"]),))
        if p.name == "bar.msg":
            return _msg(attachments=(_att("b.pdf", b"BAR-B"),))
        return None
    _install_reader(monkeypatch, reader)

    ensure_unpacked(foo)
    ensure_unpacked(bar)

    foo_attachment["name"] = "a2.pdf"
    foo_attachment["data"] = b"FOO-A2"
    _bump_mtime(foo)

    ensure_unpacked(foo)

    assert (tmp_path / "attachments" / "bar__b.pdf").read_bytes() == b"BAR-B"
    assert (tmp_path / "attachments" / "foo__a2.pdf").read_bytes() == b"FOO-A2"
    assert not (tmp_path / "attachments" / "foo__a.pdf").exists()


def test_nested_msg_attachment_recurses(tmp_path: Path, monkeypatch) -> None:
    from msg_handling.unpack import ensure_unpacked

    outer = tmp_path / "outer.msg"
    _touch_msg(outer)

    def reader(p: Path):
        if p.name == "outer.msg":
            return _msg(
                subject="Outer",
                body="outer body",
                attachments=(_att("inner.msg", b"INNER-BYTES", is_msg=True),),
            )
        # The inner blob lives in attachments/ and is named with the parent's
        # stem prefix once the outer unpack writes it. The recursion calls
        # back into read_message for this path.
        if p.name == "outer__inner.msg":
            return _msg(subject="Inner", body="inner body", attachments=())
        return None
    _install_reader(monkeypatch, reader)

    ensure_unpacked(outer)

    outer_txt = tmp_path / "outer.txt"
    assert outer_txt.is_file()
    assert "outer body" in outer_txt.read_text(encoding="utf-8")

    inner_blob = tmp_path / "attachments" / "outer__inner.msg"
    assert inner_blob.is_file()
    assert inner_blob.read_bytes() == b"INNER-BYTES"

    inner_txt = tmp_path / "attachments" / "outer__inner.txt"
    assert inner_txt.is_file()
    assert "inner body" in inner_txt.read_text(encoding="utf-8")


def test_broken_msg_logs_warning_and_does_not_raise(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from msg_handling import unpack
    from msg_handling.unpack import ensure_unpacked

    msg_path = tmp_path / "broken.msg"
    _touch_msg(msg_path)

    def explode(_path):
        raise ValueError("intentional test failure")
    monkeypatch.setattr(unpack, "read_message", explode)

    ensure_unpacked(msg_path)  # must not raise

    captured = capsys.readouterr()
    assert "broken.msg" in captured.err
    assert not (tmp_path / "broken.txt").exists()


def test_broken_msg_preserves_prior_outputs(tmp_path: Path, monkeypatch) -> None:
    """A failed re-extract must not delete the previously-extracted .txt and
    attachments — cleanup happens only after a successful parse."""
    from msg_handling import unpack
    from msg_handling.unpack import ensure_unpacked

    msg_path = tmp_path / "foo.msg"
    _touch_msg(msg_path)
    _install_reader(monkeypatch, lambda p: _msg(
        subject="First",
        body="first body",
        attachments=(_att("doc.pdf", b"PDF1"),),
    ))
    ensure_unpacked(msg_path)
    assert (tmp_path / "foo.txt").is_file()
    assert (tmp_path / "attachments" / "foo__doc.pdf").is_file()

    # Bump the source so we re-extract on next call, but make the reader fail.
    _bump_mtime(msg_path)
    def explode(_path):
        raise RuntimeError("simulated parse failure")
    monkeypatch.setattr(unpack, "read_message", explode)

    ensure_unpacked(msg_path)

    # Prior outputs preserved
    assert (tmp_path / "foo.txt").read_text(encoding="utf-8").find("first body") != -1
    assert (tmp_path / "attachments" / "foo__doc.pdf").read_bytes() == b"PDF1"


def test_attachment_filename_path_traversal_is_neutralized(
    tmp_path: Path, monkeypatch
) -> None:
    """A malicious .msg whose attachment name tries to escape via ../ should
    land inside attachments/ regardless."""
    from msg_handling.unpack import ensure_unpacked

    msg_path = tmp_path / "foo.msg"
    _touch_msg(msg_path)
    _install_reader(monkeypatch, lambda p: _msg(
        attachments=(_att("../escape.txt", b"PAYLOAD"),),
    ))

    ensure_unpacked(msg_path)

    # Sanitized — basename used, traversal stripped.
    assert (tmp_path / "attachments" / "foo__escape.txt").is_file()
    # Nothing landed outside attachments/.
    assert not (tmp_path.parent / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_ensure_unpacked_writes_to_target_when_provided(
    tmp_path: Path, monkeypatch
) -> None:
    """With target=<dir>, outputs land in that directory instead of next to
    the source .msg. The source folder is untouched."""
    from msg_handling.unpack import ensure_unpacked

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    msg_path = src / "foo.msg"
    _touch_msg(msg_path)
    _install_reader(monkeypatch, lambda p: _msg(
        subject="Hello",
        body="hi there",
        attachments=(_att("report.pdf", b"PDF"),),
    ))

    ensure_unpacked(msg_path, target=dst)

    # Outputs land in dst, not src.
    assert (dst / "foo.txt").is_file()
    assert (dst / "attachments" / "foo__report.pdf").read_bytes() == b"PDF"
    # Source is untouched aside from the .msg we created.
    assert not (src / "foo.txt").exists()
    assert not (src / "attachments").exists()


def test_ensure_unpacked_target_is_idempotent_when_txt_fresher(
    tmp_path: Path, monkeypatch
) -> None:
    """The mtime skip path checks the .txt inside `target`, not next to the
    source .msg."""
    from msg_handling.unpack import ensure_unpacked
    from msg_handling import unpack

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    msg_path = src / "foo.msg"
    _touch_msg(msg_path)
    _install_reader(monkeypatch, lambda p: _msg(subject="A", body="b"))
    ensure_unpacked(msg_path, target=dst)
    target_txt = dst / "foo.txt"
    assert target_txt.is_file()
    _bump_mtime(target_txt)

    def tripwire(_path):
        raise AssertionError("read_message should not be called on idempotent skip")
    monkeypatch.setattr(unpack, "read_message", tripwire)

    ensure_unpacked(msg_path, target=dst)  # must not raise


def test_ensure_unpacked_target_cleans_only_stem_scoped_files(
    tmp_path: Path, monkeypatch
) -> None:
    """A re-extract via target= only removes target/<stem>.txt and
    target/attachments/<stem>__* — pre-existing unrelated files in target
    are preserved."""
    from msg_handling.unpack import ensure_unpacked

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    msg_path = src / "foo.msg"
    _touch_msg(msg_path)
    foo_att_name = {"name": "old.pdf"}
    _install_reader(monkeypatch, lambda p: _msg(
        attachments=(_att(foo_att_name["name"], b"x"),),
    ))
    ensure_unpacked(msg_path, target=dst)

    # Drop an unrelated bystander into target before re-extract.
    bystander = dst / "attachments" / "bar__keep.pdf"
    bystander.write_bytes(b"KEEP")
    _bump_mtime(msg_path)
    foo_att_name["name"] = "new.pdf"

    ensure_unpacked(msg_path, target=dst)

    assert (dst / "attachments" / "foo__new.pdf").is_file()
    assert not (dst / "attachments" / "foo__old.pdf").exists()
    assert bystander.read_bytes() == b"KEEP"


def test_index_folder_preprocesses_msg(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: index_folder() runs the .msg preprocessing pass, the
    spawned .txt is indexed, and the .msg itself is not."""
    from unittest.mock import patch
    import numpy as np
    from server.indexer import index_folder
    from server.store import VectorStore, EMBEDDING_DIM
    from msg_handling import unpack as msg_unpack

    def fake_embed(texts):
        # Deterministic non-zero vectors; one row per chunk.
        return np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32) * 0.1

    msg_path = tmp_path / "report.msg"
    _touch_msg(msg_path)

    def reader(p: Path):
        if p.name == "report.msg":
            return _msg(
                subject="Quarterly review",
                body="Revenue grew 23% this quarter.",
            )
        return None
    monkeypatch.setattr(
        msg_unpack,
        "read_message",
        lambda p: reader(Path(p)) or (_ for _ in ()).throw(
            FileNotFoundError(f"no fake for {p}")
        ),
    )

    db_path = str(tmp_path / "testdb")
    mock_model = type(
        "MockModel",
        (),
        {"encode": lambda self, texts, **kw: fake_embed(texts)},
    )()

    with patch("server.indexer.get_model", return_value=mock_model):
        result = index_folder(str(tmp_path), db_path=db_path)

    assert result["status"] == "completed"
    # The unpack pass should have written the sibling .txt.
    assert (tmp_path / "report.txt").is_file()
    indexed = VectorStore(db_path).get_all_files()
    assert any(f.endswith("report.txt") for f in indexed), (
        f"spawned .txt should be indexed; got: {indexed}"
    )
    assert not any(f.endswith(".msg") for f in indexed), (
        f".msg should not be indexed; got: {indexed}"
    )
