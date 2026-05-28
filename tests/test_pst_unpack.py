"""Tests for pst_handling.unpack — per-archive .pst unpacker.

libpff has no programmatic writer, so these tests inject a fake
``iter_pst_messages`` instead of crafting binary .pst fixtures. The
unit under test is the file-layout / idempotency / mail-filter /
attachment-streaming logic, all observable through filesystem
inspection.

Body / header / attachment-size / message-class tests for the
pst_handling.messages helpers (``_pst_body``, ``_is_mail_item``, etc.)
live alongside in this file because the helpers moved here from
server.parsers when .pst was switched from inline extraction to
preprocessing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


# --- Test helpers -----------------------------------------------------------


def _msg(
    *,
    folder_path: str = "",
    subject: str = "Hello",
    sender: str = "alice@example.com",
    date: str = "2024-03-01 12:00:00+00:00",
    body: str = "body text",
    attachments: tuple = (),
    error: str | None = None,
):
    from pst_handling.messages import ParsedPstMessage
    return ParsedPstMessage(
        folder_path=folder_path,
        subject=subject,
        sender=sender,
        date=date,
        body=body,
        attachments=attachments,
        error=error,
    )


def _att(filename: str, data: bytes = b"x", size: int | None = None):
    """Build a ParsedPstAttachment whose ``write_to`` dumps ``data`` to disk."""
    from pst_handling.messages import ParsedPstAttachment

    actual_size = len(data) if size is None else size

    def write_to(target: Path) -> None:
        target.write_bytes(data)

    return ParsedPstAttachment(
        filename=filename, size=actual_size, write_to=write_to
    )


def _install_iter(monkeypatch, messages):
    """Replace pst_handling.unpack.iter_pst_messages with a fake that yields
    ``messages`` (a list or a callable returning a list)."""
    from pst_handling import unpack

    def fake(pst_path):
        result = messages(pst_path) if callable(messages) else messages
        yield from result

    monkeypatch.setattr(unpack, "iter_pst_messages", fake)


def _touch_pst(path: Path, contents: bytes = b"\x00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def _bump_mtime(path: Path) -> None:
    future = int(time.time()) + 60
    os.utime(path, (future, future))


# --- ensure_unpacked file layout --------------------------------------------


def test_unpack_writes_sibling_tree_mirroring_folder_hierarchy(
    tmp_path: Path, monkeypatch
) -> None:
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top/Inbox", subject="Mail one",
             sender="alice@example.com",
             date="2024-03-01 12:00:00+00:00", body="body one"),
        _msg(folder_path="Top/Inbox", subject="Mail two",
             sender="bob@example.com",
             date="2024-03-02 12:00:00+00:00", body="body two"),
        _msg(folder_path="Top/Sent", subject="Mail three",
             sender="me@example.com",
             date="2024-03-03 12:00:00+00:00", body="body three"),
    ])

    out = ensure_unpacked(pst_path)
    assert out == tmp_path / "foo_unpacked"

    inbox = out / "Top" / "Inbox"
    assert any(inbox.glob("msg-0001_20240301_*.txt")), \
        list(inbox.iterdir())
    assert any(inbox.glob("msg-0002_20240302_*.txt")), \
        list(inbox.iterdir())

    sent = out / "Top" / "Sent"
    assert any(sent.glob("msg-0001_20240303_*.txt")), \
        list(sent.iterdir())


def test_unpack_writes_attachments_under_per_message_dir(
    tmp_path: Path, monkeypatch
) -> None:
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top/Inbox", body="see attached",
             attachments=(
                 _att("report.pdf", b"PDF-DATA"),
                 _att("photo.jpg", b"JPG-DATA"),
             )),
    ])

    out = ensure_unpacked(pst_path)
    inbox = out / "Top" / "Inbox"
    attachments_dir = inbox / "attachments" / "msg-0001"
    assert (attachments_dir / "report.pdf").read_bytes() == b"PDF-DATA"
    assert (attachments_dir / "photo.jpg").read_bytes() == b"JPG-DATA"

    txt = next(inbox.glob("msg-0001_*.txt"))
    content = txt.read_text(encoding="utf-8")
    assert "Attachments:" in content
    assert "attachments/msg-0001/report.pdf" in content
    assert "attachments/msg-0001/photo.jpg" in content


def test_unpack_header_block_includes_folder_and_canonical_lines(
    tmp_path: Path, monkeypatch
) -> None:
    """Header layout must match the mbox/msg shape so unpacked PST
    messages produce identical embedding context."""
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top/Inbox", subject="Quarterly numbers",
             sender="Alice", date="2024-03-01 10:00:00+00:00",
             body="Revenue is up."),
    ])

    out = ensure_unpacked(pst_path)
    txt = next((out / "Top" / "Inbox").glob("msg-0001_*.txt"))
    content = txt.read_text(encoding="utf-8")
    assert "From: Alice" in content
    assert "To: " in content
    assert "Cc: " in content
    assert "Subject: Quarterly numbers" in content
    assert "Date: 2024-03-01 10:00:00+00:00" in content
    assert "Message-ID: " in content
    assert "Folder: Top/Inbox" in content
    assert "Revenue is up." in content


def test_unpack_with_target_writes_into_explicit_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """With target=<dir>, outputs land in that directory instead of next
    to the source. The source folder is untouched."""
    from pst_handling.unpack import ensure_unpacked

    src = tmp_path / "src"
    dst = tmp_path / "dst" / "foo_unpacked"
    src.mkdir()
    pst_path = src / "foo.pst"
    _touch_pst(pst_path)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top", subject="hi", body="hello",
             attachments=(_att("a.txt", b"AAA"),)),
    ])

    out = ensure_unpacked(pst_path, target=dst)
    assert out == dst
    assert (dst / "Top").is_dir()
    assert (dst / "Top" / "attachments" / "msg-0001" / "a.txt").read_bytes() \
        == b"AAA"
    # Source folder is untouched.
    assert not (src / "foo_unpacked").exists()


def test_unpack_handles_messages_at_root_with_empty_folder_path(
    tmp_path: Path, monkeypatch
) -> None:
    """A message whose folder path is empty (rare but happens on
    damaged archives or root-level entries) lands directly under
    ``<stem>_unpacked/`` instead of a subdir."""
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    _install_iter(monkeypatch, [
        _msg(folder_path="", subject="root mail", body="b"),
    ])

    out = ensure_unpacked(pst_path)
    assert any(out.glob("msg-0001_*.txt")), list(out.iterdir())


# --- Idempotency ------------------------------------------------------------


def test_unpack_is_idempotent_when_unpacked_dir_fresher(
    tmp_path: Path, monkeypatch
) -> None:
    from pst_handling.unpack import ensure_unpacked
    from pst_handling import unpack

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top/Inbox", subject="A", body="first"),
    ])

    ensure_unpacked(pst_path)
    out_dir = tmp_path / "foo_unpacked"
    assert out_dir.is_dir()

    # Force the unpacked dir to be unambiguously newer than the .pst.
    _bump_mtime(out_dir)

    # Replace the iterator with a tripwire — proving the skip branch ran.
    def tripwire(_path):
        raise AssertionError(
            "iter_pst_messages should not run on idempotent skip"
        )
        yield  # pragma: no cover - never reached
    monkeypatch.setattr(unpack, "iter_pst_messages", tripwire)

    ensure_unpacked(pst_path)  # must not raise


def test_unpack_clobbers_when_pst_newer(
    tmp_path: Path, monkeypatch
) -> None:
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)

    state = {"body": "first", "attachment_name": "old.pdf",
             "attachment_data": b"OLD"}

    def iter_factory(_path):
        return [_msg(folder_path="Top", body=state["body"],
                     attachments=(_att(state["attachment_name"],
                                       state["attachment_data"]),))]
    _install_iter(monkeypatch, iter_factory)

    ensure_unpacked(pst_path)
    out_dir = tmp_path / "foo_unpacked"
    old_att = out_dir / "Top" / "attachments" / "msg-0001" / "old.pdf"
    assert old_att.read_bytes() == b"OLD"

    state["body"] = "second"
    state["attachment_name"] = "new.pdf"
    state["attachment_data"] = b"NEW"
    _bump_mtime(pst_path)

    ensure_unpacked(pst_path)
    new_att = out_dir / "Top" / "attachments" / "msg-0001" / "new.pdf"
    assert new_att.read_bytes() == b"NEW"
    # Old attachment was cleared by the clobber.
    assert not old_att.exists()
    txt = next((out_dir / "Top").glob("msg-0001_*.txt"))
    assert "second" in txt.read_text(encoding="utf-8")


# --- Per-attachment size cap ------------------------------------------------


def test_oversized_attachment_is_skipped_with_note(
    tmp_path: Path, monkeypatch
) -> None:
    """When an attachment's size exceeds MAX_FILE_SIZE_BYTES the
    attachment is not written and a note appears in the .txt instead
    — same OOM-guard posture as the prior inline extractor."""
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    monkeypatch.setattr("server.indexer.MAX_FILE_SIZE_BYTES", 16)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top", subject="big",
             attachments=(_att("big.txt", data=b"", size=1024),)),
    ])

    out = ensure_unpacked(pst_path)
    txt = next((out / "Top").glob("msg-0001_*.txt"))
    content = txt.read_text(encoding="utf-8")
    assert "big.txt" in content and "skipped" in content
    # The attachment file itself was not materialised.
    attachments_dir = out / "Top" / "attachments" / "msg-0001"
    assert not (attachments_dir / "big.txt").exists()


def test_attachment_size_cap_disabled_still_writes(
    tmp_path: Path, monkeypatch
) -> None:
    """MAX_FILE_SIZE_BYTES==0 disables the cap, so a 'large' attachment
    is still written."""
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    monkeypatch.setattr("server.indexer.MAX_FILE_SIZE_BYTES", 0)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top",
             attachments=(_att("big.txt", data=b"hello", size=10**9),)),
    ])

    out = ensure_unpacked(pst_path)
    attachments_dir = out / "Top" / "attachments" / "msg-0001"
    assert (attachments_dir / "big.txt").read_bytes() == b"hello"


# --- Error handling ---------------------------------------------------------


def test_unpack_writes_error_marker_for_broken_message(
    tmp_path: Path, monkeypatch
) -> None:
    """A ParsedPstMessage with ``error`` set produces a
    ``msg-NNNN_error.txt`` and the rest of the archive continues."""
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top/Inbox", error="ValueError: corrupt MAPI prop"),
        _msg(folder_path="Top/Inbox", subject="good", body="ok"),
    ])

    out = ensure_unpacked(pst_path)
    inbox = out / "Top" / "Inbox"
    err_files = list(inbox.glob("msg-0001_error.txt"))
    assert err_files, list(inbox.iterdir())
    assert "ValueError" in err_files[0].read_text(encoding="utf-8")
    # The good message still landed.
    assert any(inbox.glob("msg-0002_*.txt")), list(inbox.iterdir())


# --- Attachment safety ------------------------------------------------------


def test_attachment_path_traversal_is_neutralised(
    tmp_path: Path, monkeypatch
) -> None:
    """A malicious .pst whose attachment name tries to escape via
    ../../ lands inside the per-message attachments dir regardless."""
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    _install_iter(monkeypatch, [
        _msg(folder_path="Top",
             attachments=(_att("../../escape.txt", b"PAYLOAD"),)),
    ])

    out = ensure_unpacked(pst_path)
    attachments_dir = out / "Top" / "attachments" / "msg-0001"
    # The name was sanitised to the basename only.
    assert (attachments_dir / "escape.txt").is_file()
    # Nothing landed outside the per-message attachments dir.
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path.parent / "escape.txt").exists()


def test_pst_folder_name_with_slash_is_sanitised(
    tmp_path: Path, monkeypatch
) -> None:
    """A PST folder named ``a/b`` (single segment containing a slash) is
    flattened to one directory, not two — pypff can surface unsanitised
    names on damaged archives."""
    from pst_handling.unpack import ensure_unpacked

    pst_path = tmp_path / "foo.pst"
    _touch_pst(pst_path)
    # Build a single-segment folder name by passing it as the whole path
    # — _walk_pst_folder joins with '/', so we test the sanitiser via
    # the inner path joining instead. We simulate by passing a leaf
    # that already contains a slash inside one segment.
    _install_iter(monkeypatch, [
        # folder_path uses '/' as the path separator — to exercise the
        # per-segment sanitiser we'd need a single segment with a ':' or
        # similar. Use a colon (illegal in Windows paths, sometimes
        # present on macOS-imported PSTs).
        _msg(folder_path="Top/weird:name"),
    ])
    out = ensure_unpacked(pst_path)
    weird = out / "Top" / "weird-name"
    assert weird.is_dir(), list((out / "Top").iterdir())


# --- pst_handling.messages helpers ------------------------------------------


def test_body_prefers_plain_text():
    from pst_handling.messages import _pst_body

    class M:
        def get_plain_text_body(self):
            return "Plain text body."
        def get_html_body(self):
            return "<p>HTML version</p>"
        def get_rtf_body(self):
            return r"{\rtf1\ansi RTF\par}"

    assert _pst_body(M()) == "Plain text body."


def test_body_falls_back_to_html_when_no_plain_text():
    from pst_handling.messages import _pst_body

    class M:
        def get_plain_text_body(self):
            return None
        def get_html_body(self):
            return "<html><body><p>HTML only body</p></body></html>"
        def get_rtf_body(self):
            return None

    body = _pst_body(M())
    assert "HTML only body" in body
    assert "<p>" not in body


def test_body_falls_back_to_rtf_as_last_resort():
    from pst_handling.messages import _pst_body

    class M:
        def get_plain_text_body(self):
            return None
        def get_html_body(self):
            return None
        def get_rtf_body(self):
            return r"{\rtf1\ansi \b Revenue report\b0\par}"

    assert "Revenue report" in _pst_body(M())


def test_body_empty_when_all_fields_missing():
    from pst_handling.messages import _pst_body

    class M:
        def get_plain_text_body(self):
            return None
        def get_html_body(self):
            return None
        def get_rtf_body(self):
            return None

    assert _pst_body(M()) == ""


def test_is_mail_item_keeps_mail_and_drops_others():
    from pst_handling.messages import _is_mail_item, _PR_MESSAGE_CLASS

    class FakeEntry:
        def __init__(self, entry_type, value):
            self.entry_type = entry_type
            self._value = value
        def get_data_as_string(self):
            return self._value

    class FakeRecordSet:
        def __init__(self, entries):
            self.entries = entries

    class M:
        def __init__(self, msg_class):
            entries = [FakeEntry(_PR_MESSAGE_CLASS, msg_class)] if msg_class else []
            self.record_sets = [FakeRecordSet(entries)]

    assert _is_mail_item(M("IPM.Note")) is True
    assert _is_mail_item(M("IPM.Note.SMIME")) is True
    assert _is_mail_item(M("IPM.Appointment")) is False
    assert _is_mail_item(M("IPM.Contact")) is False
    assert _is_mail_item(M("IPM.Task")) is False
    # Unreadable class — kept so mail is never lost.
    assert _is_mail_item(M(None)) is True


def test_iter_pst_messages_filters_non_mail_items_via_real_walk(monkeypatch):
    """End-to-end-ish: the iterator's mail filter drops non-IPM.Note
    entries even when pypff returns them."""
    from pst_handling.messages import iter_pst_messages, _PR_MESSAGE_CLASS

    class FakeEntry:
        def __init__(self, entry_type, value):
            self.entry_type = entry_type
            self._value = value
        def get_data_as_string(self):
            return self._value

    class FakeRecordSet:
        def __init__(self, entries):
            self.entries = entries

    class FakeAttachment:
        pass

    class FakeMessage:
        def __init__(self, subject, message_class, body):
            self._subject = subject
            self._body = body
            entries = [FakeEntry(_PR_MESSAGE_CLASS, message_class)]
            self.record_sets = [FakeRecordSet(entries)]
        def get_subject(self): return self._subject
        def get_sender_name(self): return "alice"
        def get_client_submit_time(self): return "2024-03-01"
        def get_delivery_time(self): return None
        def get_plain_text_body(self): return self._body
        def get_html_body(self): return None
        def get_rtf_body(self): return None
        def get_number_of_attachments(self): return 0
        def get_attachment(self, i): raise IndexError

    class FakeFolder:
        def __init__(self, name, messages, subs=()):
            self._name = name
            self._messages = list(messages)
            self._subs = list(subs)
        def get_name(self): return self._name
        @property
        def number_of_sub_messages(self): return len(self._messages)
        def get_sub_message(self, i): return self._messages[i]
        @property
        def number_of_sub_folders(self): return len(self._subs)
        def get_sub_folder(self, i): return self._subs[i]

    class FakeFile:
        def __init__(self, root): self._root = root
        def open(self, path): pass
        def get_root_folder(self): return self._root
        def close(self): pass

    inbox = FakeFolder("Inbox", messages=[
        FakeMessage("A meeting", "IPM.Appointment", "skip me"),
        FakeMessage("A real mail", "IPM.Note", "hello"),
    ])
    root = FakeFolder("Top", messages=[], subs=[inbox])

    import pst_handling.messages as msgs
    class FakePypff:
        def file(self):
            return FakeFile(root)
    monkeypatch.setitem(__import__("sys").modules, "pypff", FakePypff())

    out = list(iter_pst_messages(Path("/dev/null")))
    assert len(out) == 1
    assert out[0].subject == "A real mail"
    # The root folder name is not included in folder_path — the walker
    # starts paths at the root's children, matching the prior
    # _walk_pst_folder convention.
    assert out[0].folder_path == "Inbox"


# --- Negative cases on parsers.py dispatch ----------------------------------


def test_pst_is_no_longer_in_supported_extensions():
    from server.parsers import SUPPORTED_EXTENSIONS
    assert ".pst" not in SUPPORTED_EXTENSIONS


def test_extract_text_rejects_pst():
    from server.parsers import extract_text
    with pytest.raises(ValueError, match=r"Unsupported file type: \.pst"):
        extract_text(Path("/some/archive.pst"))


# --- Opt-in: real .pst fixture ----------------------------------------------


_FIXTURE = Path(__file__).parent / "fixtures" / "sample.pst"


@pytest.mark.skipif(
    not _FIXTURE.exists(),
    reason="no real .pst fixture at tests/fixtures/sample.pst",
)
def test_unpack_real_pst_fixture(tmp_path: Path):
    """End-to-end against a real archive. To enable: drop a small
    Outlook .pst at tests/fixtures/sample.pst and install the 'pst'
    extra. The unpack produces a non-empty tree of per-message .txt
    files."""
    pytest.importorskip("pypff")
    from pst_handling.unpack import ensure_unpacked

    pst_copy = tmp_path / "sample.pst"
    pst_copy.write_bytes(_FIXTURE.read_bytes())

    out = ensure_unpacked(pst_copy)
    assert out.is_dir()
    txt_files = list(out.rglob("*.txt"))
    assert txt_files, "expected at least one .txt to be written"
    for txt in txt_files:
        assert txt.read_text(encoding="utf-8", errors="replace").strip()
