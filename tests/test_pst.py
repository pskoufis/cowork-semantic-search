"""Tests for .pst (Outlook archive) extraction.

pypff is read-only — a .pst cannot be generated programmatically — so the
extraction logic is exercised with duck-typed fakes that mimic the pypff
object shapes (file / folder / message / attachment / record_set /
record_entry). One opt-in test runs against a real fixture if one is present.
"""

from pathlib import Path

import pytest

from server import parsers as p
from server.parsers import extract_text, SUPPORTED_EXTENSIONS

# MAPI property entry types, mirrored from server.parsers.
_PR_MESSAGE_CLASS = 0x001A
_PR_ATTACH_LONG_FILENAME = 0x3707


# --- Fake pypff objects -----------------------------------------------------

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
    def __init__(self, filename, data=b""):
        self._filename = filename
        self._data = data
        self._offset = 0
        entries = []
        if filename:
            entries.append(FakeEntry(_PR_ATTACH_LONG_FILENAME, filename))
        self.record_sets = [FakeRecordSet(entries)]

    def get_size(self):
        return len(self._data)

    def read_buffer(self, size):
        block = self._data[self._offset:self._offset + size]
        self._offset += len(block)
        return block


class FakeMessage:
    def __init__(self, subject="(no subject)", sender="", message_class="IPM.Note",
                 plain_body=None, html_body=None, rtf_body=None,
                 submit_time="2024-03-01", attachments=None):
        self._subject = subject
        self._sender = sender
        self._plain = plain_body
        self._html = html_body
        self._rtf = rtf_body
        self._submit = submit_time
        self._attachments = attachments or []
        entries = []
        if message_class is not None:
            entries.append(FakeEntry(_PR_MESSAGE_CLASS, message_class))
        self.record_sets = [FakeRecordSet(entries)]

    def get_subject(self):
        return self._subject

    def get_sender_name(self):
        return self._sender

    def get_client_submit_time(self):
        return self._submit

    def get_delivery_time(self):
        return None

    def get_plain_text_body(self):
        return self._plain

    def get_html_body(self):
        return self._html

    def get_rtf_body(self):
        return self._rtf

    def get_number_of_attachments(self):
        return len(self._attachments)

    def get_attachment(self, i):
        return self._attachments[i]


class FakeFolder:
    def __init__(self, name="", messages=None, sub_folders=None):
        self._name = name
        self._messages = messages or []
        self._sub_folders = sub_folders or []

    def get_name(self):
        return self._name

    @property
    def number_of_sub_messages(self):
        return len(self._messages)

    def get_sub_message(self, i):
        return self._messages[i]

    @property
    def number_of_sub_folders(self):
        return len(self._sub_folders)

    def get_sub_folder(self, i):
        return self._sub_folders[i]


# --- Body extraction --------------------------------------------------------

def test_body_prefers_plain_text():
    msg = FakeMessage(plain_body="The plain text body.",
                      html_body="<p>HTML version</p>")
    assert p._pst_body(msg) == "The plain text body."


def test_body_falls_back_to_html_when_no_plain_text():
    """Modern Outlook mail is frequently HTML-only — it must not be lost."""
    msg = FakeMessage(plain_body=None,
                      html_body="<html><body><p>HTML only body</p></body></html>")
    body = p._pst_body(msg)
    assert "HTML only body" in body
    assert "<p>" not in body  # tags stripped


def test_body_falls_back_to_rtf_as_last_resort():
    msg = FakeMessage(plain_body=None, html_body=None,
                      rtf_body=r"{\rtf1\ansi \b Revenue report\b0\par}")
    assert "Revenue report" in p._pst_body(msg)


def test_body_empty_when_all_fields_missing():
    assert p._pst_body(FakeMessage(plain_body=None)) == ""


# --- Message -> part --------------------------------------------------------

def test_message_to_part_folds_header_into_text():
    msg = FakeMessage(subject="Quarterly numbers", sender="Alice",
                      plain_body="Revenue is up.")
    part = p._pst_message_to_part(msg, "Top/Inbox")
    text = part["text"]
    assert "Subject: Quarterly numbers" in text
    assert "From: Alice" in text
    assert "Folder: Top/Inbox" in text
    assert "Revenue is up." in text
    assert part["metadata"]["pst_folder"] == "Top/Inbox"


def test_message_with_no_body_still_yields_header_text():
    """A bodyless message must not become an empty part — the header survives,
    so it is never dropped by the chunker's empty-text skip."""
    part = p._pst_message_to_part(FakeMessage(plain_body=None), "Inbox")
    assert part["text"].strip()
    assert "Subject:" in part["text"]


# --- Mail-only filter -------------------------------------------------------

def test_is_mail_item_keeps_mail():
    assert p._is_mail_item(FakeMessage(message_class="IPM.Note")) is True
    assert p._is_mail_item(FakeMessage(message_class="IPM.Note.SMIME")) is True


def test_is_mail_item_drops_calendar_and_contacts():
    assert p._is_mail_item(FakeMessage(message_class="IPM.Appointment")) is False
    assert p._is_mail_item(FakeMessage(message_class="IPM.Contact")) is False
    assert p._is_mail_item(FakeMessage(message_class="IPM.Task")) is False


def test_is_mail_item_keeps_item_with_unreadable_class():
    """When the class cannot be read, the item is kept so mail is never lost."""
    assert p._is_mail_item(FakeMessage(message_class=None)) is True


# --- Folder walk ------------------------------------------------------------

def test_walk_collects_messages_and_recurses_subfolders():
    inbox = FakeFolder("Inbox", messages=[
        FakeMessage(subject="Mail one", plain_body="body one"),
        FakeMessage(subject="Mail two", plain_body="body two"),
    ])
    root = FakeFolder("Top", messages=[], sub_folders=[inbox])
    parts = []
    p._walk_pst_folder(root, "", parts)
    assert len(parts) == 2
    assert all("Folder: Inbox" in part["text"] for part in parts)


def test_walk_skips_non_mail_items():
    folder = FakeFolder("Calendar", messages=[
        FakeMessage(subject="A meeting", message_class="IPM.Appointment"),
        FakeMessage(subject="A real mail", message_class="IPM.Note",
                    plain_body="hello"),
    ])
    parts = []
    p._walk_pst_folder(folder, "", parts)
    assert len(parts) == 1
    assert "A real mail" in parts[0]["text"]


# --- Attachments ------------------------------------------------------------

def test_attachment_supported_type_is_extracted():
    msg = FakeMessage(plain_body="See attached.", attachments=[
        FakeAttachment("contract.txt", b"The signed contract terms."),
    ])
    part = p._pst_message_to_part(msg, "Inbox")
    assert "--- Attachment: contract.txt ---" in part["text"]
    assert "signed contract terms" in part["text"]


def test_attachment_unsupported_type_is_noted_not_extracted():
    text = p._pst_attachments_text(FakeMessage(attachments=[
        FakeAttachment("photo.png", b"\x89PNG binary data"),
    ]))
    assert text == "[Attachment: photo.png — not indexed]"


def test_attachment_nested_pst_is_not_recursed_into():
    """A .pst attached inside a .pst must not be recursed — infinite loop."""
    text = p._pst_attachments_text(FakeMessage(attachments=[
        FakeAttachment("nested.pst", b"PST bytes"),
    ]))
    assert text == "[Attachment: nested.pst — not indexed]"


def test_attachment_without_filename_is_skipped_silently():
    text = p._pst_attachments_text(FakeMessage(attachments=[
        FakeAttachment(None, b"embedded message body"),
    ]))
    assert text == ""


def test_attachment_over_size_cap_is_skipped(monkeypatch):
    """The whole-file cap cannot see inside a .pst, so the cap is re-applied
    per attachment as the OOM guard."""
    monkeypatch.setattr("server.indexer.MAX_FILE_SIZE_BYTES", 16)
    text = p._pst_attachments_text(FakeMessage(attachments=[
        FakeAttachment("big.txt", b"x" * 1024),
    ]))
    assert "big.txt" in text and "skipped" in text


# --- HTML / RTF helpers -----------------------------------------------------

def test_strip_html_removes_tags_and_scripts():
    html = "<html><body><p>Visible text</p><script>secret()</script></body></html>"
    out = p._strip_html(html)
    assert "Visible text" in out
    assert "secret" not in out
    assert "<" not in out


def test_strip_rtf_returns_readable_text():
    out = p._strip_rtf(r"{\rtf1\ansi\deff0 \b Hello\b0  world\par}")
    assert "Hello" in out and "world" in out
    assert "\\rtf1" not in out


# --- Dispatch & registration ------------------------------------------------

def test_pst_is_a_supported_extension():
    assert ".pst" in SUPPORTED_EXTENSIONS


def test_extract_text_dispatches_pst(monkeypatch):
    sentinel = [{"text": "routed", "metadata": {}}]
    monkeypatch.setattr("server.parsers._extract_pst", lambda path: sentinel)
    assert extract_text(Path("/some/archive.pst")) is sentinel


def test_extract_pst_missing_pypff_raises_helpful_error(monkeypatch):
    """When libpff-python is absent, the error names the fix."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypff":
            raise ImportError("No module named 'pypff'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"\[pst\]"):
        p._extract_pst(Path("/some/archive.pst"))


# --- Opt-in: real .pst fixture ----------------------------------------------

_FIXTURE = Path(__file__).parent / "fixtures" / "sample.pst"


@pytest.mark.skipif(
    not _FIXTURE.exists(),
    reason="no real .pst fixture at tests/fixtures/sample.pst (see test_pst.py)",
)
def test_extract_real_pst_fixture():
    """End-to-end against a real archive. To enable: drop a small Outlook
    .pst at tests/fixtures/sample.pst (e.g. exported from Outlook with a few
    test messages, or a permissively-licensed sample from the libyal/libpff
    test data) and install the 'pst' extra."""
    pytest.importorskip("pypff")
    parts = extract_text(_FIXTURE)
    assert isinstance(parts, list)
    for part in parts:
        assert "text" in part and "metadata" in part
        assert part["text"].strip()
