"""Reader for Outlook .pst archives via libpff-python (pypff).

Exposes ``iter_pst_messages(pst_path)`` plus the ``ParsedPstMessage`` /
``ParsedPstAttachment`` dataclasses. The reader is the only place that
touches ``pypff``; ``pst_handling.unpack`` is pure file-layout logic so
its tests can monkeypatch this boundary out without crafting binary
.pst fixtures.

A .pst is a container of folders, each containing mail and other items
(calendar, contacts, tasks, notes). Only mail items (IPM.Note*) are
yielded — non-mail item classes are filtered out, matching the prior
inline extractor's behavior.

Attachments expose a ``write_to(target)`` streaming method instead of
eager bytes, because Outlook attachments can be hundreds of MB and
pypff already reads them incrementally. The unpacker applies a
per-attachment size cap before calling ``write_to``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Tuple

# MAPI property entry types reachable only via record-set scanning on a
# pypff item — the high-level getters don't expose these.
_PR_MESSAGE_CLASS = 0x001A          # PidTagMessageClass
_PR_ATTACH_LONG_FILENAME = 0x3707   # PidTagAttachLongFilename
_PR_ATTACH_FILENAME = 0x3704        # PidTagAttachFilename (8.3 short name)

# Attachment streaming chunk size — matches the prior inline extractor's
# 1 MB step. pypff returns at most this many bytes per ``read_buffer``
# call regardless of the requested size, so a larger value buys nothing.
_READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ParsedPstAttachment:
    filename: str               # raw name from PR_ATTACH_LONG_FILENAME / _FILENAME
    size: int                   # logical size in bytes, from pypff
    write_to: Callable[[Path], None]   # streams the bytes to ``target``


@dataclass(frozen=True)
class ParsedPstMessage:
    folder_path: str            # slash-joined PST folder path, e.g. "Top/Inbox"
    subject: str
    sender: str
    date: str
    body: str
    attachments: Tuple[ParsedPstAttachment, ...]
    error: str | None = None    # populated when this single message failed to parse


def iter_pst_messages(pst_path: Path) -> Iterator[ParsedPstMessage]:
    """Yield each mail message from a .pst archive.

    The ``pypff`` dependency is imported lazily so a repo install without
    the ``[pst]`` extra can still import this module (the failure surfaces
    only when an actual ``.pst`` is unpacked).

    A per-message parse failure yields a ``ParsedPstMessage`` with
    ``error`` set instead of aborting the iteration. Calendar / contact /
    task / note items are filtered out.
    """
    try:
        import pypff
    except ImportError as exc:
        raise ImportError(
            "Reading .pst archives requires the 'pst' extra: "
            "pip install 'cowork-semantic-search[pst]'"
        ) from exc

    pst = pypff.file()
    try:
        pst.open(str(pst_path))
        root = pst.get_root_folder()
        if root is not None:
            yield from _walk_pst_folder(root, "")
    finally:
        try:
            pst.close()
        except Exception:
            pass


def _walk_pst_folder(folder, folder_path: str) -> Iterator[ParsedPstMessage]:
    """Yield each mail message in this folder, then recurse into subfolders."""
    count = _pst_safe(lambda: folder.number_of_sub_messages) or 0
    for i in range(count):
        try:
            message = folder.get_sub_message(i)
        except Exception as exc:
            yield ParsedPstMessage(
                folder_path=folder_path, subject="", sender="", date="",
                body="", attachments=(),
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        if not _is_mail_item(message):
            continue
        try:
            yield _build_parsed_message(message, folder_path)
        except Exception as exc:
            yield ParsedPstMessage(
                folder_path=folder_path, subject="", sender="", date="",
                body="", attachments=(),
                error=f"{type(exc).__name__}: {exc}",
            )

    sub_count = _pst_safe(lambda: folder.number_of_sub_folders) or 0
    for i in range(sub_count):
        sub = folder.get_sub_folder(i)
        sub_name = _pst_decode(_pst_safe(sub.get_name)) or "_unnamed"
        child_path = f"{folder_path}/{sub_name}" if folder_path else sub_name
        yield from _walk_pst_folder(sub, child_path)


def _build_parsed_message(message, folder_path: str) -> ParsedPstMessage:
    subject = _pst_decode(_pst_safe(message.get_subject)) or ""
    sender = _pst_decode(_pst_safe(message.get_sender_name)) or ""
    sent = (_pst_safe(message.get_client_submit_time)
            or _pst_safe(message.get_delivery_time))
    date = "" if sent is None else str(sent)
    body = _pst_body(message)

    count = _pst_safe(message.get_number_of_attachments) or 0
    attachments: list[ParsedPstAttachment] = []
    for i in range(count):
        try:
            attachment = message.get_attachment(i)
        except Exception:
            continue
        parsed = _build_parsed_attachment(attachment)
        if parsed is not None:
            attachments.append(parsed)

    return ParsedPstMessage(
        folder_path=folder_path,
        subject=subject,
        sender=sender,
        date=date,
        body=body,
        attachments=tuple(attachments),
    )


def _build_parsed_attachment(attachment) -> ParsedPstAttachment | None:
    name = (_pst_prop_string(attachment, _PR_ATTACH_LONG_FILENAME)
            or _pst_prop_string(attachment, _PR_ATTACH_FILENAME))
    if not name:
        return None
    name = name.strip()
    size = _pst_safe(attachment.get_size) or 0

    def write_to(target: Path) -> None:
        with open(target, "wb") as fh:
            remaining = size
            while remaining > 0:
                block = attachment.read_buffer(min(remaining, _READ_CHUNK_BYTES))
                if not block:
                    break
                fh.write(block)
                remaining -= len(block)

    return ParsedPstAttachment(filename=name, size=size, write_to=write_to)


def _pst_body(message) -> str:
    """Return the message body — plain text preferred, else HTML, else RTF.

    Modern Outlook mail is frequently HTML-only with no plain-text body,
    so a plain-text-only read would silently lose those messages' content.
    """
    plain = _pst_decode(_pst_safe(message.get_plain_text_body))
    if plain.strip():
        return plain
    html = _pst_decode(_pst_safe(message.get_html_body))
    if html.strip():
        from server.parsers import _strip_html
        return _strip_html(html)
    rtf = _pst_decode(_pst_safe(message.get_rtf_body))
    if rtf.strip():
        from server.parsers import _strip_rtf
        return _strip_rtf(rtf)
    return ""


def _is_mail_item(message) -> bool:
    """True for ordinary mail.

    A PST folder also stores calendar items, contacts, tasks and notes
    as 'messages', distinguished by a non-mail message class
    (IPM.Appointment, IPM.Contact, ...). When the class cannot be read
    the item is kept, so genuine mail is never dropped.
    """
    message_class = _pst_prop_string(message, _PR_MESSAGE_CLASS)
    if not message_class:
        return True
    return message_class.upper().startswith("IPM.NOTE")


def _pst_prop_string(item, entry_type: int) -> str | None:
    """Read a MAPI property as a string from a pypff item's record sets.

    pypff surfaces a fixed set of fields directly; everything else
    (message class, attachment filename) is only reachable by scanning
    record entries for a matching entry type.
    """
    try:
        for record_set in item.record_sets:
            for entry in record_set.entries:
                if entry.entry_type == entry_type:
                    value = entry.get_data_as_string()
                    if value:
                        return value
    except Exception:
        return None
    return None


def _pst_safe(getter):
    """Call a pypff getter, returning None instead of raising when the
    underlying property is absent."""
    try:
        return getter()
    except Exception:
        return None


def _pst_decode(raw) -> str:
    """Coerce a pypff value (bytes, str, or None) to a clean string."""
    if raw is None:
        return ""
    text = (raw.decode("utf-8", errors="replace")
            if isinstance(raw, bytes) else str(raw))
    return text.replace("\x00", "")
