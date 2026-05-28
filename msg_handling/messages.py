"""Reader for Outlook .msg files via the extract-msg library.

Exposes a single ``read_message(path)`` function plus ``ParsedMessage`` /
``ParsedAttachment`` dataclasses. The reader is the only place that touches
``extract_msg``; ``msg_handling.unpack`` is pure file-layout logic so its
tests can swap this boundary out for a fake without crafting binary .msg
fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ParsedAttachment:
    filename: str          # raw filename as reported by extract-msg
    data: bytes
    is_msg: bool = False   # True when this attachment is itself a .msg


@dataclass(frozen=True)
class ParsedMessage:
    from_addr: str
    to: str
    cc: str
    subject: str
    date: str
    message_id: str
    body: str
    attachments: Tuple[ParsedAttachment, ...]


def read_message(msg_path: Path) -> ParsedMessage:
    """Parse a .msg file into a ``ParsedMessage``.

    The ``extract_msg`` dependency is imported lazily — missing-package errors
    surface with a friendly install hint, mirroring the .pst path in
    ``server.parsers._extract_pst``.
    """
    try:
        import extract_msg
        from extract_msg.enums import AttachmentType
    except ImportError as exc:
        raise ImportError(
            "Reading .msg files requires the 'msg' extra: "
            "pip install 'cowork-semantic-search[msg]'"
        ) from exc

    attachments: list[ParsedAttachment] = []
    with extract_msg.openMsg(str(msg_path)) as msg:
        for att in msg.attachments:
            parsed = _parse_attachment(att, AttachmentType)
            if parsed is not None:
                attachments.append(parsed)

        return ParsedMessage(
            from_addr=_coerce_str(getattr(msg, "sender", None)),
            to=_coerce_str(getattr(msg, "to", None)),
            cc=_coerce_str(getattr(msg, "cc", None)),
            subject=_coerce_str(getattr(msg, "subject", None)),
            date=_extract_date(msg),
            message_id=_coerce_str(getattr(msg, "messageId", None)),
            body=_select_body(msg),
            attachments=tuple(attachments),
        )


def _extract_date(msg) -> str:
    """RFC2822-format the best-available timestamp on the message.

    .msg files don't always carry a ``clientSubmitTime`` (the canonical
    "when sent" timestamp that ``msg.date`` reads). When it's missing we
    fall back through ``receivedTime`` and the raw MAPI creation and
    last-modification times, in that order. Returns "" when no usable
    timestamp is available so the Date header line stays present-but-empty
    rather than disappearing.
    """
    for candidate in (
        getattr(msg, "date", None),
        getattr(msg, "receivedTime", None),
        _prop_datetime(msg, "30070040"),  # PR_CREATION_TIME
        _prop_datetime(msg, "30080040"),  # PR_LAST_MODIFICATION_TIME
    ):
        if isinstance(candidate, datetime):
            return format_datetime(candidate)
    return ""


def _prop_datetime(msg, tag: str) -> datetime | None:
    try:
        prop = msg.props.get(tag)
    except Exception:
        return None
    if prop is None:
        return None
    value = getattr(prop, "value", None)
    return value if isinstance(value, datetime) else None


def _select_body(msg) -> str:
    """Plain-text body preferred; fall back to HTML-stripped body."""
    plain = _coerce_str(getattr(msg, "body", None))
    if plain.strip():
        return plain
    html = getattr(msg, "htmlBody", None)
    if html:
        # _strip_html lives in server.parsers; reused for parity with the
        # .pst extractor.
        from server.parsers import _strip_html
        if isinstance(html, (bytes, bytearray)):
            html = html.decode("utf-8", errors="replace")
        return _strip_html(html)
    return ""


def _parse_attachment(att, AttachmentType) -> ParsedAttachment | None:
    name = _attachment_name(att)
    if not name:
        return None

    # Don't rely solely on att.type — for SIGNED vs SIGNED_EMBEDDED the
    # library distinguishes by whether the bytes look like an OLE2
    # compound document, so the most reliable embedded-MSG test is
    # "is .data a bytes-like object?". A normal Attachment with
    # type=MSG also has an MSGFile in .data.
    raw = att.data
    if isinstance(raw, (bytes, bytearray)):
        return ParsedAttachment(filename=name, data=bytes(raw), is_msg=False)

    # Non-bytes .data means an embedded MSGFile (or signed-embedded).
    # Serialise it back to .msg bytes so the recursive unpacker can
    # treat it as an on-disk file.
    data = _serialize_embedded_msg(raw)
    if data is None:
        return None
    return ParsedAttachment(filename=name, data=data, is_msg=True)


def _attachment_name(att) -> str:
    """Best-available attachment filename — long name first, then short, then
    the generic ``name`` field."""
    for candidate in (
        getattr(att, "longFilename", None),
        getattr(att, "shortFilename", None),
        getattr(att, "name", None),
    ):
        if candidate:
            return _coerce_str(candidate)
    return ""


def _serialize_embedded_msg(embedded) -> bytes | None:
    """Re-serialize an embedded MSGFile to .msg bytes so the recursive
    unpacker can treat it as an on-disk file.

    ``MSGFile.__bytes__`` calls ``exportBytes()`` — that's the in-memory
    round-trip API. If the library can't round-trip this particular
    embedded item (rare formats), return None — the attachment is
    dropped without aborting the outer parse.
    """
    if embedded is None:
        return None
    try:
        return bytes(embedded)
    except Exception:
        return None


def _coerce_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)
