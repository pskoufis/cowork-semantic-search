"""Shared mbox parsed-message iterator.

Yields ParsedMessage records — one per mbox entry — with full threading
headers (Message-ID, In-Reply-To, References) and attachment bytes. The
exact same iterator backs both the on-disk unpacker
(`mbox_handling.unpack`) and the in-process indexer extractor
(`server.parsers._extract_mbox`).

Stdlib only. Per-message errors are isolated: a broken message yields a
ParsedMessage with ``error`` set, never aborts the iterator.
"""

from __future__ import annotations

import mailbox
import re
from dataclasses import dataclass, field
from email.header import decode_header, make_header
from email.message import Message
from email.utils import getaddresses
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Attachment:
    filename: str
    content_type: str
    content_id: str | None
    data: bytes


@dataclass(frozen=True)
class ParsedMessage:
    index: int
    message_id: str | None
    in_reply_to: str | None
    references: tuple[str, ...]
    from_addr: str | None
    to: tuple[str, ...]
    cc: tuple[str, ...]
    date: str | None
    subject: str | None
    body: str
    attachments: tuple[Attachment, ...]
    error: str | None = None


def iter_messages(mbox_path: Path) -> Iterator[ParsedMessage]:
    """Yield one ParsedMessage per entry in the mbox at ``mbox_path``."""
    mbox = mailbox.mbox(str(mbox_path), create=False)
    for idx, msg in enumerate(mbox, start=1):
        yield _parse_one(msg, idx)


def _parse_one(msg: Message, idx: int) -> ParsedMessage:
    # Headers are extracted up front so a body-extraction failure still
    # produces a partially-populated record with the from/subject/date the
    # caller needs for error reporting.
    message_id = _decode_header_value(msg.get("Message-ID"))
    in_reply_to = _decode_header_value(msg.get("In-Reply-To"))
    references = _parse_references(msg.get("References"))
    from_addr = _decode_header_value(msg.get("From"))
    to = _address_list(msg.get("To"))
    cc = _address_list(msg.get("Cc"))
    date = _decode_header_value(msg.get("Date"))
    subject = _decode_header_value(msg.get("Subject"))

    try:
        body = _extract_body(msg)
        attachments = tuple(_extract_attachments(msg))
        error = None
    except Exception as exc:
        body = ""
        attachments = ()
        error = f"{type(exc).__name__}: {exc}"

    return ParsedMessage(
        index=idx,
        message_id=message_id,
        in_reply_to=in_reply_to,
        references=references,
        from_addr=from_addr,
        to=to,
        cc=cc,
        date=date,
        subject=subject,
        body=body,
        attachments=attachments,
        error=error,
    )


def _extract_body(msg: Message) -> str:
    plain = _find_part_text(msg, "text/plain")
    if plain.strip():
        return plain
    html = _find_part_text(msg, "text/html")
    if html.strip():
        return _strip_html(html)
    return ""


def _find_part_text(msg: Message, content_type: str) -> str:
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_type() != content_type:
            continue
        if part.get_filename():
            continue
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue
        return _decode_text_part(part)
    return ""


def _decode_text_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_attachments(msg: Message) -> list[Attachment]:
    out: list[Attachment] = []
    used_names: dict[str, int] = {}
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        content_type = part.get_content_type()
        if content_type == "message/rfc822":
            payload = part.as_bytes()
            safe = _safe_filename(filename or f"nested-{len(out) + 1}.eml")
        elif filename:
            payload = part.get_payload(decode=True) or b""
            safe = _safe_filename(filename)
        else:
            continue

        safe = _dedupe_filename(safe, used_names)
        out.append(
            Attachment(
                filename=safe,
                content_type=content_type,
                content_id=_decode_header_value(part.get("Content-ID")),
                data=payload,
            )
        )
    return out


def _dedupe_filename(name: str, used: dict[str, int]) -> str:
    if name not in used:
        used[name] = 1
        return name
    used[name] += 1
    stem, dot, ext = name.rpartition(".")
    if dot:
        return f"{stem}-{used[name]}.{ext}"
    return f"{name}-{used[name]}"


_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f/\\]+")


def _safe_filename(name: str) -> str:
    cleaned = _UNSAFE_FILENAME_CHARS.sub("", name).strip(" .")
    cleaned = cleaned.replace("..", "")
    return cleaned or "attachment.bin"


def _decode_header_value(raw: str | None) -> str | None:
    if raw is None:
        return None
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def _address_list(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    decoded = _decode_header_value(raw) or ""
    parsed = getaddresses([decoded])
    out: list[str] = []
    for name, addr in parsed:
        name = (name or "").strip()
        addr = (addr or "").strip()
        if name and addr:
            out.append(f"{name} <{addr}>")
        elif addr:
            out.append(addr)
        elif name:
            out.append(name)
    return tuple(out)


def _parse_references(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    decoded = _decode_header_value(raw) or ""
    refs = [piece.strip() for piece in decoded.split() if piece.strip()]
    normalized: list[str] = []
    for ref in refs:
        stripped = ref.strip("<>")
        if stripped:
            normalized.append(f"<{stripped}>")
    return tuple(normalized)


class _HTMLTextExtractor(HTMLParser):
    """Mirrors server/parsers.py:_HTMLTextExtractor — kept in sync deliberately."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._suppress = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._suppress += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._suppress:
            self._suppress -= 1

    def handle_data(self, data):
        if not self._suppress:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def _strip_html(html_text: str) -> str:
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html_text)
        text = extractor.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
