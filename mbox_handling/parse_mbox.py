"""Standalone mbox parser — exploration script.

Parses an mbox file into messages.jsonl (one JSON object per message) and
recreates every attachment on disk. Stdlib only. Not wired into
server/parsers.py — see docs/superpowers/specs/2026-05-24-mbox-parser-design.md.
"""

from __future__ import annotations

import argparse
import json
import mailbox
import re
import shutil
import sys
from email.header import decode_header, make_header
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MBOX = REPO_ROOT / "mbox_handling" / "samples" / "sample.mbox"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "mbox_path",
        nargs="?",
        default=str(DEFAULT_MBOX),
        help=f"Path to the mbox file (default: {DEFAULT_MBOX})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for messages.jsonl and attachments/ (default: mbox_handling/output/<stem>/)",
    )
    args = parser.parse_args(argv)

    mbox_path = Path(args.mbox_path).expanduser().resolve()
    if not mbox_path.is_file():
        print(f"error: mbox file not found: {mbox_path}", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "mbox_handling" / "output" / mbox_path.stem
    )
    attachments_root = output_dir / "attachments"
    jsonl_path = output_dir / "messages.jsonl"

    if output_dir.exists() and (jsonl_path.exists() or attachments_root.exists()):
        print(f"notice: clearing prior output at {output_dir}", file=sys.stderr)
        if jsonl_path.exists():
            jsonl_path.unlink()
        if attachments_root.exists():
            shutil.rmtree(attachments_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    mbox = mailbox.mbox(str(mbox_path))
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as out:
        for idx, msg in enumerate(mbox, start=1):
            try:
                record, attachments = _parse_message(msg, idx)
            except Exception as exc:
                record = {
                    "index": idx,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                attachments = []

            saved = []
            for filename, content_type, content_id, payload in attachments:
                msg_dir = attachments_root / f"msg-{idx:04d}"
                msg_dir.mkdir(parents=True, exist_ok=True)
                target = _unique_path(msg_dir, filename)
                target.write_bytes(payload)
                saved.append({
                    "filename": target.name,
                    "content_type": content_type,
                    "content_id": content_id,
                    "size_bytes": len(payload),
                    "saved_to": str(target.relative_to(output_dir)),
                })
            if "error" not in record:
                record["attachments"] = saved

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Parsed {count} messages -> {jsonl_path}", file=sys.stderr)
    return 0


def _parse_message(msg: Message, idx: int) -> tuple[dict, list[tuple[str, str, str | None, bytes]]]:
    """Build the JSONL record + collect attachment payloads. Pure — no I/O."""
    body = _extract_body(msg)
    attachments = _extract_attachments(msg)
    record = {
        "index": idx,
        "from": _decode_header_value(msg.get("From")),
        "to": _address_list(msg.get("To")),
        "cc": _address_list(msg.get("Cc")),
        "date": _decode_header_value(msg.get("Date")),
        "subject": _decode_header_value(msg.get("Subject")),
        "message_id": _decode_header_value(msg.get("Message-ID")),
        "body": body,
    }
    return record, attachments


def _extract_body(msg: Message) -> str:
    """text/plain → HTML-stripped text → empty. Matches server/parsers.py:_pst_body."""
    plain = _find_part_text(msg, "text/plain")
    if plain.strip():
        return plain
    html = _find_part_text(msg, "text/html")
    if html.strip():
        return _strip_html(html)
    return ""


def _find_part_text(msg: Message, content_type: str) -> str:
    """Return the first non-attachment part of the given content type, decoded to str."""
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_type() != content_type:
            continue
        if part.get_filename():
            continue  # it's an attachment that happens to be text/*, not the body
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue
        return _decode_text_part(part)
    return ""


def _decode_text_part(part: Message) -> str:
    """Decode a text/* part to str, tolerating bad/missing charsets."""
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_attachments(msg: Message) -> list[tuple[str, str, str | None, bytes]]:
    """Collect every part with a filename (inline or attachment). Returns (filename, content_type, content_id, bytes)."""
    out: list[tuple[str, str, str | None, bytes]] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        content_type = part.get_content_type()
        if content_type == "message/rfc822":
            # Nested message — not recursed in v1; save raw bytes as placeholder .eml.
            payload = part.as_bytes()
            out.append((
                _safe_filename(filename or f"nested-{len(out) + 1}.eml"),
                content_type,
                _decode_header_value(part.get("Content-ID")),
                payload,
            ))
            continue
        if not filename:
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = b""
        out.append((
            _safe_filename(filename),
            content_type,
            _decode_header_value(part.get("Content-ID")),
            payload,
        ))
    return out


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif tag in {"br", "p", "div", "li", "tr"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._chunks)).strip()


def _strip_html(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    parser.close()
    return parser.text()


def _decode_header_value(raw: str | None) -> str | None:
    if raw is None:
        return None
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def _address_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    decoded = _decode_header_value(raw) or ""
    # Naive split on commas — adequate for the exploration script; address
    # parsers (email.utils.getaddresses) over-normalize and lose display names.
    return [piece.strip() for piece in decoded.split(",") if piece.strip()]


_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f/\\]+")


def _safe_filename(name: str) -> str:
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    return cleaned or "attachment.bin"


def _unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    n = 2
    while True:
        candidate = directory / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


if __name__ == "__main__":
    raise SystemExit(main())
