"""Per-format text extraction from document files.

Spreadsheet formats (.csv, .xlsx, .xlsm, .xls) do not flow through
extract_text — they route via server/spreadsheets.py to a description-based
queue (preview → LLM description → embedded chunk). server.indexer routes
them directly; extract_text rejects them so a stray caller cannot fall into
a raw-row scheme.
"""

import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx", ".pptx",
    # Spreadsheets (.csv/.xlsx/.xlsm/.xls) temporarily disabled.
    ".pst",
    ".mbox",
}


def extract_text(file_path: Path) -> list[dict]:
    """Extract text from a file, returning list of {text, metadata} dicts.

    Metadata may include page_number (PDF), slide_number (PPTX),
    pst_folder (PST). Spreadsheets are not handled here — see the module
    docstring.
    """
    suffix = file_path.suffix.lower()

    match suffix:
        case ".txt" | ".md":
            text = file_path.read_text(encoding="utf-8", errors="replace")
            return [{"text": text, "metadata": {}}]
        case ".pdf":
            return _extract_pdf(file_path)
        case ".docx":
            return _extract_docx(file_path)
        case ".pptx":
            return _extract_pptx(file_path)
        case ".pst":
            return _extract_pst(file_path)
        case ".mbox":
            return _extract_mbox(file_path)
        case _:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported via extract_text: .txt, .md, .pdf, .docx, .pptx, "
                f".pst, .mbox. Spreadsheets (.csv/.xlsx/.xlsm/.xls) route through "
                f"server.spreadsheets."
            )


def _extract_pdf(file_path: Path) -> list[dict]:
    import pymupdf

    doc = pymupdf.open(str(file_path))
    parts = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        parts.append({"text": text.strip(), "metadata": {"page_number": page_num}})
    doc.close()
    return parts


def _extract_docx(file_path: Path) -> list[dict]:
    from docx import Document

    doc = Document(str(file_path))
    text = "\n\n".join(p.text for p in doc.paragraphs)
    return [{"text": text, "metadata": {}}]


def _extract_pptx(file_path: Path) -> list[dict]:
    from pptx import Presentation

    prs = Presentation(str(file_path))
    parts = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts.append(shape.text_frame.text)
        parts.append({
            "text": "\n".join(texts),
            "metadata": {"slide_number": slide_num},
        })
    return parts


# --- PST (Outlook archive) extraction --------------------------------------
#
# A .pst is a container of folders and email messages, not a single document.
# It is handled like a multi-page PDF: the whole archive is one source_file and
# each mail message becomes one part. pypff surfaces the common message fields
# directly but not arbitrary MAPI properties — the message class and an
# attachment's filename are only reachable by scanning record-set entries.

_PR_MESSAGE_CLASS = 0x001A          # PidTagMessageClass
_PR_ATTACH_LONG_FILENAME = 0x3707   # PidTagAttachLongFilename
_PR_ATTACH_FILENAME = 0x3704        # PidTagAttachFilename (8.3 short name)


def _extract_pst(file_path: Path) -> list[dict]:
    """Extract one part per mail message from an Outlook .pst archive."""
    try:
        import pypff
    except ImportError as exc:
        raise ImportError(
            "Reading .pst archives requires the 'pst' extra: "
            "pip install 'cowork-semantic-search[pst]'"
        ) from exc

    parts: list[dict] = []
    pst = pypff.file()
    try:
        pst.open(str(file_path))
        root = pst.get_root_folder()
        if root is not None:
            _walk_pst_folder(root, "", parts)
    finally:
        try:
            pst.close()
        except Exception:
            pass
    return parts


def _walk_pst_folder(folder, folder_path: str, parts: list[dict]) -> None:
    """Recursively collect mail messages from a PST folder tree."""
    for i in range(folder.number_of_sub_messages):
        message = folder.get_sub_message(i)
        if not _is_mail_item(message):
            continue  # skip calendar items, contacts, tasks, notes
        parts.append(_pst_message_to_part(message, folder_path))
    for i in range(folder.number_of_sub_folders):
        sub = folder.get_sub_folder(i)
        sub_name = _pst_decode(_pst_safe(sub.get_name)) or "(unnamed)"
        child_path = f"{folder_path}/{sub_name}" if folder_path else sub_name
        _walk_pst_folder(sub, child_path, parts)


def _pst_message_to_part(message, folder_path: str) -> dict:
    """Turn one mail message into a {text, metadata} part.

    A Subject/From/Date/Folder header is folded into the text so the message
    stays attributable and searchable — the store schema has no column for it.
    The header is always present, so a message is never an empty part even when
    every body field is missing. Text extracted from indexable attachments is
    appended after the body.
    """
    subject = _pst_decode(_pst_safe(message.get_subject)) or "(no subject)"
    sender = _pst_decode(_pst_safe(message.get_sender_name))
    sent = (_pst_safe(message.get_client_submit_time)
            or _pst_safe(message.get_delivery_time))
    header = (
        f"Subject: {subject}\n"
        f"From: {sender}\n"
        f"Date: {sent or ''}\n"
        f"Folder: {folder_path}"
    )
    segments = [header]
    body = _pst_body(message)
    if body.strip():
        segments.append(body)
    attachments = _pst_attachments_text(message)
    if attachments.strip():
        segments.append(attachments)
    return {
        "text": "\n\n".join(segments),
        "metadata": {"pst_folder": folder_path},
    }


def _pst_body(message) -> str:
    """Return the message body — plain text, else HTML, else RTF.

    Modern Outlook mail is frequently HTML-only with no plain-text body, so a
    plain-text-only read would silently lose those messages' content.
    """
    plain = _pst_decode(_pst_safe(message.get_plain_text_body))
    if plain.strip():
        return plain
    html = _pst_decode(_pst_safe(message.get_html_body))
    if html.strip():
        return _strip_html(html)
    rtf = _pst_decode(_pst_safe(message.get_rtf_body))
    if rtf.strip():
        return _strip_rtf(rtf)
    return ""


def _pst_attachments_text(message) -> str:
    """Extract text from a mail message's file attachments.

    Each attachment is isolated — a skip or failure appends a short note
    rather than aborting the message.
    """
    count = _pst_safe(message.get_number_of_attachments) or 0
    sections: list[str] = []
    for i in range(count):
        try:
            attachment = message.get_attachment(i)
        except Exception:
            continue
        section = _pst_attachment_section(attachment)
        if section:
            sections.append(section)
    return "\n\n".join(sections)


def _pst_attachment_section(attachment) -> str:
    """Return text for one attachment, or a one-line note when it is skipped.

    A supported, in-cap attachment is written to a temp file and re-run through
    extract_text. The per-file size cap cannot see inside a .pst, so it is
    re-applied here per attachment as the OOM guard. Nested .pst attachments
    are never recursed into.
    """
    # Lazy import: indexer.py imports this module, so a module-level import
    # would be circular.
    from server.indexer import MAX_FILE_SIZE_BYTES

    name = (_pst_prop_string(attachment, _PR_ATTACH_LONG_FILENAME)
            or _pst_prop_string(attachment, _PR_ATTACH_FILENAME))
    if not name:
        return ""  # embedded message or inline content — no filename to route on
    name = name.strip()
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS or ext == ".pst":
        return f"[Attachment: {name} — not indexed]"

    size = _pst_safe(attachment.get_size) or 0
    if MAX_FILE_SIZE_BYTES and size > MAX_FILE_SIZE_BYTES:
        return f"[Attachment: {name} — skipped, {size / 1024 / 1024:.1f} MB]"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            remaining = size
            while remaining > 0:
                block = attachment.read_buffer(min(remaining, 1024 * 1024))
                if not block:
                    break
                tmp.write(block)
                remaining -= len(block)
            tmp_path = Path(tmp.name)
        parts = extract_text(tmp_path)
    except Exception as exc:
        return f"[Attachment: {name} — extraction failed: {exc}]"
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    body = "\n\n".join(p["text"] for p in parts if p.get("text", "").strip())
    if not body.strip():
        return f"[Attachment: {name} — no extractable text]"
    return f"--- Attachment: {name} ---\n{body}"


def _is_mail_item(message) -> bool:
    """True for ordinary mail.

    A PST folder also stores calendar items, contacts, tasks and notes as
    'messages', distinguished by a non-mail message class (IPM.Appointment,
    IPM.Contact, ...). When the class cannot be read the item is kept, so
    genuine mail is never dropped.
    """
    message_class = _pst_prop_string(message, _PR_MESSAGE_CLASS)
    if not message_class:
        return True
    return message_class.upper().startswith("IPM.NOTE")


def _pst_prop_string(item, entry_type: int) -> str | None:
    """Read a MAPI property as a string from a pypff item's record sets.

    pypff surfaces a fixed set of fields directly; everything else (message
    class, attachment filename) is only reachable by scanning record entries
    for a matching entry type.
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


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text from an HTML document, dropping script/style."""

    def __init__(self):
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
    """Reduce an HTML body to readable plain text."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html_text)
        text = extractor.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html_text)  # crude fallback
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_rtf(rtf_text: str) -> str:
    """Best-effort plain text from an RTF body.

    RTF is only the last-resort body fallback (almost all mail has a plain-text
    or HTML body), so this does a crude control-word strip, not a full parse.
    """
    text = re.sub(r"\\par[d]?\b", "\n", rtf_text)
    text = re.sub(r"\\(line|tab)\b", " ", text)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d*\s?", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --- mbox (Unix mailbox) extraction ----------------------------------------
#
# Mirrors _extract_pst: one source_file per mbox, one part per message, with
# a Subject/From/Date/Thread header folded into the text so each message
# stays attributable. Threading metadata (thread_id/in_reply_to/references)
# rides on the part so the search layer can re-group by conversation at
# query time without re-indexing.


def _extract_mbox(file_path: Path) -> list[dict]:
    """Extract one part per mail message from an mbox archive."""
    from mbox_handling.messages import iter_messages
    from mbox_handling.threading import compute_thread_assignments

    messages = list(iter_messages(file_path))
    assignments = compute_thread_assignments(messages)

    parts: list[dict] = []
    for msg in messages:
        thread_id = "unthreaded"
        if msg.message_id and msg.message_id in assignments:
            thread_id = assignments[msg.message_id].thread_id
        elif not msg.message_id and f"no-id-{msg.index}" in assignments:
            thread_id = assignments[f"no-id-{msg.index}"].thread_id

        header = (
            f"Subject: {msg.subject or '(no subject)'}\n"
            f"From: {msg.from_addr or ''}\n"
            f"Date: {msg.date or ''}\n"
            f"Thread: {thread_id}"
        )
        segments = [header]
        if msg.body and msg.body.strip():
            segments.append(msg.body)
        parts.append({
            "text": "\n\n".join(segments),
            "metadata": {
                "thread_id": thread_id,
                "in_reply_to": msg.in_reply_to or "",
                "references": ",".join(msg.references) if msg.references else "",
            },
        })
    return parts
