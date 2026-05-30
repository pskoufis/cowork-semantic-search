"""Per-message Outlook .msg unpacker.

For each ``foo.msg``, writes a sibling ``foo.txt`` (headers + body) and
places attachments in a shared sibling ``attachments/`` folder using the
naming scheme ``attachments/foo__<original-name>`` so multiple .msg files
in the same directory don't collide. Nested .msg attachments are extracted
to disk and recursively unpacked by the same algorithm.

Two callers:

1. The indexer (``server/indexer.py``): a preprocessing pass that runs
   :func:`ensure_unpacked` per .msg before the main walk, so the normal
   indexer sees the spawned ``.txt`` + extracted attachments via their
   native extensions.
2. The CLI (``python -m msg_handling.unpack <msg-path>``): manual unpack.

Usage:
    python -m msg_handling.unpack <msg-path>
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from msg_handling.messages import ParsedMessage, read_message

# Filesystem NAME_MAX is 255 bytes on the common targets (APFS, exFAT, ext4).
# Nested .msg recursion compounds attachment names (``a__b__c…``), which can
# blow past this and raise ENAMETOOLONG, so output filenames are capped.
_MAX_BASENAME_BYTES = 255


def ensure_unpacked(msg_path: Path, target: Path | None = None) -> Path:
    """Ensure ``<stem>.txt`` and ``attachments/<stem>__*`` are fresh.

    By default (``target=None``), outputs land next to the source ``.msg``
    — the in-place behavior the indexer's preprocessing pass and the
    ``python -m msg_handling.unpack`` CLI both rely on.

    When ``target=<dir>`` is provided, outputs land in that directory
    instead: ``<target>/<stem>.txt`` + ``<target>/attachments/<stem>__*``.
    The source ``.msg`` is left untouched. Used by the batch
    ``unpack_msg_folder.py`` script to mirror a corpus's subdirectory
    layout into a separate output tree.

    Idempotency: when ``<out>/<stem>.txt`` exists and is at least as new
    as the ``.msg``, no work is done. A re-extract first parses the
    ``.msg``, *then* cleans prior outputs for this stem, *then* writes —
    so a parse failure leaves the prior outputs in place.

    Returns the directory containing the outputs (either ``msg_path.parent``
    or the resolved ``target``).
    """
    msg_path = Path(msg_path).expanduser().resolve()
    if not msg_path.is_file():
        raise FileNotFoundError(f"msg file not found: {msg_path}")

    stem = msg_path.stem
    out_dir = (
        msg_path.parent if target is None
        else Path(target).expanduser().resolve()
    )
    txt_path = out_dir / f"{stem}.txt"

    if _is_fresh(msg_path, txt_path):
        return out_dir

    try:
        parsed = read_message(msg_path)
    except Exception as exc:
        print(
            f"warning: failed to read {msg_path.name}: {exc}",
            file=sys.stderr,
        )
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_prior_outputs(out_dir, stem)
    _write_outputs(out_dir, stem, parsed)
    _recurse_into_nested(out_dir, stem, parsed)
    return out_dir


def _is_fresh(msg_path: Path, txt_path: Path) -> bool:
    if not txt_path.exists():
        return False
    return msg_path.stat().st_mtime <= txt_path.stat().st_mtime


def _cleanup_prior_outputs(parent: Path, stem: str) -> None:
    """Remove the prior ``<stem>.txt`` and every ``attachments/<stem>__*`` file
    so a re-extract converges. Sibling files (other .msg's prior outputs and
    arbitrary unrelated content) are left untouched."""
    txt = parent / f"{stem}.txt"
    if txt.exists():
        txt.unlink()
    attachments_dir = parent / "attachments"
    if attachments_dir.is_dir():
        prefix = f"{stem}__"
        for child in attachments_dir.iterdir():
            if child.is_file() and child.name.startswith(prefix):
                child.unlink()


def _write_outputs(parent: Path, stem: str, parsed: ParsedMessage) -> None:
    attachment_rels: list[str] = []
    if parsed.attachments:
        attachments_dir = parent / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)
        for att in parsed.attachments:
            safe_name = _safe_basename(att.filename)
            if not safe_name:
                continue
            out = attachments_dir / _attachment_disk_name(stem, safe_name)
            out.write_bytes(att.data)
            attachment_rels.append(f"attachments/{out.name}")

    txt = parent / f"{stem}.txt"
    txt.write_text(_format_message_text(parsed, attachment_rels), encoding="utf-8")


def _recurse_into_nested(parent: Path, stem: str, parsed: ParsedMessage) -> None:
    """For each nested .msg attachment, recursively unpack the on-disk copy
    we just wrote. A failure in the inner unpack is logged and skipped so the
    outer unpack still succeeds."""
    for att in parsed.attachments:
        if not att.is_msg:
            continue
        safe_name = _safe_basename(att.filename)
        if not safe_name:
            continue
        inner = parent / "attachments" / _attachment_disk_name(stem, safe_name)
        if not inner.is_file():
            continue
        try:
            ensure_unpacked(inner)
        except Exception as exc:
            print(
                f"warning: nested unpack failed for {inner.name}: {exc}",
                file=sys.stderr,
            )


def _safe_basename(name: str) -> str:
    """Reduce an attachment-supplied filename to a safe basename.

    Defense against a malicious .msg whose attachment 'name' tries to escape
    the attachments/ directory via path traversal or absolute paths.
    ``Path(...).name`` already discards leading dirs and ``..`` parents on
    Posix; the regex also collapses any stray backslash separators that
    Path treats as ordinary characters here.
    """
    if not name:
        return ""
    name = name.replace("\x00", "")
    base = Path(name).name
    base = re.sub(r"[\\/]+", "_", base)
    return base.strip()


def _attachment_disk_name(stem: str, safe_name: str) -> str:
    """Build the on-disk attachment filename ``<stem>__<safe_name>``, capped to
    a filesystem-safe byte length.

    Short names pass through unchanged. When the combined name would exceed
    ``_MAX_BASENAME_BYTES`` (e.g. deep nested-.msg recursion compounding
    ``a__b__c…``), the extension is preserved and a short hash of the full name
    is appended for uniqueness, with the rest truncated to fit. Both the writer
    and the nested-recursion path call this, so the name they compute always
    matches the file on disk. Capping at each level also keeps the next
    recursion level's stem bounded.
    """
    full = f"{stem}__{safe_name}"
    if len(full.encode("utf-8")) <= _MAX_BASENAME_BYTES:
        return full
    ext = Path(safe_name).suffix
    if len(ext.encode("utf-8")) > 16:  # absurdly long "extension" — treat as none
        ext = ""
    digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:8]
    suffix = f".{digest}{ext}"
    head = full[: len(full) - len(ext)] if ext else full
    budget = _MAX_BASENAME_BYTES - len(suffix.encode("utf-8"))
    return _truncate_to_bytes(head, budget) + suffix


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Truncate ``text`` so its UTF-8 encoding is at most ``max_bytes``, without
    splitting a multibyte character."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _format_message_text(parsed: ParsedMessage, attachment_rels: list[str]) -> str:
    """Match the mbox unpack header-block layout so unpacked .msg and
    unpacked mbox messages produce identical embedding context."""
    lines = [
        f"From: {parsed.from_addr or ''}",
        f"To: {parsed.to or ''}",
        f"Cc: {parsed.cc or ''}",
        f"Subject: {parsed.subject or ''}",
        f"Date: {parsed.date or ''}",
        f"Message-ID: {parsed.message_id or ''}",
        "",
    ]
    lines.append(parsed.body or "")
    if attachment_rels:
        lines.append("")
        lines.append("Attachments:")
        for rel in attachment_rels:
            lines.append(f"  - {rel}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    msg_path = Path(args.msg_path).expanduser().resolve()
    if not msg_path.is_file():
        print(f"error: msg file not found: {msg_path}", file=sys.stderr)
        return 1
    ensure_unpacked(msg_path)
    print(
        f"unpacked {msg_path.name} into {msg_path.parent}",
        file=sys.stderr,
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unpack an Outlook .msg file into a sibling .txt + "
            "attachments/<stem>__* layout."
        ),
    )
    parser.add_argument("msg_path", help="Path to the .msg file")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
