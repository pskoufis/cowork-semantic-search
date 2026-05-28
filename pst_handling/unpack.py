"""Per-archive Outlook .pst unpacker.

For each ``foo.pst``, writes a sibling ``foo_unpacked/`` tree of
per-message ``.txt`` files mirroring the PST's folder hierarchy, with
attachments materialized to disk under ``attachments/msg-NNNN/`` next
to the corresponding message ``.txt``. The source ``foo.pst`` is left
untouched.

Two callers:

1. The indexer (``server/indexer.py``): a preprocessing pass that runs
   :func:`ensure_unpacked` per .pst before the main walk, so the normal
   indexer sees the spawned ``.txt`` + extracted attachments via their
   native extensions.
2. The CLI (``python -m pst_handling.unpack <pst-path>``): manual
   unpacking outside an indexing run. ``--output-dir`` is optional;
   when omitted, the unpacker writes to ``<stem>_unpacked/`` next to
   the source, picking ``-1``/``-2``/... suffixes if that sibling
   already exists and is non-empty.

Usage:
    python -m pst_handling.unpack <pst-path>
    python -m pst_handling.unpack <pst-path> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pst_handling.messages import (
    ParsedPstAttachment,
    ParsedPstMessage,
    iter_pst_messages,
)


def ensure_unpacked(pst_path: Path, target: Path | None = None) -> Path:
    """Ensure an unpacked tree exists for ``pst_path``; return its path.

    - ``target=None`` (default): writes to ``<stem>_unpacked/`` next to
      the .pst. Used by the indexer's preprocessing pass and the
      ``python -m pst_handling.unpack`` CLI.
    - ``target=<path>``: writes to that path instead. Used by the
      batch ``unpack_pst_folder.py`` script to mirror a corpus's
      subdirectory layout into a separate output tree.

    In both modes, the unpack is mtime-idempotent: when the output
    directory already exists, is non-empty, and is at least as new as
    the source ``.pst``, the existing tree is reused; otherwise the
    output is clobbered and rewritten from scratch. This keeps re-runs
    cheap and matches the msg-handling preprocessor's posture.

    Raises ``FileNotFoundError`` if ``pst_path`` does not exist.
    """
    pst_path = Path(pst_path).expanduser().resolve()
    if not pst_path.is_file():
        raise FileNotFoundError(f"pst file not found: {pst_path}")

    if target is None:
        out = pst_path.parent / f"{pst_path.stem}_unpacked"
    else:
        out = Path(target).expanduser().resolve()

    if out.exists() and any(out.iterdir()):
        if pst_path.stat().st_mtime <= out.stat().st_mtime:
            return out
        # .pst is newer → fall through to full clobber + rewrite.

    _prepare_output(out)
    msg_count, folder_count = _write_unpacked_tree(pst_path, out)
    print(
        f"Unpacked {msg_count} message(s) from {pst_path.name} across "
        f"{folder_count} folder(s) at {out}",
        file=sys.stderr,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pst_path = Path(args.pst_path).expanduser().resolve()

    if not pst_path.is_file():
        print(f"error: pst file not found: {pst_path}", file=sys.stderr)
        return 1

    if args.output_dir is None:
        target = _pick_default_output_dir(pst_path)
        print(f"notice: writing to default output dir {target}", file=sys.stderr)
    else:
        target = Path(args.output_dir).expanduser().resolve()

    ensure_unpacked(pst_path, target=target)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unpack a .pst archive into a sibling <stem>_unpacked/ tree "
            "of per-message .txt files mirroring the PST folder hierarchy."
        ),
    )
    parser.add_argument("pst_path", help="Path to the .pst file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write the unpacked tree into. Defaults to "
            "<pst-stem>_unpacked next to the .pst (with -1/-2/... "
            "suffixes if that sibling exists and is non-empty)."
        ),
    )
    return parser.parse_args(argv)


def _pick_default_output_dir(pst_path: Path) -> Path:
    """Choose ``<stem>_unpacked`` next to the .pst, falling back to
    ``<stem>_unpacked-1/-2/...`` when the candidate exists and is
    non-empty. An existing empty directory is reused (no suffix bump).

    Only used by the CLI's default branch — the indexer's preprocessing
    pass always uses the plain ``<stem>_unpacked`` path via
    :func:`ensure_unpacked`.
    """
    parent = pst_path.parent
    stem = pst_path.stem
    base = parent / f"{stem}_unpacked"
    if not _is_in_use(base):
        return base
    for n in range(1, 1000):
        candidate = parent / f"{stem}_unpacked-{n}"
        if not _is_in_use(candidate):
            return candidate
    raise RuntimeError(
        f"could not find an unused default output dir next to {pst_path}; "
        "pass --output-dir explicitly."
    )


def _is_in_use(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _prepare_output(out: Path) -> None:
    if out.exists():
        if any(out.iterdir()):
            print(f"notice: clearing prior output at {out}", file=sys.stderr)
            for child in out.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        out.mkdir(parents=True, exist_ok=True)


def _write_unpacked_tree(pst_path: Path, out: Path) -> tuple[int, int]:
    """Walk ``pst_path`` and write per-message ``.txt`` files into ``out``.

    Returns ``(message_count, folder_count)`` for the summary line. The
    folder count is the number of distinct PST folders that contained
    at least one mail message — empty folders don't contribute.
    """
    # Lazy import — server.indexer imports this module (via the
    # preprocessing hook), so a module-level import would be circular.
    from server.indexer import MAX_FILE_SIZE_BYTES

    out.mkdir(parents=True, exist_ok=True)
    per_folder_counts: dict[str, int] = {}
    msg_count = 0
    for message in iter_pst_messages(pst_path):
        folder_dir = _make_folder_dir(out, message.folder_path)
        folder_dir.mkdir(parents=True, exist_ok=True)
        pos = per_folder_counts.get(message.folder_path, 0) + 1
        per_folder_counts[message.folder_path] = pos
        attachments_root = folder_dir / "attachments"
        _write_message(folder_dir, attachments_root, message, pos, MAX_FILE_SIZE_BYTES)
        msg_count += 1
    return msg_count, len(per_folder_counts)


def _make_folder_dir(out: Path, folder_path: str) -> Path:
    """Resolve a PST folder path (``"Top/Inbox"``) into an on-disk
    directory under ``out``. Empty paths land at the root."""
    if not folder_path:
        return out
    segments = [_safe_path_segment(part) for part in folder_path.split("/")]
    segments = [s for s in segments if s]
    return out.joinpath(*segments) if segments else out


def _write_message(
    folder_dir: Path,
    attachments_root: Path,
    message: ParsedPstMessage,
    pos: int,
    max_attachment_bytes: int,
) -> None:
    if message.error:
        # Surface the error in the filename so it's eyeballable without opening.
        txt_path = folder_dir / f"msg-{pos:04d}_error.txt"
        txt_path.write_text(
            f"ERROR: {message.error}\n\nFolder: {message.folder_path}\n",
            encoding="utf-8",
        )
        return

    date_slug = _date_slug(message.date)
    from_slug = _from_slug(message.sender) or "unknown"
    base = f"msg-{pos:04d}_{date_slug}_{from_slug}"
    txt_path = folder_dir / f"{base}.txt"

    saved_attachments: list[str] = []
    skipped_notes: list[str] = []
    if message.attachments:
        msg_attach_dir = attachments_root / f"msg-{pos:04d}"
        msg_attach_dir.mkdir(parents=True, exist_ok=True)
        for att in message.attachments:
            outcome = _materialize_attachment(
                att, msg_attach_dir, max_attachment_bytes
            )
            if outcome.rel_path is not None:
                saved_attachments.append(outcome.rel_path)
            if outcome.skip_note is not None:
                skipped_notes.append(outcome.skip_note)

    txt_path.write_text(
        _format_message_text(message, saved_attachments, skipped_notes, folder_dir),
        encoding="utf-8",
    )


class _AttachmentOutcome:
    """Result of trying to materialize one attachment to disk."""

    __slots__ = ("rel_path", "skip_note")

    def __init__(self, *, rel_path: str | None, skip_note: str | None) -> None:
        self.rel_path = rel_path
        self.skip_note = skip_note


def _materialize_attachment(
    att: ParsedPstAttachment,
    msg_attach_dir: Path,
    max_attachment_bytes: int,
) -> _AttachmentOutcome:
    """Write one attachment to ``msg_attach_dir``, or skip it with a note.

    An attachment over the per-file size cap is skipped with a note
    folded into the ``.txt`` instead of failing — same OOM-guard
    posture as the prior inline PST extractor.
    """
    safe_name = _safe_basename(att.filename)
    if not safe_name:
        return _AttachmentOutcome(rel_path=None, skip_note=None)
    if max_attachment_bytes and att.size > max_attachment_bytes:
        return _AttachmentOutcome(
            rel_path=None,
            skip_note=(
                f"[Attachment: {safe_name} — skipped, "
                f"{att.size / 1024 / 1024:.1f} MB]"
            ),
        )
    target = msg_attach_dir / safe_name
    try:
        att.write_to(target)
    except Exception as exc:
        return _AttachmentOutcome(
            rel_path=None,
            skip_note=f"[Attachment: {safe_name} — extraction failed: {exc}]",
        )
    rel = str(target.relative_to(msg_attach_dir.parent.parent))
    return _AttachmentOutcome(rel_path=rel, skip_note=None)


def _format_message_text(
    message: ParsedPstMessage,
    saved_attachments: list[str],
    skipped_notes: list[str],
    folder_dir: Path,
) -> str:
    """Match the mbox/msg unpack header-block layout so unpacked PST
    messages produce identical embedding context.

    PST exposes Subject / From / Date directly via pypff but not To /
    Cc / Message-ID — those header lines are emitted but left empty,
    so the header shape stays consistent across all three formats.
    """
    lines = [
        f"From: {message.sender or ''}",
        "To: ",
        "Cc: ",
        f"Subject: {message.subject or ''}",
        f"Date: {message.date or ''}",
        "Message-ID: ",
        f"Folder: {message.folder_path or ''}",
        "",
        message.body or "",
    ]
    if saved_attachments or skipped_notes:
        lines.append("")
        lines.append("Attachments:")
        # ``saved_attachments`` paths are relative to folder_dir's parent
        # of the attachments dir — rebase to be relative to folder_dir.
        for rel in saved_attachments:
            lines.append(f"  - {rel}")
        for note in skipped_notes:
            lines.append(f"  - {note}")
    return "\n".join(lines).rstrip() + "\n"


_SLUG_BAD_CHARS = re.compile(r"[^a-z0-9]+")
# Reject path separators and control chars from anything we use as a
# directory or filename segment — pypff folder names can contain
# slashes, NULs, or other surprises in damaged archives.
_PATH_BAD_CHARS = re.compile(r"[\x00-\x1f/\\:]+")


def _safe_path_segment(name: str) -> str:
    """Reduce a PST folder name to a safe single-segment directory name.

    Strips path separators and control chars; collapses whitespace.
    Empty / dotted names fall back to ``_unnamed`` so the path never
    contains a ``.`` or ``..`` segment.
    """
    cleaned = _PATH_BAD_CHARS.sub("-", name or "").strip()
    cleaned = cleaned.strip(". ")
    return cleaned or "_unnamed"


def _safe_basename(name: str) -> str:
    """Reduce an attachment filename to a safe basename.

    Defense against a malicious .pst whose attachment 'name' tries to
    escape the per-message attachments directory via path traversal
    or absolute paths.
    """
    if not name:
        return ""
    name = name.replace("\x00", "")
    base = Path(name).name
    base = re.sub(r"[\\/]+", "_", base)
    return base.strip()


def _date_slug(date_str: str | None) -> str:
    """Render a date string as ``YYYYMMDD`` for the message filename.

    pypff's submit/delivery times come back as ``datetime`` objects in
    practice but the dataclass stringifies them; this best-effort
    parser handles both ``str(datetime)`` and ``YYYY-MM-DD`` shapes.
    Anything unparseable falls back to ``nodate``.
    """
    if not date_str:
        return "nodate"
    # Common datetime str() output starts with "YYYY-MM-DD" — grab those
    # ten chars and strip the dashes. This handles both naive and
    # tz-aware datetime stringifications.
    candidate = date_str[:10]
    try:
        dt = datetime.strptime(candidate, "%Y-%m-%d")
    except ValueError:
        return "nodate"
    return dt.strftime("%Y%m%d")


def _from_slug(sender: str | None) -> str:
    if not sender:
        return ""
    candidate = sender.lower()
    candidate = _SLUG_BAD_CHARS.sub("-", candidate).strip("-")
    return candidate[:40]


if __name__ == "__main__":
    raise SystemExit(main())
