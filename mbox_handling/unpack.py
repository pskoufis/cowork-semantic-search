"""Thread-grouped mbox unpacker CLI.

Writes one directory per thread under ``--output-dir``, with one .txt per
message inside the thread directory and attachments co-located under
``attachments/msg-NNNN/``. Orphan messages (no resolvable parent) go to
``_unthreaded/``.

``--output-dir`` is required (no default) so the tree never accidentally
lands inside an indexed folder.

Usage:
    python -m mbox_handling.unpack <mbox-path> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

from mbox_handling.messages import ParsedMessage, iter_messages
from mbox_handling.threading import ThreadAssignment, compute_thread_assignments


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mbox_path = Path(args.mbox_path).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()

    if not mbox_path.is_file():
        print(f"error: mbox file not found: {mbox_path}", file=sys.stderr)
        return 1

    _prepare_output(out)

    messages = list(iter_messages(mbox_path))
    assignments = compute_thread_assignments(messages)

    groups = _group_messages(messages, assignments)
    thread_count = 0
    for thread_idx, group in enumerate(_iter_threads(groups), start=1):
        thread_dir = _thread_dir(out, thread_idx, group)
        thread_dir.mkdir(parents=True, exist_ok=True)
        attachments_root = thread_dir / "attachments"
        for msg_pos, msg in enumerate(group.messages, start=1):
            _write_message(thread_dir, attachments_root, msg, msg_pos)
        if not group.is_orphan_bucket:
            thread_count += 1

    print(
        f"Unpacked {len(messages)} message(s) into {thread_count} thread(s)",
        file=sys.stderr,
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unpack an mbox file into thread-grouped directories."
    )
    parser.add_argument("mbox_path", help="Path to the mbox file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write thread/<msg>.txt + attachments/ into. Required.",
    )
    return parser.parse_args(argv)


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


class _Group:
    """One bucket of messages — either a real thread or the orphan bucket."""

    def __init__(self, is_orphan_bucket: bool) -> None:
        self.is_orphan_bucket = is_orphan_bucket
        self.messages: list[ParsedMessage] = []


def _group_messages(
    messages: list[ParsedMessage],
    assignments: dict[str, ThreadAssignment],
) -> dict[str, _Group]:
    """Return {key -> _Group} where key is thread_id for real threads or
    ``"_unthreaded"`` for the single orphan bucket. Messages inside each
    real thread are ordered by position_in_thread; orphans are ordered by
    index.
    """
    groups: dict[str, _Group] = {}
    # Real threads first.
    for msg in messages:
        assignment = _assignment_for(msg, assignments)
        if assignment is None:
            continue
        if assignment.is_orphan:
            bucket = groups.setdefault("_unthreaded", _Group(is_orphan_bucket=True))
            bucket.messages.append(msg)
        else:
            key = assignment.thread_id
            group = groups.setdefault(key, _Group(is_orphan_bucket=False))
            group.messages.append(msg)
    # Sort messages within each thread by position_in_thread (real threads)
    # or by index (orphans).
    for key, group in groups.items():
        if group.is_orphan_bucket:
            group.messages.sort(key=lambda m: m.index)
        else:
            group.messages.sort(
                key=lambda m: _assignment_for(m, assignments).position_in_thread
            )
    return groups


def _assignment_for(
    msg: ParsedMessage, assignments: dict[str, ThreadAssignment]
) -> ThreadAssignment | None:
    if msg.message_id and msg.message_id in assignments:
        return assignments[msg.message_id]
    key = f"no-id-{msg.index}"
    return assignments.get(key)


def _iter_threads(groups: dict[str, _Group]):
    """Yield groups in deterministic order: real threads (by min date / index of
    first message) first, then the orphan bucket last."""
    real = [g for g in groups.values() if not g.is_orphan_bucket]
    real.sort(key=lambda g: (g.messages[0].date or "", g.messages[0].index))
    yield from real
    if "_unthreaded" in groups:
        yield groups["_unthreaded"]


def _thread_dir(out: Path, thread_idx: int, group: _Group) -> Path:
    if group.is_orphan_bucket:
        return out / "_unthreaded"
    first = group.messages[0]
    slug = _subject_slug(first.subject)
    return out / f"thread-{thread_idx:04d}_{slug}"


def _write_message(
    thread_dir: Path,
    attachments_root: Path,
    msg: ParsedMessage,
    msg_pos: int,
) -> None:
    date_slug = _date_slug(msg.date)
    from_slug = _from_slug(msg.from_addr) or "unknown"
    base = f"msg-{msg_pos:04d}_{date_slug}_{from_slug}"
    if msg.error:
        # Surface the error in the filename so it's eyeballable without opening.
        txt_path = thread_dir / f"msg-{msg_pos:04d}_error.txt"
    else:
        txt_path = thread_dir / f"{base}.txt"

    saved_attachments: list[str] = []
    if msg.attachments:
        msg_attach_dir = attachments_root / f"msg-{msg_pos:04d}"
        msg_attach_dir.mkdir(parents=True, exist_ok=True)
        for att in msg.attachments:
            target = msg_attach_dir / att.filename
            target.write_bytes(att.data)
            saved_attachments.append(
                str(target.relative_to(thread_dir))
            )

    txt_path.write_text(_format_message_text(msg, saved_attachments), encoding="utf-8")


def _format_message_text(msg: ParsedMessage, attachment_rels: list[str]) -> str:
    lines = [
        f"From: {msg.from_addr or ''}",
        f"To: {', '.join(msg.to)}",
        f"Cc: {', '.join(msg.cc)}",
        f"Subject: {msg.subject or ''}",
        f"Date: {msg.date or ''}",
        f"Message-ID: {msg.message_id or ''}",
        f"In-Reply-To: {msg.in_reply_to or ''}",
        f"References: {' '.join(msg.references)}",
        "",
    ]
    if msg.error:
        lines.append(f"ERROR: {msg.error}")
        lines.append("")
    lines.append(msg.body or "")
    if attachment_rels:
        lines.append("")
        lines.append("Attachments:")
        for rel in attachment_rels:
            lines.append(f"  - {rel}")
    return "\n".join(lines).rstrip() + "\n"


_SUBJECT_PREFIX_RE = re.compile(r"^\s*(re|fwd?|aw|wg)\s*:\s*", re.IGNORECASE)
_SLUG_BAD_CHARS = re.compile(r"[^a-z0-9]+")


def _subject_slug(subject: str | None) -> str:
    s = (subject or "").strip()
    # Strip Re:/Fwd:/Aw:/Wg: prefixes repeatedly.
    while True:
        new = _SUBJECT_PREFIX_RE.sub("", s)
        if new == s:
            break
        s = new
    s = s.lower()
    s = _SLUG_BAD_CHARS.sub("-", s).strip("-")
    return (s[:60] or "no-subject")


def _date_slug(date_str: str | None) -> str:
    if not date_str:
        return "nodate"
    try:
        dt = parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return "nodate"
    if dt is None:
        return "nodate"
    return dt.strftime("%Y%m%d")


def _from_slug(addr: str | None) -> str:
    if not addr:
        return ""
    _, email_only = parseaddr(addr)
    candidate = email_only.split("@", 1)[0] if "@" in email_only else email_only
    if not candidate:
        candidate = addr
    candidate = candidate.lower()
    candidate = _SLUG_BAD_CHARS.sub("-", candidate).strip("-")
    return candidate[:40]


if __name__ == "__main__":
    raise SystemExit(main())
