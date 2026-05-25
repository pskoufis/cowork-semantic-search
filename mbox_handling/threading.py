"""Headers-only mbox thread assignment.

Builds a Message-ID parent graph from In-Reply-To and References headers,
collapses to connected components, and assigns each message a stable
short-hash thread_id. Messages with no resolvable parent become roots of
their own thread; if the thread has only one member, it is marked orphan
(consumers put orphans in an `_unthreaded/` bucket).

Note: messages without a Message-ID at all get a synthetic key
``no-id-{index}`` in the returned assignment map. They are always orphans.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from mbox_handling.messages import ParsedMessage


@dataclass(frozen=True)
class ThreadAssignment:
    thread_id: str
    is_orphan: bool
    root_message_id: str
    position_in_thread: int


def compute_thread_assignments(
    messages: Iterable[ParsedMessage],
) -> dict[str, ThreadAssignment]:
    msgs = list(messages)
    by_id = {m.message_id: m for m in msgs if m.message_id}

    # parent[mid] = parent_mid or None (root). Only built for messages WITH a Message-ID.
    parent: dict[str, str | None] = {}
    for m in msgs:
        if not m.message_id:
            continue
        parent[m.message_id] = _resolve_parent(m, by_id)

    # Collapse each mid to its root, breaking cycles.
    root_of: dict[str, str] = {}
    for mid in parent:
        root_of[mid] = _find_root(mid, parent)

    # Group: root_id -> [mid, ...]
    groups: dict[str, list[str]] = defaultdict(list)
    for mid, root in root_of.items():
        groups[root].append(mid)

    assignments: dict[str, ThreadAssignment] = {}
    for root_id, member_ids in groups.items():
        members_sorted = sorted(
            (by_id[mid] for mid in member_ids),
            key=lambda m: (m.date or "", m.index),
        )
        thread_id = _short_hash(root_id)
        is_orphan = len(member_ids) == 1
        for pos, m in enumerate(members_sorted):
            assignments[m.message_id] = ThreadAssignment(
                thread_id=thread_id,
                is_orphan=is_orphan,
                root_message_id=root_id,
                position_in_thread=pos,
            )

    # Messages without Message-ID: each gets a synthetic orphan thread.
    for m in msgs:
        if not m.message_id:
            synthetic = f"no-id-{m.index}"
            assignments[synthetic] = ThreadAssignment(
                thread_id=_short_hash(synthetic),
                is_orphan=True,
                root_message_id=synthetic,
                position_in_thread=0,
            )

    return assignments


def _resolve_parent(
    msg: ParsedMessage, by_id: dict[str, ParsedMessage]
) -> str | None:
    """Return the parent Message-ID present in this mbox, or None if root."""
    if msg.in_reply_to and msg.in_reply_to in by_id and msg.in_reply_to != msg.message_id:
        return msg.in_reply_to
    # Walk references right-to-left (nearest ancestor first).
    for ref in reversed(msg.references):
        if ref in by_id and ref != msg.message_id:
            return ref
    return None


def _find_root(mid: str, parent: dict[str, str | None]) -> str:
    """Walk parent chain. On cycle, return lexicographically smallest visited mid."""
    seen: list[str] = []
    cur: str | None = mid
    while cur is not None:
        if cur in seen:
            # Cycle — pick deterministic root.
            cycle = seen[seen.index(cur):]
            return min(cycle)
        seen.append(cur)
        cur = parent.get(cur)
    return seen[-1]


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
