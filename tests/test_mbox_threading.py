"""Tests for mbox_handling.threading — headers-only thread assignment."""

from __future__ import annotations

import pytest

from mbox_handling.messages import ParsedMessage


def _msg(
    index: int,
    message_id: str | None,
    in_reply_to: str | None = None,
    references: tuple[str, ...] = (),
    date: str = "",
) -> ParsedMessage:
    """Build a minimal ParsedMessage for threading tests."""
    return ParsedMessage(
        index=index,
        message_id=message_id,
        in_reply_to=in_reply_to,
        references=references,
        from_addr=None,
        to=(),
        cc=(),
        date=date,
        subject=None,
        body="",
        attachments=(),
        error=None,
    )


def test_single_message_is_its_own_thread_and_orphan() -> None:
    from mbox_handling.threading import compute_thread_assignments

    msgs = [_msg(1, "<a@x>")]
    out = compute_thread_assignments(msgs)
    assert len(out) == 1
    assignment = out["<a@x>"]
    assert assignment.is_orphan is True
    assert assignment.root_message_id == "<a@x>"
    assert assignment.position_in_thread == 0
    assert len(assignment.thread_id) == 16
    assert all(c in "0123456789abcdef" for c in assignment.thread_id)


def test_reply_groups_with_parent_via_in_reply_to() -> None:
    from mbox_handling.threading import compute_thread_assignments

    msgs = [
        _msg(1, "<root@x>", date="2024-01-01"),
        _msg(2, "<reply@x>", in_reply_to="<root@x>", date="2024-01-02"),
    ]
    out = compute_thread_assignments(msgs)
    assert out["<root@x>"].thread_id == out["<reply@x>"].thread_id
    assert out["<root@x>"].is_orphan is False
    assert out["<reply@x>"].is_orphan is False
    assert out["<root@x>"].position_in_thread == 0
    assert out["<reply@x>"].position_in_thread == 1


def test_walks_references_when_in_reply_to_unresolvable() -> None:
    from mbox_handling.threading import compute_thread_assignments

    msgs = [
        _msg(1, "<root@x>"),
        _msg(
            2,
            "<reply@x>",
            in_reply_to="<missing@x>",  # not in mbox
            references=("<root@x>", "<missing@x>"),
        ),
    ]
    out = compute_thread_assignments(msgs)
    assert out["<root@x>"].thread_id == out["<reply@x>"].thread_id


def test_external_parent_reference_creates_new_root() -> None:
    from mbox_handling.threading import compute_thread_assignments

    msgs = [
        _msg(1, "<orphan@x>", in_reply_to="<external@somewhere>"),
    ]
    out = compute_thread_assignments(msgs)
    assert out["<orphan@x>"].is_orphan is True
    assert out["<orphan@x>"].root_message_id == "<orphan@x>"


def test_breaks_cycles_safely() -> None:
    """A→B and B→A in In-Reply-To must not loop. Lowest Message-ID wins as root."""
    from mbox_handling.threading import compute_thread_assignments

    msgs = [
        _msg(1, "<b@x>", in_reply_to="<a@x>"),
        _msg(2, "<a@x>", in_reply_to="<b@x>"),
    ]
    out = compute_thread_assignments(msgs)
    # Same thread regardless
    assert out["<a@x>"].thread_id == out["<b@x>"].thread_id
    # Root is the lexicographically smallest of the cycle
    assert out["<a@x>"].root_message_id == "<a@x>"
    assert out["<b@x>"].root_message_id == "<a@x>"


def test_groups_three_message_chain() -> None:
    from mbox_handling.threading import compute_thread_assignments

    msgs = [
        _msg(1, "<root@x>", date="2024-01-01"),
        _msg(2, "<r1@x>", in_reply_to="<root@x>", date="2024-01-02"),
        _msg(3, "<r2@x>", in_reply_to="<r1@x>", date="2024-01-03"),
    ]
    out = compute_thread_assignments(msgs)
    tid = out["<root@x>"].thread_id
    assert out["<r1@x>"].thread_id == tid
    assert out["<r2@x>"].thread_id == tid
    assert out["<root@x>"].position_in_thread == 0
    assert out["<r1@x>"].position_in_thread == 1
    assert out["<r2@x>"].position_in_thread == 2


def test_thread_id_is_stable_short_hash() -> None:
    from mbox_handling.threading import compute_thread_assignments

    msgs = [_msg(1, "<a@x>"), _msg(2, "<b@x>", in_reply_to="<a@x>")]
    out1 = compute_thread_assignments(msgs)
    out2 = compute_thread_assignments(msgs)
    assert out1["<a@x>"].thread_id == out2["<a@x>"].thread_id
    assert len(out1["<a@x>"].thread_id) == 16


def test_messages_without_message_id_are_each_own_orphan() -> None:
    from mbox_handling.threading import compute_thread_assignments

    msgs = [_msg(1, None), _msg(2, None)]
    out = compute_thread_assignments(msgs)
    assert "no-id-1" in out
    assert "no-id-2" in out
    assert out["no-id-1"].is_orphan is True
    assert out["no-id-2"].is_orphan is True
    assert out["no-id-1"].thread_id != out["no-id-2"].thread_id
