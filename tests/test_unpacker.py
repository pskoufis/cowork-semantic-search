"""Tests for server.unpacker — the shared pst/mbox/msg unpack orchestration.

The three preprocessing passes used to live inline in
``server.indexer.index_folder`` as ``_ensure_*_unpacked`` helpers. They were
extracted into ``server.unpacker.run_unpack_passes`` so the CLI's ``unpack``
verb can call them directly with progress reporting, while the indexer
continues to call them when ``unpack_first=True``.

These tests cover the orchestration: return shape, per-phase progress
callback, exclusion + recursive semantics, idempotency, and per-archive
error isolation. The underlying ``ensure_unpacked`` functions are tested
separately in tests/test_mbox_unpack.py, tests/test_pst_unpack.py,
tests/test_msg_unpack.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from tests.test_mbox_messages import _make_entry, _make_mbox


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


def test_run_unpack_passes_returns_dict_with_three_phases(tmp_path: Path) -> None:
    from server.unpacker import run_unpack_passes

    result = run_unpack_passes(tmp_path)

    assert set(result.keys()) == {"pst", "mbox", "msg"}
    for phase, stats in result.items():
        assert set(stats.keys()) >= {"discovered", "failed", "elapsed_s"}
        assert stats["discovered"] == 0
        assert stats["failed"] == 0
        assert stats["elapsed_s"] >= 0.0


def test_run_unpack_passes_counts_mbox_archives(tmp_path: Path) -> None:
    from server.unpacker import run_unpack_passes

    _make_mbox(tmp_path, _make_entry({"From": "a@x", "Subject": "one"}, "body"), name="a.mbox")
    _make_mbox(tmp_path, _make_entry({"From": "b@x", "Subject": "two"}, "body"), name="b.mbox")

    result = run_unpack_passes(tmp_path)

    assert result["mbox"]["discovered"] == 2
    assert result["mbox"]["failed"] == 0
    assert (tmp_path / "a_unpacked").is_dir()
    assert (tmp_path / "b_unpacked").is_dir()


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def test_run_unpack_passes_invokes_progress_callback(tmp_path: Path) -> None:
    from server.unpacker import run_unpack_passes

    _make_mbox(tmp_path, _make_entry({"From": "a@x", "Subject": "one"}, "body"), name="a.mbox")
    _make_mbox(tmp_path, _make_entry({"From": "b@x", "Subject": "two"}, "body"), name="b.mbox")

    events: list[tuple[str, int, int]] = []

    def cb(phase: str, processed: int, total: int) -> None:
        events.append((phase, processed, total))

    run_unpack_passes(tmp_path, progress_callback=cb)

    # Each phase fires at least one event. The mbox phase fires
    # one event per archive (start) plus a final completion event with
    # processed == total.
    phases_seen = {p for p, _, _ in events}
    assert phases_seen == {"pst", "mbox", "msg"}

    mbox_events = [e for e in events if e[0] == "mbox"]
    assert mbox_events[-1] == ("mbox", 2, 2), (
        "final event for a phase must report processed == total"
    )
    # Two archives, so we expect start events at (0, 2) and (1, 2)
    # plus the final (2, 2). Order matters: processed is monotonic.
    processed_values = [p for _, p, _ in mbox_events]
    assert processed_values == sorted(processed_values)


def test_run_unpack_passes_callback_fires_completion_for_empty_phase(
    tmp_path: Path,
) -> None:
    """Even with zero archives, a phase fires a single (0, 0) completion event
    so the CLI's progress UI can close out the bar cleanly."""
    from server.unpacker import run_unpack_passes

    events: list[tuple[str, int, int]] = []

    def cb(phase: str, processed: int, total: int) -> None:
        events.append((phase, processed, total))

    run_unpack_passes(tmp_path, progress_callback=cb)

    for phase in ("pst", "mbox", "msg"):
        phase_events = [e for e in events if e[0] == phase]
        assert phase_events == [(phase, 0, 0)], (
            f"phase {phase!r} with no archives must emit exactly one (0, 0) event, "
            f"got {phase_events!r}"
        )


# ---------------------------------------------------------------------------
# Exclusion + recursive semantics — same contract as the previous inline
# ``_ensure_*_unpacked`` helpers, just routed through the new orchestrator.
# ---------------------------------------------------------------------------


def test_run_unpack_passes_honours_exclusions(tmp_path: Path) -> None:
    from server.exclusions import ExclusionRules
    from server.unpacker import run_unpack_passes

    _make_mbox(tmp_path, _make_entry({"From": "a@x", "Subject": "keep"}, "b"), name="keep.mbox")
    skip_dir = tmp_path / "skip"
    skip_dir.mkdir()
    _make_mbox(skip_dir, _make_entry({"From": "b@x", "Subject": "drop"}, "b"), name="drop.mbox")

    exclusions = ExclusionRules.load(
        tmp_path, extra_patterns=["skip/**"], db_dir=str(tmp_path / "lancedb")
    )

    result = run_unpack_passes(tmp_path, exclusions=exclusions)

    # Only the un-excluded mbox is unpacked.
    assert result["mbox"]["discovered"] == 1
    assert (tmp_path / "keep_unpacked").is_dir()
    assert not (skip_dir / "drop_unpacked").exists()


def test_run_unpack_passes_honours_recursive_false(tmp_path: Path) -> None:
    from server.unpacker import run_unpack_passes

    _make_mbox(tmp_path, _make_entry({"From": "a@x", "Subject": "top"}, "b"), name="top.mbox")
    nested = tmp_path / "nested"
    nested.mkdir()
    _make_mbox(nested, _make_entry({"From": "b@x", "Subject": "deep"}, "b"), name="deep.mbox")

    result = run_unpack_passes(tmp_path, recursive=False)

    assert result["mbox"]["discovered"] == 1
    assert (tmp_path / "top_unpacked").is_dir()
    assert not (nested / "deep_unpacked").exists()


# ---------------------------------------------------------------------------
# Idempotency — second run must not re-unpack already-fresh archives.
# ---------------------------------------------------------------------------


def test_run_unpack_passes_idempotent_on_second_run(tmp_path: Path) -> None:
    from server.unpacker import run_unpack_passes

    _make_mbox(tmp_path, _make_entry({"From": "a@x", "Subject": "one"}, "body"), name="a.mbox")

    first = run_unpack_passes(tmp_path)
    assert first["mbox"]["discovered"] == 1
    unpacked = tmp_path / "a_unpacked"
    assert unpacked.is_dir()
    first_mtime = max(p.stat().st_mtime_ns for p in unpacked.rglob("*") if p.is_file())

    second = run_unpack_passes(tmp_path)

    # The mbox still shows up as discovered — discovery counts files we
    # checked, not files we actually re-extracted. The mtime guard inside
    # the per-archive unpacker is what makes this cheap; here we assert that
    # nothing was rewritten.
    assert second["mbox"]["discovered"] == 1
    later_mtime = max(p.stat().st_mtime_ns for p in unpacked.rglob("*") if p.is_file())
    assert later_mtime == first_mtime, (
        "second pass must not rewrite an up-to-date unpacked tree"
    )


# ---------------------------------------------------------------------------
# Per-archive error isolation — one broken archive must not abort the run.
# ---------------------------------------------------------------------------


def test_run_unpack_passes_isolates_failing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import unpacker as unpacker_mod
    from server.unpacker import run_unpack_passes

    a = _make_mbox(tmp_path, _make_entry({"From": "a@x", "Subject": "ok"}, "b"), name="a.mbox")
    b = _make_mbox(tmp_path, _make_entry({"From": "b@x", "Subject": "boom"}, "b"), name="b.mbox")

    real_ensure = unpacker_mod.ensure_mbox_unpacked

    def flaky(path: Path) -> object:
        if path == b:
            raise RuntimeError("simulated mbox failure")
        return real_ensure(path)

    monkeypatch.setattr(unpacker_mod, "ensure_mbox_unpacked", flaky)

    result = run_unpack_passes(tmp_path)

    assert result["mbox"]["discovered"] == 2
    assert result["mbox"]["failed"] == 1
    # Good archive still unpacked despite the bad sibling.
    assert (tmp_path / "a_unpacked").is_dir()


# ---------------------------------------------------------------------------
# Phase ordering — pst runs before mbox before msg so .msg attachments
# surfaced by mbox unpacking and .mbox files surfaced inside a pst extract
# are caught by the subsequent passes.
# ---------------------------------------------------------------------------


def test_run_unpack_passes_runs_phases_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import unpacker as unpacker_mod
    from server.unpacker import run_unpack_passes

    order: list[str] = []

    real_ensure_pst = unpacker_mod.ensure_pst_unpacked
    real_ensure_mbox = unpacker_mod.ensure_mbox_unpacked
    real_ensure_msg = unpacker_mod.ensure_msg_unpacked

    # Even with no archives discovered, the phase loop still runs; record
    # the phase ordering by tagging each fake unpacker. Done by tagging the
    # progress callback so phases are observed in the order they're started.
    seen_phases: list[str] = []

    def cb(phase: str, processed: int, total: int) -> None:
        if phase not in seen_phases:
            seen_phases.append(phase)

    run_unpack_passes(tmp_path, progress_callback=cb)

    assert seen_phases == ["pst", "mbox", "msg"], (
        "phase order is load-bearing: a .pst extract can yield .mbox files, "
        "and a .mbox extract can yield .msg files. Running them in this order "
        "means each subsequent phase sees archives the prior phase produced."
    )
