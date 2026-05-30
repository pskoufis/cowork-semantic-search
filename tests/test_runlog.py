"""Tests for scripts/_runlog.py — structured JSONL run-log + idempotency ledger.

The run-log is the single artifact behind three needs: post-hoc analysis (a
stable machine-readable record per item), idempotent re-runs (a ledger that lets
a script skip already-done inputs), and surfacing failures. These tests pin the
schema and the ledger semantics that the scripts rely on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _read_events(log_dir: Path, script: str) -> list[dict]:
    """All JSON objects across every run-log file for ``script``, in file +
    line order."""
    events: list[dict] = []
    for f in sorted(log_dir.glob(f"{script}-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _make_input(p: Path, content: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_writes_run_start_item_run_end_with_schema(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")
    target = out / "a.done"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    rl = RunLog(
        "myscript", input_root=src, output_root=out, argv=["in", "out"]
    )
    rl.record(inp, kind="txt", output=target, status="ok", duration_s=1.5)
    rl.finish(exit_code=0)

    events = _read_events(out / "_runlogs", "myscript")
    assert [e["event"] for e in events] == ["run_start", "item", "run_end"]

    start = events[0]
    assert start["schema"] == 1
    assert start["script"] == "myscript"
    assert start["argv"] == ["in", "out"]
    assert start["input_root"] == str(src)
    assert start["output_root"] == str(out)
    assert "run_id" in start and "ts" in start

    item = events[1]
    assert item["input"] == "a.txt"  # relative to input_root
    assert item["kind"] == "txt"
    assert item["status"] == "ok"
    assert item["output"] == str(target)
    assert item["duration_s"] == 1.5
    assert item["error_type"] is None and item["error"] is None
    assert isinstance(item["input_mtime"], (int, float))
    assert item["input_size"] == len("A")

    end = events[2]
    assert end["totals"] == {"ok": 1, "skip": 0, "fail": 1 - 1}  # 0 fail
    assert end["exit_code"] == 0


def test_failure_record_captures_exception_type_and_message(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "bad.zip", "Z")

    rl = RunLog("zz", input_root=src, output_root=out, argv=[])
    rl.record(
        inp,
        kind="zip",
        output=None,
        status="fail",
        error=ValueError("boom"),
    )
    rl.finish(exit_code=1)

    item = [e for e in _read_events(out / "_runlogs", "zz") if e["event"] == "item"][0]
    assert item["status"] == "fail"
    assert item["error_type"] == "ValueError"
    assert item["error"] == "boom"


def test_ledger_marks_input_done(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")
    target = out / "a.out"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    # Run 1: record success.
    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(inp, kind="txt", output=target, status="ok")
    rl1.finish(exit_code=0)

    # Run 2: a fresh RunLog over the same dir sees the input as done.
    rl2 = RunLog("s", input_root=src, output_root=out, argv=[])
    assert rl2.done_output(inp) == target


def test_changed_input_invalidates_ledger(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")
    target = out / "a.out"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(inp, kind="txt", output=target, status="ok")
    rl1.finish(exit_code=0)

    # Edit the input so its size/mtime change.
    inp.write_text("A much longer body now")

    rl2 = RunLog("s", input_root=src, output_root=out, argv=[])
    assert rl2.done_output(inp) is None


def test_missing_output_invalidates_ledger(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")
    target = out / "a.out"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(inp, kind="txt", output=target, status="ok")
    rl1.finish(exit_code=0)

    target.unlink()  # output gone — must reprocess

    rl2 = RunLog("s", input_root=src, output_root=out, argv=[])
    assert rl2.done_output(inp) is None


def test_failed_input_is_not_done(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")

    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(inp, kind="txt", output=None, status="fail", error=OSError("x"))
    rl1.finish(exit_code=1)

    rl2 = RunLog("s", input_root=src, output_root=out, argv=[])
    assert rl2.done_output(inp) is None


def test_latest_record_wins_in_ledger(tmp_path: Path) -> None:
    """A later successful run supersedes an earlier failure for the same input."""
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")
    target = out / "a.out"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(inp, kind="txt", output=None, status="fail", error=OSError("x"))
    rl1.finish(exit_code=1)

    rl2 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl2.record(inp, kind="txt", output=target, status="ok")
    rl2.finish(exit_code=0)

    rl3 = RunLog("s", input_root=src, output_root=out, argv=[])
    assert rl3.done_output(inp) == target


def test_planned_output_returns_last_output_even_when_stale(tmp_path: Path) -> None:
    """planned_output gives the prior output for a known input regardless of
    freshness — so a reprocess (force, or changed input) overwrites in place
    instead of allocating a duplicate name."""
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.jpg", "A")
    target = out / "a.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("pdf")

    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(inp, kind="img", output=target, status="ok")
    rl1.finish(exit_code=0)

    inp.write_text("A changed")  # stale now — done_output would say None
    rl2 = RunLog("s", input_root=src, output_root=out, argv=[])
    assert rl2.done_output(inp) is None
    assert rl2.planned_output(inp) == target


def test_planned_output_none_for_failed_or_unseen(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    failed = _make_input(src / "bad.jpg", "B")
    unseen = _make_input(src / "new.jpg", "N")

    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(failed, kind="img", output=None, status="fail", error=OSError("x"))
    rl1.finish(exit_code=1)

    rl2 = RunLog("s", input_root=src, output_root=out, argv=[])
    assert rl2.planned_output(failed) is None
    assert rl2.planned_output(unseen) is None


def test_force_ignores_ledger(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")
    target = out / "a.out"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(inp, kind="txt", output=target, status="ok")
    rl1.finish(exit_code=0)

    rl2 = RunLog("s", input_root=src, output_root=out, argv=[], force=True)
    assert rl2.done_output(inp) is None


def test_custom_log_dir(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    logs = tmp_path / "elsewhere"
    inp = _make_input(src / "a.txt", "A")
    target = out / "a.out"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    rl = RunLog("s", input_root=src, output_root=out, argv=[], log_dir=logs)
    rl.record(inp, kind="txt", output=target, status="ok")
    rl.finish(exit_code=0)

    assert list(logs.glob("s-*.jsonl"))
    assert not (out / "_runlogs").exists()


def test_disabled_is_a_noop(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")

    rl = RunLog("s", input_root=src, output_root=out, argv=[], enabled=False)
    assert rl.done_output(inp) is None
    rl.record(inp, kind="txt", output=None, status="ok")  # must not raise
    rl.finish(exit_code=0)

    assert not (out / "_runlogs").exists()


def test_run_end_totals_count_each_status(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    a = _make_input(src / "a.txt", "A")
    b = _make_input(src / "b.txt", "B")
    c = _make_input(src / "c.txt", "C")
    target = out / "a.out"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    rl = RunLog("s", input_root=src, output_root=out, argv=[])
    rl.record(a, kind="txt", output=target, status="ok")
    rl.record(b, kind="txt", output=target, status="skip")
    rl.record(c, kind="txt", output=None, status="fail", error=OSError("x"))
    rl.finish(exit_code=1)

    end = [e for e in _read_events(out / "_runlogs", "s") if e["event"] == "run_end"][0]
    assert end["totals"] == {"ok": 1, "skip": 1, "fail": 1}


def test_context_manager_writes_run_end_on_exception(tmp_path: Path) -> None:
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    _make_input(src / "a.txt", "A")

    try:
        with RunLog("s", input_root=src, output_root=out, argv=[]):
            raise RuntimeError("kaboom")
    except RuntimeError:
        pass

    events = _read_events(out / "_runlogs", "s")
    assert events[-1]["event"] == "run_end"
    assert events[-1]["exit_code"] == 1


def test_two_runs_same_second_do_not_clobber(tmp_path: Path) -> None:
    """Distinct runs must land in distinct files even back-to-back, or the
    ledger history (and analysis) loses records."""
    from scripts._runlog import RunLog

    src = tmp_path / "in"
    out = tmp_path / "out"
    inp = _make_input(src / "a.txt", "A")
    target = out / "a.out"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("done")

    rl1 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl1.record(inp, kind="txt", output=target, status="ok")
    rl1.finish(exit_code=0)

    rl2 = RunLog("s", input_root=src, output_root=out, argv=[])
    rl2.record(inp, kind="txt", output=target, status="skip")
    rl2.finish(exit_code=0)

    assert len(list((out / "_runlogs").glob("s-*.jsonl"))) == 2
