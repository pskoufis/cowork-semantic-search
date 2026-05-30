"""Structured JSONL run-log + idempotency ledger for the batch scripts.

Every unpack / convert / copy script under ``scripts/`` writes human-readable
progress to stderr (unchanged) *and* a machine-readable run-log via this module.
That one artifact serves three needs:

* **Analyze a run afterwards** — one JSON object per line, a stable schema shared
  by every script, so ``jq`` (or anything) can answer "which inputs failed, with
  what error class?" across a whole run.
* **Re-execute only what's left** — the log doubles as a ledger keyed by input
  identity. On a later run, an input whose last record was ``ok`` (with the input
  unchanged and its output still present) is reported done, so the script can
  skip it. Fix the cause, run the same command again, and only the previously
  failed / missing items are reprocessed.
* **Surface failures** — failed items carry the exception type and message.

This is deliberately dependency-free (no ``logging``, no third-party): a tiny
append-only JSONL writer is simpler to reason about and keeps the scripts'
``capsys``-based tests clean.

Schema (``schema: 1``)::

    {"event":"run_start","schema":1,"ts":..,"run_id":..,"script":..,
     "argv":[..],"input_root":..,"output_root":..}
    {"event":"item","ts":..,"run_id":..,"script":..,"input":<rel>,"input_abs":..,
     "input_mtime":..,"input_size":..,"kind":..,"output":..,
     "status":"ok"|"skip"|"fail","error_type":..,"error":..,"duration_s":..}
    {"event":"run_end","ts":..,"run_id":..,"script":..,
     "totals":{"ok":..,"skip":..,"fail":..},"elapsed_s":..,"exit_code":..}
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = 1
_DEFAULT_LOG_SUBDIR = "_runlogs"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared run-log flags to a script's argument parser."""
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for the JSONL run-log (default: <output>/_runlogs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess every input, ignoring the run-log ledger of past "
        "successes (no idempotent skipping).",
    )
    parser.add_argument(
        "--no-runlog",
        action="store_true",
        help="Disable the structured JSONL run-log entirely.",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_run_id() -> str:
    # Microsecond-precision timestamp so filename sort is reliably
    # chronological (the ledger's "latest record wins" depends on this), plus
    # a short random suffix so two runs in the same microsecond still land in
    # distinct files rather than merging their records.
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


class RunLog:
    """Append-only JSONL run-log with an idempotency ledger.

    Construct one per run (after argument validation, before the discovery
    loop). Call :meth:`done_output` to decide whether to skip an input,
    :meth:`record` once per processed input, and :meth:`finish` at the end.
    Also usable as a context manager so ``run_end`` is written even on an
    unhandled exception.
    """

    def __init__(
        self,
        script: str,
        *,
        input_root: Path | str,
        output_root: Path | str,
        argv: list[str],
        log_dir: Path | str | None = None,
        force: bool = False,
        enabled: bool = True,
    ) -> None:
        self.script = script
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.force = force
        self.enabled = enabled
        self._totals = {"ok": 0, "skip": 0, "fail": 0}
        self._ledger: dict[str, dict[str, Any]] = {}
        self._started = time.monotonic()
        self._finished = False
        self._fh = None

        if not enabled:
            return

        self.log_dir = (
            Path(log_dir)
            if log_dir is not None
            else self.output_root / _DEFAULT_LOG_SUBDIR
        )
        # Always load the ledger: ``done_output`` honours ``--force`` by
        # returning None (reprocess), but ``planned_output`` still needs the
        # prior input->output mapping so a forced reprocess overwrites in place
        # rather than minting a duplicate name.
        self._ledger = _load_ledger(self.log_dir, script)

        self.run_id = _new_run_id()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.log_dir / f"{script}-{self.run_id}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")
        self._write(
            {
                "event": "run_start",
                "schema": SCHEMA,
                "ts": _now_iso(),
                "run_id": self.run_id,
                "script": script,
                "argv": list(argv),
                "input_root": str(self.input_root),
                "output_root": str(self.output_root),
            }
        )

    # -- public API -------------------------------------------------------

    def done_output(self, input_path: Path | str) -> Path | None:
        """Return the recorded output path iff ``input_path`` is already done.

        Done means: the last ledger record for this input was ``ok``, the input
        is byte-identical by ``(mtime, size)`` to when it was recorded, and the
        recorded output still exists on disk. Otherwise ``None`` — process it.
        ``--force`` always returns ``None``.
        """
        if not self.enabled or self.force:
            return None
        rec = self._ledger.get(self._key(input_path))
        if not rec or rec.get("status") != "ok":
            return None
        out = rec.get("output")
        if not out:
            return None
        out_path = Path(out)
        if not out_path.exists():
            return None
        try:
            st = Path(input_path).stat()
        except OSError:
            return None
        if rec.get("input_mtime") != st.st_mtime or rec.get("input_size") != st.st_size:
            return None
        return out_path

    def planned_output(self, input_path: Path | str) -> Path | None:
        """Return the last output path recorded for this input, regardless of
        status or freshness, or ``None`` if it was never produced.

        Use this to reuse a stable output location when *re*processing an input
        (under ``--force`` or because the input changed), so the new result
        overwrites the old in place instead of getting a duplicate name.
        Failed items recorded no output, so this returns ``None`` for them.
        """
        if not self.enabled:
            return None
        rec = self._ledger.get(self._key(input_path))
        if rec and rec.get("output"):
            return Path(rec["output"])
        return None

    def record(
        self,
        input_path: Path | str,
        *,
        kind: str,
        output: Path | str | None,
        status: str,
        error: BaseException | str | None = None,
        duration_s: float | None = None,
    ) -> None:
        """Append one ``item`` record and update the running totals.

        ``status`` is ``ok`` / ``skip`` / ``fail``. ``error`` may be an
        exception (its class name and message are captured) or a string.
        """
        if status not in self._totals:
            raise ValueError(f"unknown status: {status!r}")
        self._totals[status] += 1
        if not self.enabled:
            return

        if isinstance(error, BaseException):
            error_type: str | None = type(error).__name__
            error_msg: str | None = str(error)
        elif error is None:
            error_type = error_msg = None
        else:
            error_type = None
            error_msg = str(error)

        try:
            st = Path(input_path).stat()
            mtime: float | None = st.st_mtime
            size: int | None = st.st_size
        except OSError:
            mtime = size = None

        self._write(
            {
                "event": "item",
                "ts": _now_iso(),
                "run_id": self.run_id,
                "script": self.script,
                "input": self._key(input_path),
                "input_abs": str(Path(input_path)),
                "input_mtime": mtime,
                "input_size": size,
                "kind": kind,
                "output": str(output) if output is not None else None,
                "status": status,
                "error_type": error_type,
                "error": error_msg,
                "duration_s": duration_s,
            }
        )

    def finish(self, *, exit_code: int) -> None:
        """Append the ``run_end`` summary and close the file. Idempotent."""
        if self._finished:
            return
        self._finished = True
        if not self.enabled or self._fh is None:
            return
        self._write(
            {
                "event": "run_end",
                "ts": _now_iso(),
                "run_id": self.run_id,
                "script": self.script,
                "totals": dict(self._totals),
                "elapsed_s": round(time.monotonic() - self._started, 3),
                "exit_code": exit_code,
            }
        )
        self._fh.close()

    # -- context manager safety net --------------------------------------

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._finished:
            self.finish(exit_code=1 if exc_type is not None else 0)
        return False

    # -- internals --------------------------------------------------------

    def _key(self, input_path: Path | str) -> str:
        p = Path(input_path)
        try:
            return p.relative_to(self.input_root).as_posix()
        except ValueError:
            return p.as_posix()

    def _write(self, obj: dict[str, Any]) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()


def _load_ledger(log_dir: Path, script: str) -> dict[str, dict[str, Any]]:
    """Build ``{input_key -> last item record}`` from every prior run-log file
    for ``script`` in ``log_dir``. Files are read in name order (run_id starts
    with a UTC timestamp, so this is chronological) and later records win."""
    ledger: dict[str, dict[str, Any]] = {}
    if not log_dir.exists():
        return ledger
    for f in sorted(log_dir.glob(f"{script}-*.jsonl")):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "item":
                ledger[rec["input"]] = rec
    return ledger
