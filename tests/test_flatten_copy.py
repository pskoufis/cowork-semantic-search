"""Tests for scripts/flatten_copy.py — recursive flat-copy utility."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _write(p: Path, content: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _runlog_items(dst: Path) -> list[dict]:
    """Item records from the most recent flatten_copy run-log under ``dst``."""
    logs = sorted((dst / "_runlogs").glob("flatten_copy-*.jsonl"))
    assert logs, "no run-log written"
    items = []
    for line in logs[-1].read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["event"] == "item":
            items.append(rec)
    return items


def _files_only(d: Path) -> dict[str, str]:
    """Map of filename -> content for regular files directly under ``d``
    (ignores the _runlogs/ directory)."""
    return {p.name: p.read_text() for p in d.iterdir() if p.is_file()}


def test_flat_copy_of_top_level_files(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "b.txt", "B")

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / "a.txt").read_text() == "A"
    assert (dst / "b.txt").read_text() == "B"


def test_recursive_walk_flattens_subdirs(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "top.txt", "T")
    _write(src / "sub" / "deep.txt", "D")
    _write(src / "sub" / "nested" / "deeper.txt", "DD")

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / "top.txt").read_text() == "T"
    assert (dst / "deep.txt").read_text() == "D"
    assert (dst / "deeper.txt").read_text() == "DD"
    # No subdirs should exist in the destination (the run-log sidecar aside).
    assert not any(p.is_dir() for p in dst.iterdir() if p.name != "_runlogs")


def test_name_collision_uses_underscore_suffix(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a" / "file.pdf", "first")
    _write(src / "b" / "file.pdf", "second")
    _write(src / "c" / "file.pdf", "third")

    rc = main([str(src), str(dst)])
    assert rc == 0
    contents = sorted((p.name, p.read_text()) for p in dst.iterdir() if p.is_file())
    names = [n for n, _ in contents]
    assert "file.pdf" in names
    assert "file_1.pdf" in names
    assert "file_2.pdf" in names
    # No file was overwritten — three distinct payloads survived.
    assert sorted(c for _, c in contents) == ["first", "second", "third"]


def test_collision_on_file_without_extension(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a" / "README", "first")
    _write(src / "b" / "README", "second")

    rc = main([str(src), str(dst)])
    assert rc == 0
    names = sorted(p.name for p in dst.iterdir() if p.is_file())
    assert names == ["README", "README_1"]


def test_creates_target_if_missing(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    _write(src / "x.txt", "X")
    dst = tmp_path / "deep" / "new" / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / "x.txt").is_file()


def test_dotfiles_included(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / ".env", "secret")
    _write(src / "sub" / ".hidden", "h")

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / ".env").read_text() == "secret"
    assert (dst / ".hidden").read_text() == "h"


def test_symlinks_skipped(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    src.mkdir()
    real = _write(src / "real.txt", "R")
    link = src / "link.txt"
    os.symlink(real, link)
    dst = tmp_path / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert (dst / "real.txt").read_text() == "R"
    assert not (dst / "link.txt").exists()


def test_missing_source_exits_nonzero(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    rc = main([str(tmp_path / "no-such"), str(tmp_path / "dst")])
    assert rc != 0
    assert "not found" in capsys.readouterr().err.lower()


def test_source_is_file_exits_nonzero(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    f = tmp_path / "single.txt"
    f.write_text("hi")
    rc = main([str(f), str(tmp_path / "dst")])
    assert rc != 0


def test_target_inside_source_refused(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    _write(src / "a.txt", "A")
    dst = src / "out"

    rc = main([str(src), str(dst)])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "inside" in err or "within" in err


def test_empty_source_exits_zero(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "0" in captured.err or "0" in captured.out


def test_preserves_mtime(tmp_path: Path) -> None:
    """shutil.copy2 should preserve mtime — verify via a manually-set value."""
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    src.mkdir()
    f = _write(src / "x.txt", "X")
    target_ts = 1_700_000_000
    os.utime(f, (target_ts, target_ts))
    dst = tmp_path / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    assert int((dst / "x.txt").stat().st_mtime) == target_ts


def test_prints_found_header_with_count(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "sub" / "b.txt", "B")

    rc = main([str(src), str(dst)])
    assert rc == 0
    err = capsys.readouterr().err
    assert re.search(r"Found 2 regular file\(s\) under .*src", err), err


def test_prints_per_item_progress_line(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "sub" / "b.txt", "B")

    rc = main([str(src), str(dst)])
    assert rc == 0
    err = capsys.readouterr().err
    # Sorted by relative path: a.txt comes before sub/b.txt.
    assert re.search(r"\[1/2\] copying a\.txt -> a\.txt", err), err
    assert re.search(r"\[2/2\] copying sub/b\.txt -> b\.txt", err), err


def test_progress_line_shows_collision_rename(tmp_path: Path, capsys) -> None:
    """When the destination name needs a `_1` suffix, the per-item line
    surfaces the renamed target so the user can see what was written."""
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a" / "file.pdf", "first")
    _write(src / "b" / "file.pdf", "second")

    rc = main([str(src), str(dst)])
    assert rc == 0
    err = capsys.readouterr().err
    assert re.search(r"\[1/2\] copying a/file\.pdf -> file\.pdf\b", err), err
    assert re.search(r"\[2/2\] copying b/file\.pdf -> file_1\.pdf\b", err), err


def test_per_file_failure_does_not_abort_batch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A copy failure on one file must be logged with the path, the batch
    must keep going, and the exit code must be non-zero."""
    import shutil

    from scripts import flatten_copy

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "good1.txt", "G1")
    _write(src / "bad.txt", "BAD")
    _write(src / "good2.txt", "G2")

    real_copy2 = shutil.copy2

    def flaky(srcp, dstp, *args, **kwargs):
        if Path(srcp).name == "bad.txt":
            raise OSError("intentional test failure")
        return real_copy2(srcp, dstp, *args, **kwargs)

    monkeypatch.setattr(flatten_copy.shutil, "copy2", flaky)

    rc = flatten_copy.main([str(src), str(dst)])
    assert rc != 0
    # Healthy files still copied.
    assert (dst / "good1.txt").read_text() == "G1"
    assert (dst / "good2.txt").read_text() == "G2"
    assert not (dst / "bad.txt").exists()
    err = capsys.readouterr().err
    assert "bad.txt" in err
    assert "intentional test failure" in err
    # Final summary distinguishes succeeded vs failed.
    assert re.search(r"Copied 2 file\(s\) into .*dst \(1 failed\)", err), err


def test_summary_includes_elapsed_seconds(tmp_path: Path, capsys) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    src.mkdir()
    _write(src / "a.txt", "A")
    dst = tmp_path / "dst"

    rc = main([str(src), str(dst)])
    assert rc == 0
    err = capsys.readouterr().err
    assert re.search(r"Copied 1 file\(s\) into .*dst .*in \d+\.\d+s", err), err


# -- run-log + idempotent re-run -----------------------------------------


def test_runlog_records_successful_copies(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "sub" / "b.txt", "B")

    assert main([str(src), str(dst)]) == 0
    items = _runlog_items(dst)
    by_input = {it["input"]: it for it in items}
    assert by_input["a.txt"]["status"] == "ok"
    assert by_input["sub/b.txt"]["status"] == "ok"
    # Output paths are recorded so the ledger can map input -> output.
    assert by_input["a.txt"]["output"].endswith("a.txt")


def test_rerun_is_idempotent_no_duplicates(tmp_path: Path, capsys) -> None:
    """A second run over the same input must skip everything — no `_1`/`_2`
    duplicates, output tree unchanged."""
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")
    _write(src / "sub" / "b.txt", "B")

    assert main([str(src), str(dst)]) == 0
    capsys.readouterr()
    before = _files_only(dst)

    assert main([str(src), str(dst)]) == 0
    err = capsys.readouterr().err
    after = _files_only(dst)

    assert after == before  # no new files, no duplicates
    assert "skip" in err.lower()
    items = _runlog_items(dst)
    assert {it["status"] for it in items} == {"skip"}


def test_collision_between_distinct_inputs_still_renamed_on_first_run(
    tmp_path: Path,
) -> None:
    """Idempotency must not break genuine same-name collisions between two
    *different* inputs in a single run — they still get `_1`."""
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a" / "file.pdf", "first")
    _write(src / "b" / "file.pdf", "second")

    assert main([str(src), str(dst)]) == 0
    names = sorted(_files_only(dst))
    assert names == ["file.pdf", "file_1.pdf"]


def test_retry_after_fix_reprocesses_only_failed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Run 1: one file fails. Fix the cause, re-run the same command — only the
    previously-failed file is copied; the rest are skipped."""
    import shutil

    from scripts import flatten_copy

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "good1.txt", "G1")
    _write(src / "bad.txt", "BAD")
    _write(src / "good2.txt", "G2")

    real_copy2 = shutil.copy2

    def flaky(srcp, dstp, *args, **kwargs):
        if Path(srcp).name == "bad.txt":
            raise OSError("intentional test failure")
        return real_copy2(srcp, dstp, *args, **kwargs)

    monkeypatch.setattr(flatten_copy.shutil, "copy2", flaky)
    assert flatten_copy.main([str(src), str(dst)]) != 0
    assert not (dst / "bad.txt").exists()
    capsys.readouterr()

    # "Fix" the cause: restore real copy2 and re-run.
    monkeypatch.setattr(flatten_copy.shutil, "copy2", real_copy2)
    assert flatten_copy.main([str(src), str(dst)]) == 0

    assert (dst / "bad.txt").read_text() == "BAD"
    by_input = {it["input"]: it["status"] for it in _runlog_items(dst)}
    assert by_input["bad.txt"] == "ok"
    assert by_input["good1.txt"] == "skip"
    assert by_input["good2.txt"] == "skip"


def test_force_reprocesses_everything(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")

    assert main([str(src), str(dst)]) == 0
    assert main([str(src), str(dst), "--force"]) == 0
    # With --force the second run re-copies rather than skipping ...
    assert {it["status"] for it in _runlog_items(dst)} == {"ok"}
    # ... but overwrites in place — no a_1.txt duplicate.
    assert sorted(_files_only(dst)) == ["a.txt"]


def test_edited_source_reprocesses_in_place(tmp_path: Path) -> None:
    """Editing a source file invalidates the skip; the re-copy overwrites the
    same destination rather than creating a_1.txt."""
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    f = _write(src / "a.txt", "v1")

    assert main([str(src), str(dst)]) == 0
    f.write_text("v2 longer body")  # changes mtime + size

    assert main([str(src), str(dst)]) == 0
    assert sorted(_files_only(dst)) == ["a.txt"]  # no duplicate
    assert (dst / "a.txt").read_text() == "v2 longer body"


def test_no_runlog_flag_writes_no_log(tmp_path: Path) -> None:
    from scripts.flatten_copy import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.txt", "A")

    assert main([str(src), str(dst), "--no-runlog"]) == 0
    assert not (dst / "_runlogs").exists()
