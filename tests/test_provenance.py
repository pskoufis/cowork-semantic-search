"""Tests for server.provenance — map an unpacked file back to its source archive.

Fixtures are built by hand from the *output* side: a source tree of empty
archive files, a target tree of unpacked outputs, and (for the runlog cases) a
``_runlogs/*.jsonl`` ledger in the schema written by ``scripts/_runlog.py``.
No real .pst/.mbox/.msg parsing is exercised here — the unit under test is the
reverse-lookup logic, observable purely through paths + the ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.provenance import trace_source


def _write_runlog(target_root: Path, script: str, items: list[dict]) -> Path:
    """Write a minimal JSONL run-log under ``<target_root>/_runlogs`` matching
    the schema of scripts/_runlog.py (a run_start meta line + item lines)."""
    log_dir = target_root / "_runlogs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{script}-20240101T000000000000Z-abc123.jsonl"
    lines = [
        {
            "event": "run_start",
            "schema": 1,
            "run_id": "r1",
            "script": script,
            "input_root": str(target_root),  # value irrelevant to the lookup
            "output_root": str(target_root),
        }
    ]
    for it in items:
        lines.append(
            {
                "event": "item",
                "run_id": "r1",
                "script": script,
                "input": it["input_abs"],
                "input_abs": it["input_abs"],
                "kind": it["kind"],
                "output": it["output"],
                "status": it.get("status", "ok"),
            }
        )
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    return path


def _pst_txt(folder: str) -> str:
    return (
        "From: alice@example.com\nTo: \nCc: \nSubject: Hi\n"
        "Date: 2024-01-01\nMessage-ID: \n"
        f"Folder: {folder}\n\nbody\n"
    )


# --- PST -------------------------------------------------------------------


def test_pst_body_via_runlog(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    pst = src / "sub" / "foo.pst"
    pst.parent.mkdir(parents=True)
    pst.write_bytes(b"")
    unpacked = dst / "sub" / "foo_unpacked" / "Top/Inbox"
    unpacked.mkdir(parents=True)
    txt = unpacked / "msg-0007_20240101_alice.txt"
    txt.write_text(_pst_txt("Top/Inbox"), encoding="utf-8")

    _write_runlog(
        dst,
        "unpack_pst_folder",
        [{"input_abs": str(pst), "kind": "pst", "output": str(dst / "sub" / "foo_unpacked")}],
    )

    trace = trace_source(txt, src, dst)
    assert trace.source_archive == pst.resolve()
    assert trace.archive_type == "pst"
    assert trace.archive_exists is True
    assert trace.method == "runlog"
    assert trace.internal_folder == "Top/Inbox"
    assert trace.message_number == 7
    assert trace.attachment_of is None


def test_pst_attachment_via_runlog(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    pst = src / "foo.pst"
    src.mkdir(parents=True)
    pst.write_bytes(b"")
    folder = dst / "foo_unpacked" / "Top/Inbox"
    folder.mkdir(parents=True)
    txt = folder / "msg-0003_20240101_alice.txt"
    txt.write_text(_pst_txt("Top/Inbox"), encoding="utf-8")
    att = folder / "attachments" / "msg-0003" / "report.pdf"
    att.parent.mkdir(parents=True)
    att.write_bytes(b"%PDF")

    _write_runlog(
        dst,
        "unpack_pst_folder",
        [{"input_abs": str(pst), "kind": "pst", "output": str(dst / "foo_unpacked")}],
    )

    trace = trace_source(att, src, dst)
    assert trace.source_archive == pst.resolve()
    assert trace.archive_type == "pst"
    assert trace.message_number == 3
    assert trace.internal_folder == "Top/Inbox"
    assert trace.attachment_of == txt.resolve()


def test_pst_path_fallback_without_runlog(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    pst = src / "a/b" / "mail.pst"
    pst.parent.mkdir(parents=True)
    pst.write_bytes(b"")
    folder = dst / "a/b" / "mail_unpacked" / "Top"
    folder.mkdir(parents=True)
    txt = folder / "msg-0001_20240101_alice.txt"
    txt.write_text(_pst_txt("Top"), encoding="utf-8")

    trace = trace_source(txt, src, dst)
    assert trace.source_archive == pst.resolve()
    assert trace.archive_type == "pst"
    assert trace.method == "path"
    assert trace.archive_exists is True


# --- MSG -------------------------------------------------------------------


def test_msg_body_via_runlog(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    msg = src / "sub" / "note.msg"
    msg.parent.mkdir(parents=True)
    msg.write_bytes(b"")
    out_dir = dst / "sub"
    out_dir.mkdir(parents=True)
    txt = out_dir / "note.txt"
    txt.write_text("From: a\n\nbody\n", encoding="utf-8")

    _write_runlog(
        dst, "unpack_msg_folder",
        [{"input_abs": str(msg), "kind": "msg", "output": str(txt)}],
    )

    trace = trace_source(txt, src, dst)
    assert trace.source_archive == msg.resolve()
    assert trace.archive_type == "msg"
    assert trace.method == "runlog"
    assert trace.attachment_of is None


def test_msg_attachment_disambiguated_by_stem(tmp_path: Path):
    """Sibling .msg files share one attachments/ dir; the <stem>__ prefix on
    the attachment filename picks the right source .msg."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    note = src / "sub" / "note.msg"
    other = src / "sub" / "other.msg"
    note.write_bytes(b"")
    other.write_bytes(b"")
    out_dir = dst / "sub"
    (out_dir / "attachments").mkdir(parents=True)
    note_txt = out_dir / "note.txt"
    note_txt.write_text("From: a\n\nbody\n", encoding="utf-8")
    (out_dir / "other.txt").write_text("From: b\n\nbody\n", encoding="utf-8")
    att = out_dir / "attachments" / "note__invoice.pdf"
    att.write_bytes(b"%PDF")

    _write_runlog(
        dst, "unpack_msg_folder",
        [
            {"input_abs": str(note), "kind": "msg", "output": str(note_txt)},
            {"input_abs": str(other), "kind": "msg", "output": str(out_dir / "other.txt")},
        ],
    )

    trace = trace_source(att, src, dst)
    assert trace.source_archive == note.resolve()
    assert trace.archive_type == "msg"
    assert trace.attachment_of == note_txt.resolve()


def test_msg_path_fallback_without_runlog(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    msg = src / "deep" / "note.msg"
    msg.parent.mkdir(parents=True)
    msg.write_bytes(b"")
    out_dir = dst / "deep"
    out_dir.mkdir(parents=True)
    txt = out_dir / "note.txt"
    txt.write_text("From: a\n\nbody\n", encoding="utf-8")

    trace = trace_source(txt, src, dst)
    assert trace.source_archive == msg.resolve()
    assert trace.archive_type == "msg"
    assert trace.method == "path"


# --- mbox ------------------------------------------------------------------


def test_inplace_mbox_unpacked_dir_resolves_to_mbox_not_pst(tmp_path: Path):
    """The per-archive (in-place) unpack writes <stem>_unpacked for BOTH pst and
    mbox, so the _unpacked suffix alone doesn't imply pst. With no runlog, probe
    the source: <stem>.mbox sitting next to the tree wins over a non-existent
    <stem>.pst."""
    root = tmp_path / "corpus"  # in-place: source_root == target_root
    mbox = root / "sub" / "archive.mbox"
    mbox.parent.mkdir(parents=True)
    mbox.write_bytes(b"")
    folder = root / "sub" / "archive_unpacked" / "thread-0001_x"
    folder.mkdir(parents=True)
    txt = folder / "msg-0001_20240101_bob.txt"
    txt.write_text("From: bob\n\nbody\n", encoding="utf-8")

    trace = trace_source(txt, root, root)
    assert trace.archive_type == "mbox"
    assert trace.source_archive == mbox.resolve()
    assert trace.archive_exists is True
    assert trace.method == "path"


def test_mbox_via_runlog_survives_flattening_and_collision(tmp_path: Path):
    """mbox batch flattens <rel> and adds -1 on stem collisions, so path-math
    can't reverse it — but the runlog's input_abs records the true source."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    mbox = src / "team" / "archive.mbox"
    mbox.parent.mkdir(parents=True)
    mbox.write_bytes(b"")
    # Output flattened to <dst>/archive-1 (a -1 collision suffix).
    out_dir = dst / "archive-1" / "thread-0001_subject"
    out_dir.mkdir(parents=True)
    txt = out_dir / "msg-0002_20240101_bob.txt"
    txt.write_text("From: bob\n\nbody\n", encoding="utf-8")

    _write_runlog(
        dst, "unpack_mbox_folder",
        [{"input_abs": str(mbox), "kind": "mbox", "output": str(dst / "archive-1")}],
    )

    trace = trace_source(txt, src, dst)
    assert trace.source_archive == mbox.resolve()
    assert trace.archive_type == "mbox"
    assert trace.method == "runlog"
    assert trace.message_number == 2


def test_mbox_path_fallback_ambiguous_raises(tmp_path: Path):
    """Without a runlog, an mbox whose stem occurs twice in the source tree is
    unresolvable by path alone — refuse rather than guess."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    (src / "a" / "archive.mbox").write_bytes(b"")
    (src / "b" / "archive.mbox").write_bytes(b"")
    out_dir = dst / "archive" / "thread-0001_x"
    out_dir.mkdir(parents=True)
    txt = out_dir / "msg-0001_20240101_bob.txt"
    txt.write_text("From: bob\n\nbody\n", encoding="utf-8")

    with pytest.raises(LookupError):
        trace_source(txt, src, dst)


# --- error handling --------------------------------------------------------


def test_file_not_under_target_root_raises(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    dst.mkdir(parents=True)
    stray = tmp_path / "elsewhere.txt"
    stray.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        trace_source(stray, src, dst)


def test_missing_file_raises(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    dst.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        trace_source(dst / "nope.txt", src, dst)


# --- end-to-end: run the real unpack scripts, then trace their output --------
#
# These guard against drift between this module's assumptions and the layout +
# runlog schema the scripts actually emit. They parse real archives, so they
# need the optional deps (extract-msg for .msg); skip cleanly if absent.

import shutil

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_e2e_msg_script_output_traces_back(tmp_path: Path):
    from scripts import unpack_msg_folder

    sample = _REPO_ROOT / "example-msg-files" / "unicode.msg"
    if not sample.is_file():
        pytest.skip("sample .msg fixture missing")

    src = tmp_path / "src" / "mail"
    src.mkdir(parents=True)
    shutil.copy(sample, src / "unicode.msg")
    dst = tmp_path / "out"

    rc = unpack_msg_folder.main([str(tmp_path / "src"), str(dst)])
    assert rc == 0

    txt = dst / "mail" / "unicode.txt"
    if not txt.is_file():
        pytest.skip("extract-msg not available; no .txt produced")

    trace = trace_source(txt, tmp_path / "src", dst)
    assert trace.archive_type == "msg"
    assert trace.source_archive == (src / "unicode.msg").resolve()
    assert trace.method == "runlog"


def test_e2e_mbox_script_output_traces_back(tmp_path: Path):
    from scripts import unpack_mbox_folder

    sample = _REPO_ROOT / "mbox_handling" / "samples" / "sample.mbox"
    if not sample.is_file():
        pytest.skip("sample .mbox fixture missing")

    src = tmp_path / "src"
    src.mkdir(parents=True)
    shutil.copy(sample, src / "sample.mbox")
    dst = tmp_path / "out"

    rc = unpack_mbox_folder.main([str(src), str(dst)])
    assert rc == 0

    txts = sorted(dst.rglob("msg-*.txt"))
    assert txts, "mbox unpack produced no message .txt files"

    trace = trace_source(txts[0], src, dst)
    assert trace.archive_type == "mbox"
    assert trace.source_archive == (src / "sample.mbox").resolve()
    assert trace.method == "runlog"
