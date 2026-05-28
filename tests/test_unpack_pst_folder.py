"""Tests for scripts/unpack_pst_folder.py — batch .pst unpacker that
preserves subdirectory layout.

Like test_pst_unpack.py, these tests inject a fake
``iter_pst_messages`` so we never need binary .pst fixtures. The unit
under test is the script's traversal + target-layout behavior; the
parsing / file-layout is exercised by the in-place unit tests.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path


def _msg(
    *,
    folder_path: str = "Top/Inbox",
    subject: str = "Hello",
    sender: str = "alice@example.com",
    date: str = "2024-03-01 12:00:00+00:00",
    body: str = "body",
    attachments: tuple = (),
    error: str | None = None,
):
    from pst_handling.messages import ParsedPstMessage
    return ParsedPstMessage(
        folder_path=folder_path, subject=subject, sender=sender,
        date=date, body=body, attachments=attachments, error=error,
    )


def _att(filename: str, data: bytes = b"x"):
    from pst_handling.messages import ParsedPstAttachment

    def write_to(target: Path) -> None:
        target.write_bytes(data)

    return ParsedPstAttachment(
        filename=filename, size=len(data), write_to=write_to
    )


def _install_iter(monkeypatch, mapping_factory):
    """Replace pst_handling.unpack.iter_pst_messages with a fake."""
    from pst_handling import unpack

    def fake(pst_path):
        result = mapping_factory(Path(pst_path))
        if result is None:
            raise FileNotFoundError(f"no fake configured for {pst_path}")
        yield from result

    monkeypatch.setattr(unpack, "iter_pst_messages", fake)


def _drop_pst(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")


def test_extracts_each_pst_into_target_with_subdir_layout(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.unpack_pst_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_pst(src / "a" / "foo.pst")
    _drop_pst(src / "b" / "deep" / "bar.pst")

    def factory(p: Path):
        if p.name == "foo.pst":
            return [_msg(folder_path="Top/Inbox", subject="Foo",
                         attachments=(_att("alpha.pdf", b"AAA"),))]
        if p.name == "bar.pst":
            return [_msg(folder_path="Top/Sent", subject="Bar",
                         attachments=(_att("beta.pdf", b"BBB"),))]
        return None
    _install_iter(monkeypatch, factory)

    rc = main([str(src), str(dst)])
    assert rc == 0

    foo_unpacked = dst / "a" / "foo_unpacked"
    bar_unpacked = dst / "b" / "deep" / "bar_unpacked"

    assert any((foo_unpacked / "Top" / "Inbox").glob("msg-0001_*.txt"))
    assert (foo_unpacked / "Top" / "Inbox" / "attachments" / "msg-0001"
            / "alpha.pdf").read_bytes() == b"AAA"

    assert any((bar_unpacked / "Top" / "Sent").glob("msg-0001_*.txt"))
    assert (bar_unpacked / "Top" / "Sent" / "attachments" / "msg-0001"
            / "beta.pdf").read_bytes() == b"BBB"

    # Source is untouched.
    assert not (src / "a" / "foo_unpacked").exists()
    assert (src / "a" / "foo.pst").is_file()


def test_idempotent_on_second_run(tmp_path: Path, monkeypatch) -> None:
    from scripts.unpack_pst_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_pst(src / "foo.pst")
    _install_iter(
        monkeypatch,
        lambda p: [_msg(folder_path="Top", subject="A", body="b")],
    )

    assert main([str(src), str(dst)]) == 0
    out_dir = dst / "foo_unpacked"
    assert out_dir.is_dir()

    # Make the unpacked dir strictly newer than the source .pst so the
    # mtime-idempotent skip path inside ensure_unpacked fires.
    future = int(time.time()) + 60
    os.utime(out_dir, (future, future))

    def tripwire(_p):
        raise AssertionError(
            "iter_pst_messages should not run on idempotent skip"
        )
        yield  # pragma: no cover
    from pst_handling import unpack
    monkeypatch.setattr(unpack, "iter_pst_messages", tripwire)

    assert main([str(src), str(dst)]) == 0


def test_output_inside_input_is_refused(tmp_path: Path) -> None:
    from scripts.unpack_pst_folder import main

    src = tmp_path / "src"
    src.mkdir()
    _drop_pst(src / "foo.pst")
    inside = src / "out"

    rc = main([str(src), str(inside)])
    assert rc != 0
    assert not inside.exists() or not any(inside.iterdir())


def test_broken_pst_does_not_abort_batch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scripts.unpack_pst_folder import main
    from pst_handling import unpack

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_pst(src / "good.pst")
    _drop_pst(src / "broken.pst")

    def fake(pst_path):
        p = Path(pst_path)
        if p.name == "broken.pst":
            raise ValueError("intentional test failure")
        yield _msg(folder_path="Top", subject="ok", body="hi")
    monkeypatch.setattr(unpack, "iter_pst_messages", fake)

    rc = main([str(src), str(dst)])
    assert rc != 0  # at least one failure → non-zero exit

    # The good one extracted; the broken one logged a warning.
    assert (dst / "good_unpacked").exists()
    assert any((dst / "good_unpacked").rglob("msg-*.txt"))
    err = capsys.readouterr().err
    assert "broken.pst" in err


def test_recursive_discovery_case_insensitive(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.unpack_pst_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_pst(src / "FOO.PST")
    _drop_pst(src / "nested" / "BAR.Pst")
    _install_iter(
        monkeypatch,
        lambda p: [_msg(folder_path="Top", subject="x", body="y")],
    )

    assert main([str(src), str(dst)]) == 0
    assert (dst / "FOO_unpacked").is_dir()
    assert (dst / "nested" / "BAR_unpacked").is_dir()


def test_input_must_be_an_existing_directory(tmp_path: Path) -> None:
    from scripts.unpack_pst_folder import main

    rc = main([str(tmp_path / "nope"), str(tmp_path / "out")])
    assert rc != 0


def test_prints_found_header_with_count(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scripts.unpack_pst_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_pst(src / "a.pst")
    _drop_pst(src / "nested" / "b.pst")
    _install_iter(
        monkeypatch,
        lambda p: [_msg(folder_path="Top", subject="x", body="y")],
    )

    rc = main([str(src), str(dst)])
    assert rc == 0
    err = capsys.readouterr().err
    assert re.search(r"Found 2 pst file\(s\) under .*src", err), err


def test_prints_per_item_progress_line(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scripts.unpack_pst_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_pst(src / "2024" / "alpha.pst")
    _drop_pst(src / "beta.pst")
    _install_iter(
        monkeypatch,
        lambda p: [_msg(folder_path="Top", subject="x", body="y")],
    )

    rc = main([str(src), str(dst)])
    assert rc == 0
    err = capsys.readouterr().err
    # Sorted by relative path: 2024/alpha.pst comes before beta.pst.
    assert re.search(r"\[1/2\] unpacking 2024/alpha\.pst", err), err
    assert re.search(r"\[2/2\] unpacking beta\.pst", err), err


def test_summary_includes_elapsed_seconds(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scripts.unpack_pst_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _drop_pst(src / "a.pst")
    _install_iter(
        monkeypatch,
        lambda p: [_msg(folder_path="Top", subject="x", body="y")],
    )

    rc = main([str(src), str(dst)])
    assert rc == 0
    err = capsys.readouterr().err
    assert re.search(r"Processed 1 pst file\(s\) in \d+\.\d+s:", err), err
