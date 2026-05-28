"""Tests for scripts/extract_archives_folder.py — the unified archive
extractor that scans for .mbox/.pst/.msg/.zip, mirrors structure into an
output folder, and recurses fully into archives revealed inside zips.

These tests focus on the *invented* part: the recursion loop, the
relative-path mirror rule, the cycle-safe processed-set, and the
guards. Per-type unpacking (mbox/msg/pst) is covered by existing tests,
so the only email type exercised here is .msg, with read_message faked
the same way test_unpack_msg_folder.py does.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """Build an in-memory .zip from {arcname: content} and return its bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, data in members.items():
            zf.writestr(arcname, data)
    return buf.getvalue()


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_zip_bytes(members))


def _fake_msg_reader(monkeypatch, bodies: dict[str, str]) -> None:
    """Patch msg_handling.unpack.read_message to return a ParsedMessage whose
    body is looked up by the .msg file's stem."""
    from msg_handling import unpack
    from msg_handling.messages import ParsedMessage

    def reader(p):
        p = Path(p)
        body = bodies.get(p.stem)
        if body is None:
            raise FileNotFoundError(f"no fake for {p}")
        return ParsedMessage(
            from_addr="a@x", to="b@x", cc="", subject=p.stem,
            date="Mon, 1 Jan 2024 00:00:00 +0000", message_id="<a@x>",
            body=body, attachments=(),
        )

    monkeypatch.setattr(unpack, "read_message", reader)


# 1. zip-containing-a-.msg fully unpacks to the right mirrored path.
def test_zip_containing_msg_unpacks_to_mirrored_path(tmp_path, monkeypatch):
    from scripts.extract_archives_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_zip(src / "a" / "foo.zip", {"bar.msg": b"\x00binary-msg"})
    _fake_msg_reader(monkeypatch, {"bar": "hello from bar"})

    rc = main([str(src), str(dst)])

    assert rc == 0
    txt = dst / "a" / "foo" / "bar.txt"
    assert txt.is_file()
    assert "hello from bar" in txt.read_text(encoding="utf-8")


# 2. nested zip (zip inside zip) extracts recursively.
def test_nested_zip_extracts_recursively(tmp_path):
    from scripts.extract_archives_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    inner = _zip_bytes({"leaf.txt": b"deep content"})
    _write_zip(src / "outer.zip", {"inner.zip": inner})

    rc = main([str(src), str(dst)])

    assert rc == 0
    leaf = dst / "outer" / "inner" / "leaf.txt"
    assert leaf.is_file()
    assert leaf.read_bytes() == b"deep content"


# 3. an already-processed zip in the output tree is not re-extracted; the
#    loop terminates (each distinct zip extracted exactly once).
def test_processed_zip_not_reextracted(tmp_path, monkeypatch):
    import scripts.extract_archives_folder as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    inner = _zip_bytes({"leaf.txt": b"x"})
    _write_zip(src / "outer.zip", {"inner.zip": inner})

    calls: list[str] = []
    real = mod._extract_zip

    def spy(zip_path, target):
        calls.append(Path(zip_path).name)
        return real(zip_path, target)

    monkeypatch.setattr(mod, "_extract_zip", spy)

    rc = mod.main([str(src), str(dst)])

    assert rc == 0
    # Each zip extracted exactly once — no infinite re-scan of the output.
    assert sorted(calls) == ["inner.zip", "outer.zip"]


# 4. output folder inside the input folder is rejected (feedback loop).
def test_output_inside_input_rejected(tmp_path):
    from scripts.extract_archives_folder import main

    src = tmp_path / "src"
    src.mkdir()
    dst = src / "out"

    rc = main([str(src), str(dst)])

    assert rc == 1
    assert not dst.exists()


# 5. --max-depth caps recursion: nested archives below the cap are left
#    unextracted rather than looping forever.
def test_max_depth_caps_recursion(tmp_path):
    from scripts.extract_archives_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    inner = _zip_bytes({"leaf.txt": b"deep"})
    _write_zip(src / "outer.zip", {"inner.zip": inner})

    rc = main([str(src), str(dst), "--max-depth", "1"])

    assert rc == 0
    # Top-level zip extracted (depth 1) ...
    assert (dst / "outer" / "inner.zip").is_file()
    # ... but the nested zip revealed inside it was not (would be depth 2).
    assert not (dst / "outer" / "inner").exists()


# 6. a real .mbox in a subdir unpacks to the mirrored <stem>_unpacked dir.
#    Exercises the orchestrator's mbox dispatch + target convention with a
#    genuine mbox (stdlib mailbox parses it — no fakes needed).
_SAMPLE_MBOX = (
    "From alice@example.com Mon Jan  1 00:00:00 2024\n"
    "From: alice@example.com\n"
    "To: bob@example.com\n"
    "Subject: Hello\n"
    "Date: Mon, 1 Jan 2024 00:00:00 +0000\n"
    "\n"
    "Hello body here\n"
)


def test_mbox_in_subdir_unpacks_to_mirrored_path(tmp_path):
    from scripts.extract_archives_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    mbox = src / "sub" / "sample.mbox"
    mbox.parent.mkdir(parents=True)
    mbox.write_text(_SAMPLE_MBOX, encoding="utf-8")

    rc = main([str(src), str(dst)])

    assert rc == 0
    unpacked = dst / "sub" / "sample_unpacked"
    assert unpacked.is_dir()
    txts = list(unpacked.rglob("*.txt"))
    assert txts, "expected at least one unpacked message .txt"
    assert any("Hello body here" in t.read_text(encoding="utf-8") for t in txts)


# 7. _target_for convention for every kind, both passes. This is the cheap
#    discriminator for the .pst branch, which cannot run end-to-end here
#    (pypff/libratom are not installed) but whose target layout is ours.
def test_target_for_conventions(tmp_path):
    from scripts.extract_archives_folder import _target_for

    src = tmp_path / "in"
    dst = tmp_path / "out"

    # From the input tree -> mirrored under dst.
    f = src / "a" / "x.pst"
    assert _target_for(f, "pst", from_input=True, src=src, dst=dst) == \
        dst / "a" / "x_unpacked"
    assert _target_for(src / "a" / "x.mbox", "mbox", from_input=True,
                       src=src, dst=dst) == dst / "a" / "x_unpacked"
    assert _target_for(src / "a" / "x.msg", "msg", from_input=True,
                       src=src, dst=dst) == dst / "a"
    assert _target_for(src / "a" / "x.zip", "zip", from_input=True,
                       src=src, dst=dst) == dst / "a" / "x"

    # Discovered inside the output tree -> processed in place.
    g = dst / "a" / "bundle" / "y.pst"
    assert _target_for(g, "pst", from_input=False, src=src, dst=dst) == \
        dst / "a" / "bundle" / "y_unpacked"
    assert _target_for(dst / "a" / "bundle" / "y.msg", "msg", from_input=False,
                       src=src, dst=dst) == dst / "a" / "bundle"


# 8. a .msg that fails to parse (no .txt produced) is counted as a failure,
#    not a silent success — mirrors unpack_msg_folder.py's guard.
def test_msg_parse_failure_counts_as_failure(tmp_path, monkeypatch):
    from scripts.extract_archives_folder import main
    from msg_handling import unpack

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src).mkdir()
    (src / "bad.msg").write_bytes(b"\x00not-a-real-msg")

    def failing_reader(p):
        raise ValueError("corrupt msg")

    monkeypatch.setattr(unpack, "read_message", failing_reader)

    rc = main([str(src), str(dst)])

    assert rc == 1
    assert not (dst / "bad.txt").exists()


# 9. --copy-others copies non-archive files into the mirrored output while
#    archives are still extracted; without the flag they are left behind.
def test_copy_others_mirrors_non_archive_files(tmp_path):
    from scripts.extract_archives_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "docs").mkdir(parents=True)
    (src / "docs" / "report.txt").write_text("hello report", encoding="utf-8")
    (src / "top.pdf").write_bytes(b"%PDF-fake")
    _write_zip(src / "a" / "bundle.zip", {"leaf.txt": b"in zip"})

    rc = main([str(src), str(dst), "--copy-others"])

    assert rc == 0
    # Non-archive files copied, structure preserved.
    assert (dst / "docs" / "report.txt").read_text(encoding="utf-8") == \
        "hello report"
    assert (dst / "top.pdf").read_bytes() == b"%PDF-fake"
    # Archive still extracted (not merely copied).
    assert (dst / "a" / "bundle" / "leaf.txt").read_bytes() == b"in zip"
    # The archive original itself is NOT copied alongside its extraction.
    assert not (dst / "a" / "bundle.zip").exists()


def test_non_archive_files_not_copied_without_flag(tmp_path):
    from scripts.extract_archives_folder import main

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "docs").mkdir(parents=True)
    (src / "docs" / "report.txt").write_text("hello", encoding="utf-8")

    rc = main([str(src), str(dst)])

    assert rc == 0
    assert not (dst / "docs" / "report.txt").exists()


# 10. re-run with --copy-others does not re-copy an up-to-date destination
#     (overwrite only when the source is newer).
def test_copy_others_skips_up_to_date_on_rerun(tmp_path, monkeypatch):
    import scripts.extract_archives_folder as mod

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src).mkdir()
    (src / "report.txt").write_text("v1", encoding="utf-8")

    assert mod.main([str(src), str(dst), "--copy-others"]) == 0
    assert (dst / "report.txt").read_text(encoding="utf-8") == "v1"

    # Second run: spy on the actual copy primitive — it must not fire again
    # because the destination is already up to date (copy2 preserves mtime).
    import shutil
    calls: list[str] = []
    real_copy2 = shutil.copy2
    monkeypatch.setattr(
        mod.shutil, "copy2",
        lambda s, d, *a, **k: (calls.append(str(s)), real_copy2(s, d, *a, **k))[1],
    )

    assert mod.main([str(src), str(dst), "--copy-others"]) == 0
    assert calls == []
