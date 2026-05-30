"""Batch-extract every ``.msg`` file under a folder into an output tree
that mirrors the input's subdirectory layout.

Usage:
    python scripts/unpack_msg_folder.py <input-folder> <output-folder>

For each ``.msg`` found (recursively, case-insensitive) at
``<input>/<rel>/foo.msg``, the script writes ``<output>/<rel>/foo.txt``
(headers + plain-text body) and ``<output>/<rel>/attachments/foo__*``
(extracted attachments). The source ``.msg`` is left untouched in the
input tree — only the extracted text and attachments end up in
``<output>``. Re-runs are mtime-idempotent: a target ``.txt`` that is at
least as new as its source ``.msg`` is reused.

Contrast with ``scripts/unpack_mbox_folder.py``, which flattens every
mbox into ``<output>/<stem>/``. Here the input's layout is preserved so
the output is a drop-in index-ready mirror of the source corpus.

Per-msg errors are logged to stderr and the batch continues; the exit
code is non-zero if at least one ``.msg`` failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow ``python scripts/unpack_msg_folder.py`` (no ``-m``) by putting the
# repo root on sys.path before importing project code — same trick as the
# sibling mbox script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from msg_handling.unpack import ensure_unpacked  # noqa: E402
from scripts._runlog import RunLog  # noqa: E402
from scripts import _runlog  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src = Path(args.input_folder).expanduser().resolve()
    dst = Path(args.output_folder).expanduser().resolve()

    if not src.exists():
        print(f"error: input folder not found: {src}", file=sys.stderr)
        return 1
    if not src.is_dir():
        print(f"error: input path is not a directory: {src}", file=sys.stderr)
        return 1
    if _is_inside(dst, src):
        print(
            f"error: output folder {dst} is inside input folder {src}; "
            "pick a target outside the input tree.",
            file=sys.stderr,
        )
        return 1

    dst.mkdir(parents=True, exist_ok=True)

    msgs = _discover_msgs(src)
    print(
        f"Found {len(msgs)} msg file(s) under {src}",
        file=sys.stderr,
    )
    rl = RunLog(
        "unpack_msg_folder",
        input_root=src,
        output_root=dst,
        argv=argv if argv is not None else sys.argv[1:],
        log_dir=args.log_dir,
        force=args.force,
        enabled=not args.no_runlog,
    )
    succeeded = 0
    skipped = 0
    failed = 0
    started = time.monotonic()

    total = len(msgs)
    for i, msg_path in enumerate(msgs, start=1):
        rel = msg_path.relative_to(src)
        rel_parent = msg_path.parent.relative_to(src)
        target_dir = dst / rel_parent
        txt_out = target_dir / f"{msg_path.stem}.txt"
        # Idempotent re-run: skip a .msg whose .txt is already present and the
        # source is unchanged.
        if (done := rl.done_output(msg_path)) is not None:
            skipped += 1
            print(f"[{i}/{total}] skip (done) {rel}", file=sys.stderr)
            rl.record(msg_path, kind="msg", output=done, status="skip")
            continue
        print(f"[{i}/{total}] extracting {rel}", file=sys.stderr)
        item_started = time.monotonic()
        try:
            ensure_unpacked(msg_path, target=target_dir)
        except Exception as exc:  # noqa: BLE001 — per-file isolation by design
            failed += 1
            print(
                f"error: failed to unpack {msg_path}: {exc}",
                file=sys.stderr,
            )
            rl.record(msg_path, kind="msg", output=None, status="fail", error=exc)
            continue
        # ``ensure_unpacked`` swallows per-file parse failures and logs to
        # stderr without raising. To still surface those in the exit code
        # and summary, treat a missing ``.txt`` after a call as a failure.
        if not txt_out.exists():
            failed += 1
            print(
                f"warning: {rel}: no .txt produced "
                "(parse failure; see prior msg_handling warning)",
                file=sys.stderr,
            )
            rl.record(msg_path, kind="msg", output=None, status="fail",
                      error="no .txt produced (parse failure)")
            continue
        succeeded += 1
        rl.record(msg_path, kind="msg", output=txt_out, status="ok",
                  duration_s=round(time.monotonic() - item_started, 3))

    elapsed = time.monotonic() - started
    skipped_part = f", {skipped} skipped" if skipped else ""
    print(
        f"Processed {total} msg file(s) in {elapsed:.1f}s: "
        f"{succeeded} succeeded, {failed} failed{skipped_part}.",
        file=sys.stderr,
    )
    exit_code = 0 if failed == 0 else 1
    rl.finish(exit_code=exit_code)
    return exit_code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-extract every .msg under <input-folder> into "
            "<output-folder>, preserving the input's subdirectory layout. "
            "Outputs are <output>/<rel>/<stem>.txt + "
            "<output>/<rel>/attachments/<stem>__*; the source .msg is "
            "left untouched."
        ),
    )
    parser.add_argument("input_folder", help="Folder to scan for .msg files")
    parser.add_argument(
        "output_folder",
        help="Folder to write the extracted .txt + attachments tree into",
    )
    _runlog.add_args(parser)
    return parser.parse_args(argv)


def _discover_msgs(root: Path) -> list[Path]:
    """Return every regular file under ``root`` with a ``.msg`` extension
    (case-insensitive), sorted deterministically by relative path."""
    results: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".msg":
            results.append(path)
    results.sort(key=lambda p: str(p.relative_to(root)))
    return results


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
