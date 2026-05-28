"""Batch-unpack every ``.pst`` file under a folder into an output tree
that mirrors the input's subdirectory layout.

Usage:
    python scripts/unpack_pst_folder.py <input-folder> <output-folder>

For each ``.pst`` found (recursively, case-insensitive) at
``<input>/<rel>/foo.pst``, the script writes
``<output>/<rel>/foo_unpacked/<pst-folder-path>/msg-NNNN_<date>_<from>.txt``
+ ``<output>/<rel>/foo_unpacked/<pst-folder-path>/attachments/msg-NNNN/*``.
The source ``.pst`` is left untouched in the input tree.

Re-runs reuse fresh outputs: when the target ``<stem>_unpacked/``
already exists and is at least as new as the source ``.pst``, the
unpack is skipped.

Per-pst errors are logged to stderr and the batch continues; the exit
code is non-zero if at least one ``.pst`` failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow ``python scripts/unpack_pst_folder.py`` (no ``-m``) by putting the
# repo root on sys.path before importing project code — same trick as the
# sibling mbox / msg scripts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pst_handling.unpack import ensure_unpacked  # noqa: E402


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

    psts = _discover_psts(src)
    print(
        f"Found {len(psts)} pst file(s) under {src}",
        file=sys.stderr,
    )
    succeeded = 0
    failed = 0
    started = time.monotonic()

    total = len(psts)
    for i, pst_path in enumerate(psts, start=1):
        rel = pst_path.relative_to(src)
        rel_parent = pst_path.parent.relative_to(src)
        target_dir = dst / rel_parent / f"{pst_path.stem}_unpacked"
        print(f"[{i}/{total}] unpacking {rel}", file=sys.stderr)
        try:
            ensure_unpacked(pst_path, target=target_dir)
        except Exception as exc:  # noqa: BLE001 — per-file isolation by design
            failed += 1
            print(
                f"error: failed to unpack {pst_path}: {exc}",
                file=sys.stderr,
            )
            continue
        # ``ensure_unpacked`` does not raise on a corrupt archive when the
        # error happens inside iter_pst_messages — the iterator yields
        # error placeholders instead. We treat an empty output dir as a
        # failure to keep the exit code honest.
        if not target_dir.exists() or not any(target_dir.iterdir()):
            failed += 1
            print(
                f"warning: {rel}: no output produced (empty .pst or "
                "parse failure; see prior pst_handling warning)",
                file=sys.stderr,
            )
            continue
        succeeded += 1

    elapsed = time.monotonic() - started
    print(
        f"Processed {total} pst file(s) in {elapsed:.1f}s: "
        f"{succeeded} succeeded, {failed} failed.",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-unpack every .pst under <input-folder> into "
            "<output-folder>, preserving the input's subdirectory layout. "
            "Outputs are <output>/<rel>/<stem>_unpacked/<pst-folder-path>/"
            "msg-NNNN_*.txt + attachments/msg-NNNN/*; the source .pst is "
            "left untouched."
        ),
    )
    parser.add_argument("input_folder", help="Folder to scan for .pst files")
    parser.add_argument(
        "output_folder",
        help="Folder to write the unpacked trees into",
    )
    return parser.parse_args(argv)


def _discover_psts(root: Path) -> list[Path]:
    """Return every regular file under ``root`` with a ``.pst`` extension
    (case-insensitive), sorted deterministically by relative path."""
    results: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pst":
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
