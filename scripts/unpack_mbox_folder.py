"""Batch-unpack every ``.mbox`` file under a folder into a chosen output tree.

Usage:
    python scripts/unpack_mbox_folder.py <input-folder> <output-folder>

For each ``.mbox`` file found (recursively, case-insensitive), the script
creates ``<output>/<mbox-stem>/`` and delegates the actual unpacking to
``mbox_handling.unpack.ensure_unpacked``. The output tree per mbox is the
standard one: ``thread-NNNN_slug/`` folders containing message ``.txt``
files and a per-message ``attachments/`` subfolder, plus ``_unthreaded/``
for orphans.

Same-stem collisions across subfolders are disambiguated with ``-1``,
``-2``, ... suffixes. Per-mbox errors are logged to stderr and the batch
continues; the exit code is non-zero if at least one mbox failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct invocation (``python scripts/unpack_mbox_folder.py``) in
# addition to ``python -m scripts.unpack_mbox_folder`` by putting the
# repo root on sys.path before importing project code.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mbox_handling.unpack import ensure_unpacked  # noqa: E402


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

    mboxes = _discover_mboxes(src)
    succeeded = 0
    failed = 0
    used_names: set[str] = set()

    for mbox in mboxes:
        target = _pick_target(dst, mbox.stem, used_names)
        used_names.add(target.name)
        try:
            ensure_unpacked(mbox, target=target)
        except Exception as exc:  # noqa: BLE001 — per-file isolation by design
            failed += 1
            print(
                f"error: failed to unpack {mbox}: {exc}",
                file=sys.stderr,
            )
            continue
        succeeded += 1

    print(
        f"Processed {len(mboxes)} mbox file(s): "
        f"{succeeded} succeeded, {failed} failed.",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-unpack every .mbox under <input-folder> into "
            "<output-folder>/<mbox-stem>/ subdirectories."
        )
    )
    parser.add_argument("input_folder", help="Folder to scan for .mbox files")
    parser.add_argument(
        "output_folder",
        help="Folder to write per-mbox unpacked trees into",
    )
    return parser.parse_args(argv)


def _discover_mboxes(root: Path) -> list[Path]:
    """Return every regular file under ``root`` with a ``.mbox`` extension
    (case-insensitive), sorted deterministically by relative path."""
    results: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".mbox":
            results.append(path)
    results.sort(key=lambda p: str(p.relative_to(root)))
    return results


def _pick_target(out_root: Path, stem: str, used: set[str]) -> Path:
    """Pick ``<out_root>/<stem>``, falling back to ``<stem>-1/-2/...`` when
    already used in this run or already non-empty on disk."""
    candidate_name = stem
    if candidate_name not in used and not _is_in_use(out_root / candidate_name):
        return out_root / candidate_name
    for n in range(1, 100_000):
        candidate_name = f"{stem}-{n}"
        candidate = out_root / candidate_name
        if candidate_name not in used and not _is_in_use(candidate):
            return candidate
    raise RuntimeError(
        f"could not find an unused output subdir for stem {stem!r} under {out_root}"
    )


def _is_in_use(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
