"""Batch-convert PowerPoint-family files under a folder into PDFs.

Standalone tool — NOT wired into the indexer/server pipeline. It walks
``<input-folder>`` recursively, converts every presentation file it finds to
PDF via LibreOffice headless, and writes the PDFs into ``<output-folder>``
mirroring the input's subdirectory layout. The source tree is never modified.

    <in>/a/b/talk.pptx   ->   <out>/a/b/talk.pdf

Recognised extensions (case-insensitive): ``.ppt .pptx .pps .ppsx .pot .potx
.pptm .ppsm .potm .odp`` — every PowerPoint + OpenDocument presentation format
LibreOffice can read, including legacy binary ``.ppt``.

Requires **LibreOffice** on the system (there is no portable pure-Python
renderer). The ``soffice`` binary is auto-detected on ``PATH``, via the
``SOFFICE`` env var, the macOS app bundle, and common Linux locations; override
with ``--soffice /path/to/soffice``. Install it any way you like:

    macOS:  brew install --cask libreoffice   (or the .dmg from libreoffice.org)
    Linux:  apt install libreoffice           (or dnf/snap/flatpak/.tar.gz)

A structured JSONL run-log is written under ``<output>/_runlogs/`` (override
with ``--log-dir``); it records per-file status and drives an idempotent
re-run — a second invocation skips files already converted (output present,
source unchanged) and only (re)processes new or previously-failed ones. Pass
``--force`` to reconvert everything in place, ``--no-runlog`` to disable the
log, and ``--dry-run`` to preview what would be converted.

Usage:
    python scripts/ppt_to_pdf_folder.py <input-folder> <output-folder>
        [--soffice PATH] [--timeout SECONDS] [--dry-run]
        [--log-dir PATH] [--force] [--no-runlog]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Allow direct invocation (``python scripts/ppt_to_pdf_folder.py``) in addition
# to ``python -m scripts.ppt_to_pdf_folder`` by putting the repo root on
# sys.path before importing the sibling run-log helper.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._runlog import RunLog  # noqa: E402
from scripts import _runlog  # noqa: E402

# Every PowerPoint + OpenDocument presentation extension LibreOffice can read.
PPT_EXTS = {
    ".ppt", ".pptx", ".pps", ".ppsx", ".pot",
    ".potx", ".pptm", ".ppsm", ".potm", ".odp",
}

DEFAULT_TIMEOUT_S = 180

# Common non-PATH locations for the LibreOffice binary, probed in order.
_SOFFICE_CANDIDATES = (
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
    "/opt/libreoffice/program/soffice",
    "/snap/bin/libreoffice",
)


class ConversionError(Exception):
    """A single file could not be converted to PDF."""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src = Path(args.input_folder).expanduser().resolve()
    dst = Path(args.output_folder).expanduser().resolve()

    if not src.exists():
        print(f"error: input folder not found: {src}", file=sys.stderr)
        return 2
    if not src.is_dir():
        print(f"error: input path is not a directory: {src}", file=sys.stderr)
        return 2
    if _is_inside(dst, src):
        print(
            f"error: output folder {dst} is inside input folder {src}; "
            "pick a target outside the input tree.",
            file=sys.stderr,
        )
        return 2

    files = _discover_presentations(src)
    print(f"Found {len(files)} presentation file(s) under {src}", file=sys.stderr)

    if args.dry_run:
        for f in files:
            rel = f.relative_to(src)
            print(f"would convert {rel} -> {rel.with_suffix('.pdf')}", file=sys.stderr)
        return 0

    soffice = _find_soffice(args.soffice)
    if soffice is None:
        print(
            "error: LibreOffice not found. Install it (brew install --cask "
            "libreoffice, the libreoffice.org .dmg, or apt/dnf/snap) or pass "
            "--soffice /path/to/soffice.",
            file=sys.stderr,
        )
        return 2

    dst.mkdir(parents=True, exist_ok=True)
    rl = RunLog(
        "ppt_to_pdf_folder",
        input_root=src,
        output_root=dst,
        argv=argv if argv is not None else sys.argv[1:],
        log_dir=args.log_dir,
        force=args.force,
        enabled=not args.no_runlog,
    )

    converted = skipped = failed = 0
    used_outputs: set[str] = set()
    started = time.monotonic()
    total = len(files)

    for i, path in enumerate(files, start=1):
        rel = path.relative_to(src)
        # Idempotent re-run: a file already converted (output present, source
        # unchanged) is skipped. Reserve its output name so a different input
        # can't pick it.
        if (done := rl.done_output(path)) is not None:
            skipped += 1
            used_outputs.add(str(done))
            print(f"[{i}/{total}] skip (done) {rel}", file=sys.stderr)
            rl.record(path, kind="ppt", output=done, status="skip")
            continue

        # Reuse the prior output path when reprocessing (force / changed) so we
        # overwrite in place rather than minting a duplicate ``name-1.pdf``.
        planned = rl.planned_output(path)
        if planned is not None:
            out_path = planned
            used_outputs.add(str(out_path))
        else:
            out_path = _unique_out_path(dst / rel.parent, path.stem, used_outputs)

        print(f"[{i}/{total}] converting {rel} -> {out_path.name}", file=sys.stderr)
        item_started = time.monotonic()
        try:
            _convert_one(path, out_path, soffice=soffice, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001 — per-file isolation by design
            failed += 1
            print(f"error: failed to convert {path}: {exc}", file=sys.stderr)
            rl.record(path, kind="ppt", output=out_path, status="fail", error=exc)
            continue
        converted += 1
        rl.record(path, kind="ppt", output=out_path, status="ok",
                  duration_s=round(time.monotonic() - item_started, 3))

    elapsed = time.monotonic() - started
    skipped_part = f", skipped {skipped} (done)" if skipped else ""
    print(
        f"Converted {converted}, failed {failed}{skipped_part} "
        f"in {elapsed:.1f}s.",
        file=sys.stderr,
    )
    exit_code = 0 if failed == 0 else 1
    rl.finish(exit_code=exit_code)
    return exit_code


def _discover_presentations(root: Path) -> list[Path]:
    """Every regular file under ``root`` whose extension is a presentation
    format (case-insensitive), sorted deterministically by relative path."""
    results: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in PPT_EXTS:
            results.append(path)
    results.sort(key=lambda p: str(p.relative_to(root)))
    return results


def _unique_out_path(out_dir: Path, stem: str, used: set[str]) -> Path:
    """Reserve a collision-free ``<out_dir>/<stem>.pdf``.

    Two inputs in the same output directory that share a stem (``talk.ppt`` +
    ``talk.pptx`` -> both want ``talk.pdf``) must not overwrite each other, so
    the second and later get ``-1``, ``-2``, ... and a WARN. Both this run's
    reservations and files already on disk (e.g. from an untracked prior run)
    are avoided.
    """
    candidate = out_dir / f"{stem}.pdf"
    n = 0
    while str(candidate) in used or candidate.exists():
        n += 1
        candidate = out_dir / f"{stem}-{n}.pdf"
    if n:
        print(
            f"WARN: name collision for {stem!r} in {out_dir}; "
            f"writing {candidate.name}",
            file=sys.stderr,
        )
    used.add(str(candidate))
    return candidate


def _find_soffice(arg: str | None) -> str | None:
    """Locate the LibreOffice ``soffice`` binary, or return ``None``.

    Order: explicit ``--soffice`` arg, ``SOFFICE`` env var, ``soffice`` /
    ``libreoffice`` on ``PATH``, then well-known install locations. An explicit
    arg that does not exist returns ``None`` so the caller reports it cleanly.
    """
    if arg:
        p = Path(arg).expanduser()
        return str(p) if p.exists() else None
    env = os.environ.get("SOFFICE")
    if env and Path(env).expanduser().exists():
        return str(Path(env).expanduser())
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in _SOFFICE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _convert_one(src_path: Path, out_path: Path, *, soffice: str, timeout: int) -> None:
    """Render ``src_path`` to ``out_path`` via LibreOffice headless.

    Raises :class:`ConversionError` on any failure (timeout, soffice error, or
    no/empty output). On success the PDF is at ``out_path``; on failure nothing
    is left at ``out_path`` (the temp output is only moved into place once it is
    confirmed valid).

    LibreOffice can't be told the output filename — ``--convert-to pdf`` writes
    ``<stem>.pdf`` into ``--outdir`` — so we convert into a private temp dir and
    move the result. Each call also gets an isolated ``UserInstallation``
    profile so concurrent / GUI-conflicting instances don't clash, and soffice
    is known to exit 0 without producing output, so success is decided by the
    PDF actually existing and being non-empty.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ppt2pdf-") as td:
        tmp = Path(td)
        outdir = tmp / "out"
        outdir.mkdir()
        profile = tmp / "profile"
        cmd = [
            soffice,
            "--headless",
            "--norestore",
            "--nolockcheck",
            f"-env:UserInstallation=file://{profile}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(outdir),
            str(src_path),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise ConversionError(f"soffice timed out after {timeout}s") from exc
        except OSError as exc:
            raise ConversionError(f"could not run soffice: {exc}") from exc

        produced = outdir / f"{src_path.stem}.pdf"
        if not produced.exists() or produced.stat().st_size == 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            tail = detail[-300:] if detail else f"exit code {proc.returncode}"
            raise ConversionError(f"no PDF produced ({tail})")

        # Move the confirmed-valid PDF into place (cross-filesystem safe).
        shutil.move(str(produced), str(out_path))


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert PowerPoint-family files under <input-folder> "
            "to PDFs in <output-folder>, mirroring the input's subdirectory "
            "layout, via LibreOffice headless. The input tree is never "
            "modified. Standalone — not part of the indexer."
        ),
    )
    parser.add_argument("input_folder", help="Folder to scan for presentation files")
    parser.add_argument("output_folder", help="Folder to write the PDFs into")
    parser.add_argument(
        "--soffice",
        default=None,
        help="Path to the LibreOffice 'soffice' binary (default: auto-detect "
        "via PATH, SOFFICE env var, and common install locations).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-file soffice timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be converted and exit (writes nothing).",
    )
    _runlog.add_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
