"""Unified archive extractor — scan a folder for .mbox/.pst/.msg/.zip and
extract every one into an output folder that mirrors the input's layout,
recursing fully into archives revealed inside extracted zips.

Usage:
    python scripts/extract_archives_folder.py <input-folder> <output-folder> \
        [--dry-run] [--max-depth N]

This is a thin orchestrator: it delegates the per-type work to the
existing libraries (``mbox_handling`` / ``msg_handling`` / ``pst_handling``
``ensure_unpacked``) and uses stdlib ``zipfile`` for zips. The only novel
logic here is the cycle-safe recursion loop and the mirror-layout rule.

Layout (output mirrors input):
    .msg   -> <out>/<rel_parent>/<stem>.txt + attachments/<stem>__*
    .mbox  -> <out>/<rel_parent>/<stem>_unpacked/   (thread tree)
    .pst   -> <out>/<rel_parent>/<stem>_unpacked/   (folder tree)
    .zip   -> <out>/<rel_parent>/<stem>/            (internal structure kept)

Recursion: pass 1 scans the input tree; later passes scan the *output*
tree for archives revealed by zip extraction and process them in place.
A ``processed`` set (resolved paths) guarantees each archive is handled
once — this is what makes zips (which have no mtime idempotency)
terminate. ``--max-depth`` is a hard backstop against self-similar
archives.

Invariants:
    * The input tree is never modified — all writes go to output.
    * The output folder must not live inside the input folder (that would
      be a feedback loop, since later passes scan the output).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Allow ``python scripts/extract_archives_folder.py`` (no ``-m``) by putting
# the repo root on sys.path before importing project code — same trick as the
# sibling unpack_*_folder.py scripts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mbox_handling.unpack import ensure_unpacked as _unpack_mbox  # noqa: E402
from msg_handling.unpack import ensure_unpacked as _unpack_msg  # noqa: E402
from pst_handling.unpack import ensure_unpacked as _unpack_pst  # noqa: E402
from scripts._runlog import RunLog  # noqa: E402
from scripts import _runlog  # noqa: E402

# Suffix (lowercased) -> archive kind. Order is irrelevant; lookup is by suffix.
_KINDS = {".mbox": "mbox", ".pst": "pst", ".msg": "msg", ".zip": "zip",
          ".rar": "rar"}

# Secondary volumes of a multi-volume rar set (foo.part2.rar, foo.part02.rar,
# ...). Only the first volume is processed — it pulls in the whole set, while
# opening a later volume directly errors. Matches ``.partN.rar`` with N > 1.
_RAR_SECONDARY_VOLUME = re.compile(r"\.part0*(\d+)\.rar$", re.IGNORECASE)


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

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    return _run(src, dst, dry_run=args.dry_run, max_depth=args.max_depth,
                copy_others=args.copy_others, argv=argv,
                log_dir=args.log_dir, force=args.force,
                runlog_enabled=not args.no_runlog)


def _run(src: Path, dst: Path, *, dry_run: bool, max_depth: int,
         copy_others: bool = False, argv: list[str] | None = None,
         log_dir: str | None = None, force: bool = False,
         runlog_enabled: bool = True) -> int:
    processed: set[Path] = set()
    counts = {"mbox": 0, "pst": 0, "msg": 0, "zip": 0}
    failed = 0
    skipped = 0
    copied = 0
    started = time.monotonic()

    # A run-log is pointless for a dry run (nothing is written). Otherwise it
    # records per-archive status and drives the idempotent re-run.
    rl = RunLog(
        "extract_archives_folder",
        input_root=src,
        output_root=dst,
        argv=argv if argv is not None else sys.argv[1:],
        log_dir=log_dir,
        force=force,
        enabled=runlog_enabled and not dry_run,
    )

    # Surface multi-volume .rar sets whose first volume is missing — otherwise
    # the secondary volumes are silently skipped and nothing is extracted.
    _warn_orphan_secondary_rars(src)

    # With --copy-others, mirror every non-archive file from the input tree
    # into the output so the result is a complete index-ready mirror. Only
    # the input tree is copied: files revealed inside extracted zips already
    # live in the output tree.
    if copy_others:
        copied, copy_failed = _copy_other_files(src, dst, dry_run=dry_run)
        failed += copy_failed

    # Pass 1 scans the input tree (target mirrors input). Every later pass
    # scans the output tree for archives revealed by a zip extraction and
    # processes them where they sit.
    for depth in range(1, max_depth + 1):
        scan_root = src if depth == 1 else dst
        new = [
            (p, kind)
            for p, kind in _discover(scan_root)
            if p not in processed
        ]
        if not new:
            break
        for path, kind in new:
            processed.add(path)
            target = _target_for(path, kind, from_input=(depth == 1),
                                  src=src, dst=dst)
            rel = path.relative_to(scan_root)
            if dry_run:
                print(f"[depth {depth}] {kind}: {rel} -> "
                      f"{target.relative_to(dst) if _is_inside(target, dst) else target}",
                      file=sys.stderr)
                counts[kind] += 1
                continue
            # Idempotent re-run: skip an archive already extracted (output
            # present, source unchanged). The ledger is loaded once before the
            # loop, so a nested archive freshly revealed *this* run is never
            # mistaken for done.
            if (done := rl.done_output(path)) is not None:
                skipped += 1
                print(f"[depth {depth}] skip (done) {kind}: {rel}",
                      file=sys.stderr)
                rl.record(path, kind=kind, output=done, status="skip")
                continue
            item_started = time.monotonic()
            try:
                _process(path, kind, target)
            except Exception as exc:  # noqa: BLE001 — per-file isolation
                failed += 1
                print(f"error: failed to extract {path}: {exc}",
                      file=sys.stderr)
                _cleanup_partial(kind, target, dst)
                # output=None: the partial was cleaned up, so a re-run recomputes
                # the deterministic target and retries.
                rl.record(path, kind=kind, output=None, status="fail", error=exc)
                continue
            counts[kind] += 1
            rl.record(path, kind=kind, output=_record_output(kind, target, path),
                      status="ok",
                      duration_s=round(time.monotonic() - item_started, 3))
    else:
        # Loop exhausted max_depth without a clean break.
        if _remaining(dst, processed):
            print(
                f"warning: reached --max-depth {max_depth}; some archives "
                "may remain unextracted.",
                file=sys.stderr,
            )

    elapsed = time.monotonic() - started
    copied_part = f" {copied} other file(s) copied;" if copy_others else ""
    skipped_part = f" {skipped} skipped;" if skipped else ""
    print(
        f"Done in {elapsed:.1f}s: "
        f"{counts['zip']} zip, {counts['mbox']} mbox, "
        f"{counts['pst']} pst, {counts['msg']} msg extracted;"
        f"{copied_part}{skipped_part} "
        f"{failed} failed.",
        file=sys.stderr,
    )
    exit_code = 0 if failed == 0 else 1
    rl.finish(exit_code=exit_code)
    return exit_code


def _record_output(kind: str, target: Path, path: Path) -> Path:
    """The output path to record for the ledger. For ``msg`` the target is the
    shared parent dir, so the meaningful per-input output is its ``.txt``."""
    if kind == "msg":
        return target / f"{path.stem}.txt"
    return target


def _cleanup_partial(kind: str, target: Path, dst: Path) -> None:
    """Remove a partially-written extraction so a re-run starts clean.

    Only dedicated output dirs (zip/rar/mbox/pst) are removed. A ``msg`` writes
    into a *shared* parent directory alongside sibling messages, so its target
    must never be deleted. Mirrors ``extract_zips.py``'s cleanup-on-failure;
    note that re-extracting a *changed* archive in place means a mid-way failure
    also drops the prior good extraction — acceptable, same as that script.
    """
    if kind == "msg":
        return
    try:
        if _is_inside(target, dst) and target.exists():
            shutil.rmtree(target, ignore_errors=True)
    except OSError:
        pass


def _warn_orphan_secondary_rars(root: Path) -> None:
    """Warn for any multi-volume ``.rar`` set under ``root`` whose first volume
    is absent — its secondary ``.partN.rar`` (N>1) volumes are skipped by
    discovery, so without this the whole set is silently missed.

    Volumes are grouped by their base name (the part before ``.part``); a group
    whose lowest volume number is >1 is missing its first volume.
    """
    if not root.exists():
        return
    groups: dict[Path, list[int]] = {}
    for p in root.rglob("*"):
        if not p.is_file() or _is_macos_junk(p):
            continue
        m = _RAR_SECONDARY_VOLUME.search(p.name)
        if m is None:
            continue
        # Slice at the matched ``.partN.rar`` position so a base name that
        # itself contains ``.part`` (e.g. ``spare.parts.part2.rar``) groups
        # correctly.
        base = p.parent / p.name[: m.start()]
        groups.setdefault(base, []).append(int(m.group(1)))
    for base, numbers in sorted(groups.items()):
        if min(numbers) > 1:
            print(
                f"warning: multi-volume rar set {base.name!r} is missing its "
                "first volume; the set will not be extracted.",
                file=sys.stderr,
            )


def _process(path: Path, kind: str, target: Path) -> None:
    """Dispatch a single archive to its extractor. ``target`` is the
    fully-resolved destination computed by ``_target_for``."""
    if kind == "zip":
        _extract_zip(path, target)
    elif kind == "rar":
        _extract_rar(path, target)
    elif kind == "msg":
        _unpack_msg(path, target=target)
        # ``msg_handling.ensure_unpacked`` swallows parse failures (logs to
        # stderr, writes nothing) rather than raising. Treat a missing
        # ``.txt`` as a failure so it surfaces in the count and exit code —
        # same guard as scripts/unpack_msg_folder.py.
        if not (target / f"{path.stem}.txt").is_file():
            raise RuntimeError(
                f"no .txt produced for {path.name} (parse failure)"
            )
    elif kind == "mbox":
        _unpack_mbox(path, target=target)
    elif kind == "pst":
        _unpack_pst(path, target=target)
    else:  # pragma: no cover — _discover only yields known kinds
        raise ValueError(f"unknown kind: {kind}")


def _target_for(path: Path, kind: str, *, from_input: bool,
                src: Path, dst: Path) -> Path:
    """Where this archive's output goes.

    First-pass (input) files mirror into ``<dst>/<rel_parent>/``. Files
    discovered inside the output tree on later passes are processed in
    place — their target parent is their own parent directory.
    """
    if from_input:
        parent = dst / path.parent.relative_to(src)
    else:
        parent = path.parent

    if kind in ("zip", "rar"):
        return parent / path.stem
    if kind == "msg":
        return parent
    # mbox / pst
    return parent / f"{path.stem}_unpacked"


def _extract_zip(zip_path: Path, target: Path) -> None:
    """Extract ``zip_path`` into ``target`` preserving internal structure.

    Each entry is path-traversal guarded: a member whose resolved
    destination escapes ``target`` aborts the whole zip (defense against a
    malicious archive). Directory entries are created; file entries are
    written under their in-zip relative path.
    """
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if dest != target and target not in dest.parents:
                raise ValueError(
                    f"unsafe path in {zip_path.name}: {member.filename!r} "
                    "escapes the extraction directory"
                )
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Stream rather than read the whole member into memory — a large
            # entry would otherwise spike RAM (bug #5).
            with zf.open(member) as srcf, open(dest, "wb") as outf:
                shutil.copyfileobj(srcf, outf)


def _extract_rar(rar_path: Path, target: Path) -> None:
    """Extract ``rar_path`` into ``target`` preserving internal structure.

    A near-clone of :func:`_extract_zip`, including the per-member
    path-traversal guard. ``.rar`` is not in the stdlib: it needs the
    optional ``rarfile`` package plus a system backend (``unar``/``unrar``;
    the already-present ``bsdtar`` also works for many archives). Both the
    missing-package and missing-backend cases raise a friendly hint, caught
    per-file by the caller's error isolation.
    """
    try:
        import rarfile
    except ImportError as exc:  # package not installed
        raise RuntimeError(
            "rar support needs the 'rarfile' package: "
            "pip install 'cowork-semantic-search[rar]'"
        ) from exc

    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    try:
        with rarfile.RarFile(rar_path) as rf:
            for member in rf.infolist():
                dest = (target / member.filename).resolve()
                if dest != target and target not in dest.parents:
                    raise ValueError(
                        f"unsafe path in {rar_path.name}: "
                        f"{member.filename!r} escapes the extraction directory"
                    )
                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with rf.open(member) as srcf, open(dest, "wb") as outf:
                    shutil.copyfileobj(srcf, outf)
    except rarfile.RarCannotExec as exc:  # no usable backend tool
        raise RuntimeError(
            "rarfile found no usable backend; install one, "
            "e.g. 'brew install unar'"
        ) from exc


def _copy_other_files(src: Path, dst: Path, *, dry_run: bool) -> tuple[int, int]:
    """Copy every non-archive file under ``src`` into ``dst`` preserving the
    relative path. Archive types (handled by extraction) are skipped. A
    destination is overwritten only when the source is newer (mtime), so
    re-runs are idempotent. Returns ``(copied, failed)``.
    """
    copied = 0
    failed = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        if _is_macos_junk(p):
            continue  # AppleDouble / .DS_Store / __MACOSX — never content
        if p.suffix.lower() in _KINDS:
            continue  # archives are extracted, not copied
        rel = p.relative_to(src)
        dest = dst / rel
        if dest.exists() and p.stat().st_mtime <= dest.stat().st_mtime:
            continue  # up to date — nothing to do
        if dry_run:
            print(f"copy: {rel}", file=sys.stderr)
            copied += 1
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        except Exception as exc:  # noqa: BLE001 — per-file isolation
            failed += 1
            print(f"error: failed to copy {p}: {exc}", file=sys.stderr)
            continue
        copied += 1
    return copied, failed


def _discover(root: Path) -> list[tuple[Path, str]]:
    """All .mbox/.pst/.msg/.zip files under ``root`` (recursive, by suffix),
    paired with their kind. Skips anything that is not a regular file (so an
    mbox-as-directory does not get treated as a file)."""
    found: list[tuple[Path, str]] = []
    if not root.exists():
        return found
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if _is_macos_junk(p):
            continue  # AppleDouble / .DS_Store / __MACOSX — not real archives
        kind = _KINDS.get(p.suffix.lower())
        if kind is None:
            continue
        if kind == "rar" and _is_secondary_rar_volume(p):
            continue  # only the first volume of a multi-volume set
        found.append((p.resolve(), kind))
    return found


def _is_macos_junk(path: Path) -> bool:
    """True for macOS filesystem metadata that is never real content:
    AppleDouble ``._*`` sidecars (created on non-HFS volumes), ``.DS_Store``,
    and anything inside a ``__MACOSX/`` directory."""
    if path.name.startswith("._") or path.name == ".DS_Store":
        return True
    return "__MACOSX" in path.parts


def _is_secondary_rar_volume(path: Path) -> bool:
    """True for ``foo.partN.rar`` with N > 1 — a continuation volume that
    must not be opened directly (the first volume pulls in the whole set)."""
    m = _RAR_SECONDARY_VOLUME.search(path.name)
    return m is not None and int(m.group(1)) > 1


def _remaining(root: Path, processed: set[Path]) -> bool:
    return any(p not in processed for p, _ in _discover(root))


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan <input-folder> for .mbox/.pst/.msg/.zip and extract each "
            "into <output-folder>, mirroring the input's subdirectory "
            "layout. Recurses fully into archives revealed inside zips. "
            "The input tree is never modified."
        ),
    )
    parser.add_argument("input_folder", help="Folder to scan for archives")
    parser.add_argument(
        "output_folder",
        help="Folder to write the extracted/unpacked tree into",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned extractions without writing anything.",
    )
    parser.add_argument(
        "--copy-others", action="store_true",
        help="Also copy every non-archive file from the input tree into the "
             "output, mirroring its layout (overwriting only when the source "
             "is newer). Makes the output a complete mirror of the input.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Maximum recursion passes (backstop against self-similar "
             "archives). Default: 10.",
    )
    _runlog.add_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
