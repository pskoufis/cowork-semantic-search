#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
drive_analyze.py — Size & file-type distribution for a directory tree.

Usage:
    # Quick mode: walks the filesystem directly
    uv run drive_analyze.py /Volumes/YourDrive

    # Use rmlint JSON output (also reports duplicate stats)
    uv run drive_analyze.py /Volumes/YourDrive --rmlint-json rmlint.json

Outputs:
    - Console: summary tables (size buckets, top extensions, top dirs by size)
    - analysis.csv: one row per file (path, size, ext, mtime)
    - analysis_report.md: human-readable report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------- Size buckets (in bytes) ----------
SIZE_BUCKETS = [
    ("0 B",            0,              0),
    ("1 B – 1 KB",     1,              1024),
    ("1 KB – 10 KB",   1024,           10 * 1024),
    ("10 KB – 100 KB", 10 * 1024,      100 * 1024),
    ("100 KB – 1 MB",  100 * 1024,     1024**2),
    ("1 MB – 10 MB",   1024**2,        10 * 1024**2),
    ("10 MB – 100 MB", 10 * 1024**2,   100 * 1024**2),
    ("100 MB – 1 GB",  100 * 1024**2,  1024**3),
    ("1 GB – 10 GB",   1024**3,        10 * 1024**3),
    ("10 GB +",        10 * 1024**3,   float("inf")),
]

# Roughly group extensions into families for a higher-level view
EXT_FAMILIES = {
    "image":    {".jpg", ".jpeg", ".png", ".heic", ".heif", ".gif", ".tiff", ".tif",
                 ".bmp", ".webp", ".raw", ".cr2", ".nef", ".arw", ".dng", ".svg"},
    "video":    {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".mpg",
                 ".mpeg", ".flv", ".3gp", ".mts", ".m2ts"},
    "audio":    {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".wma", ".aiff", ".opus"},
    "document": {".pdf", ".doc", ".docx", ".odt", ".rtf", ".pages", ".txt", ".md", ".tex"},
    "sheet":    {".xls", ".xlsx", ".csv", ".tsv", ".ods", ".numbers"},
    "slides":   {".ppt", ".pptx", ".key", ".odp"},
    "archive":  {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".dmg", ".iso"},
    "code":     {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt",
                 ".swift", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".sh", ".sql",
                 ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css"},
    "email":    {".pst", ".ost", ".mbox", ".eml", ".msg", ".olm", ".nsf"},
    "design":   {".psd", ".ai", ".sketch", ".fig", ".xd", ".indd"},
    "data":     {".parquet", ".arrow", ".feather", ".h5", ".hdf5", ".pkl", ".npy", ".npz"},
}
EXT_TO_FAMILY = {ext: family for family, exts in EXT_FAMILIES.items() for ext in exts}

# Email-archive extensions — these get their own dedicated section in the report
# because PST/OST files in particular are often the single largest files on a drive
# and warrant individual attention (potential dedupe targets, conversion candidates).
EMAIL_ARCHIVE_EXTS = {".pst", ".ost", ".mbox", ".eml", ".msg", ".olm", ".nsf"}


def human(n: float) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} EB"


def bucket_for(size: int) -> str:
    for label, lo, hi in SIZE_BUCKETS:
        if lo <= size <= hi:
            return label
    return "?"


def walk_filesystem(root: Path, follow_symlinks: bool = False):
    """Yield (path, size, mtime) for every regular file under root."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        # Skip macOS junk that bloats counts without being meaningful
        dirnames[:] = [d for d in dirnames if d not in {".Spotlight-V100", ".Trashes",
                                                          ".fseventsd", ".TemporaryItems"}]
        for name in filenames:
            if name == ".DS_Store":
                continue
            p = Path(dirpath) / name
            try:
                st = p.lstat()
                if not (st.st_mode & 0o170000) == 0o100000:  # regular file
                    continue
                yield p, st.st_size, st.st_mtime
            except (OSError, PermissionError):
                continue


def load_rmlint_json(path: Path):
    """Yield (path, size, mtime, is_duplicate, is_original) from rmlint JSON."""
    with path.open() as f:
        data = json.load(f)
    # rmlint json: first element is header, last is footer, middle are findings
    for entry in data:
        if entry.get("type") != "duplicate_file":
            continue
        yield (
            Path(entry["path"]),
            entry["size"],
            entry.get("mtime", 0),
            True,
            entry.get("is_original", False),
        )


def analyze(records: list[tuple[Path, int, float]], root: Path) -> dict:
    """Build all distributions from a list of file records."""
    total_count = 0
    total_size = 0
    size_buckets = Counter()
    size_buckets_bytes = Counter()
    ext_count = Counter()
    ext_bytes = Counter()
    family_count = Counter()
    family_bytes = Counter()
    top_dirs = defaultdict(int)  # bytes per top-level dir under root

    largest = []  # heap-ish — we'll just sort at the end
    email_archives = []  # every file matching EMAIL_ARCHIVE_EXTS, with metadata

    for path, size, mtime in records:
        total_count += 1
        total_size += size

        bucket = bucket_for(size)
        size_buckets[bucket] += 1
        size_buckets_bytes[bucket] += size

        ext = path.suffix.lower() or "(no ext)"
        ext_count[ext] += 1
        ext_bytes[ext] += size

        family = EXT_TO_FAMILY.get(ext, "other")
        family_count[family] += 1
        family_bytes[family] += size

        # Top-level dir under root
        try:
            rel = path.relative_to(root)
            top = rel.parts[0] if rel.parts else "(root)"
        except ValueError:
            top = "(outside root)"
        top_dirs[top] += size

        largest.append((size, path))

        # Capture every email-archive file individually
        if ext in EMAIL_ARCHIVE_EXTS:
            email_archives.append({"path": path, "size": size, "mtime": mtime, "ext": ext})

    largest.sort(reverse=True)
    email_archives.sort(key=lambda x: -x["size"])
    return {
        "total_count": total_count,
        "total_size": total_size,
        "size_buckets": size_buckets,
        "size_buckets_bytes": size_buckets_bytes,
        "ext_count": ext_count,
        "ext_bytes": ext_bytes,
        "family_count": family_count,
        "family_bytes": family_bytes,
        "top_dirs": top_dirs,
        "largest_files": largest[:50],
        "email_archives": email_archives,
    }


def print_table(title: str, rows: list[tuple], headers: tuple[str, ...]):
    print(f"\n=== {title} ===")
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


def write_report(a: dict, out_path: Path, root: Path, dup_stats: dict | None = None):
    lines = []
    lines.append(f"# Drive analysis: `{root}`\n")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()}_\n")
    lines.append(f"- **Total files:** {a['total_count']:,}")
    lines.append(f"- **Total size:** {human(a['total_size'])}\n")

    if dup_stats:
        lines.append("## Duplicates (from rmlint)\n")
        lines.append(f"- **Duplicate files:** {dup_stats['dup_count']:,}")
        lines.append(f"- **Reclaimable space:** {human(dup_stats['dup_bytes'])}\n")

    lines.append("## Size distribution\n")
    lines.append("| Bucket | Count | Total size | % of total |")
    lines.append("|---|---:|---:|---:|")
    for label, _, _ in SIZE_BUCKETS:
        c = a["size_buckets"].get(label, 0)
        b = a["size_buckets_bytes"].get(label, 0)
        pct = (b / a["total_size"] * 100) if a["total_size"] else 0
        lines.append(f"| {label} | {c:,} | {human(b)} | {pct:.1f}% |")

    lines.append("\n## File-type families\n")
    lines.append("| Family | Count | Total size | % of total |")
    lines.append("|---|---:|---:|---:|")
    for fam, b in sorted(a["family_bytes"].items(), key=lambda x: -x[1]):
        c = a["family_count"][fam]
        pct = (b / a["total_size"] * 100) if a["total_size"] else 0
        lines.append(f"| {fam} | {c:,} | {human(b)} | {pct:.1f}% |")

    lines.append("\n## Top 30 extensions by size\n")
    lines.append("| Extension | Count | Total size |")
    lines.append("|---|---:|---:|")
    for ext, b in sorted(a["ext_bytes"].items(), key=lambda x: -x[1])[:30]:
        lines.append(f"| `{ext}` | {a['ext_count'][ext]:,} | {human(b)} |")

    lines.append("\n## Top-level folders by size\n")
    lines.append("| Folder | Total size |")
    lines.append("|---|---:|")
    for folder, b in sorted(a["top_dirs"].items(), key=lambda x: -x[1])[:30]:
        lines.append(f"| `{folder}` | {human(b)} |")

    lines.append("\n## Top 50 largest files\n")
    lines.append("| Size | Path |")
    lines.append("|---:|---|")
    for size, path in a["largest_files"]:
        lines.append(f"| {human(size)} | `{path}` |")

    # Dedicated email-archive section — PSTs in particular tend to be very large,
    # often single-handedly dominating drive usage, and are worth listing individually.
    archives = a.get("email_archives", [])
    if archives:
        total_email_bytes = sum(e["size"] for e in archives)
        by_ext = Counter()
        bytes_by_ext = Counter()
        for e in archives:
            by_ext[e["ext"]] += 1
            bytes_by_ext[e["ext"]] += e["size"]

        lines.append("\n## Email archive files (.pst, .ost, .mbox, .eml, .msg, .olm, .nsf)\n")
        lines.append(f"- **Total email-archive files:** {len(archives):,}")
        lines.append(f"- **Combined size:** {human(total_email_bytes)} "
                     f"({(total_email_bytes / a['total_size'] * 100) if a['total_size'] else 0:.1f}% of drive)\n")

        lines.append("### Breakdown by extension\n")
        lines.append("| Extension | Count | Total size | Avg size |")
        lines.append("|---|---:|---:|---:|")
        for ext, b in sorted(bytes_by_ext.items(), key=lambda x: -x[1]):
            c = by_ext[ext]
            avg = b / c if c else 0
            lines.append(f"| `{ext}` | {c:,} | {human(b)} | {human(avg)} |")

        lines.append("\n### Every email-archive file (sorted by size)\n")
        lines.append("| Size | Last modified | Path |")
        lines.append("|---:|---|---|")
        for e in archives:
            mtime_str = datetime.fromtimestamp(e["mtime"]).strftime("%Y-%m-%d") if e["mtime"] else "?"
            lines.append(f"| {human(e['size'])} | {mtime_str} | `{e['path']}` |")

    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Directory to analyze")
    ap.add_argument("--rmlint-json", type=Path, default=None,
                    help="Optional rmlint JSON output to merge in duplicate stats")
    ap.add_argument("--csv", type=Path, default=Path("analysis.csv"))
    ap.add_argument("--report", type=Path, default=Path("analysis_report.md"))
    args = ap.parse_args()

    if not args.root.exists():
        print(f"Path does not exist: {args.root}", file=sys.stderr)
        sys.exit(1)

    print(f"Walking {args.root} ...")
    records = []
    with args.csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "size", "ext", "mtime"])
        for i, (p, size, mtime) in enumerate(walk_filesystem(args.root), 1):
            records.append((p, size, mtime))
            w.writerow([str(p), size, p.suffix.lower(), mtime])
            if i % 50_000 == 0:
                print(f"  scanned {i:,} files...")

    print(f"Done. {len(records):,} files. Analyzing ...")
    a = analyze(records, args.root)

    # Optional duplicate stats
    dup_stats = None
    if args.rmlint_json and args.rmlint_json.exists():
        dup_count = 0
        dup_bytes = 0
        for _, size, _, is_dup, is_orig in load_rmlint_json(args.rmlint_json):
            if is_dup and not is_orig:
                dup_count += 1
                dup_bytes += size
        dup_stats = {"dup_count": dup_count, "dup_bytes": dup_bytes}

    # Console tables
    print_table(
        "Size distribution",
        [(lbl, f"{a['size_buckets'].get(lbl,0):,}", human(a['size_buckets_bytes'].get(lbl,0)))
         for lbl, _, _ in SIZE_BUCKETS],
        ("Bucket", "Count", "Total size"),
    )
    print_table(
        "File-type families",
        [(fam, f"{a['family_count'][fam]:,}", human(b))
         for fam, b in sorted(a["family_bytes"].items(), key=lambda x: -x[1])],
        ("Family", "Count", "Total size"),
    )
    print_table(
        "Top 15 extensions by size",
        [(ext, f"{a['ext_count'][ext]:,}", human(b))
         for ext, b in sorted(a["ext_bytes"].items(), key=lambda x: -x[1])[:15]],
        ("Ext", "Count", "Total size"),
    )

    # Always surface email archives in the console — these are often the most
    # impactful files on a drive in terms of size and warrant individual review.
    archives = a.get("email_archives", [])
    if archives:
        rows = []
        for e in archives[:25]:  # cap console output; full list is in the report
            mtime_str = datetime.fromtimestamp(e["mtime"]).strftime("%Y-%m-%d") if e["mtime"] else "?"
            # Truncate path for terminal readability
            p = str(e["path"])
            if len(p) > 70:
                p = "..." + p[-67:]
            rows.append((human(e["size"]), e["ext"], mtime_str, p))
        print_table(
            f"Email archive files ({len(archives)} total, "
            f"{human(sum(e['size'] for e in archives))})",
            rows,
            ("Size", "Ext", "Modified", "Path"),
        )

    write_report(a, args.report, args.root, dup_stats)
    print(f"\nReport written to {args.report}")
    print(f"Raw file list written to {args.csv}")
    if dup_stats:
        print(f"Duplicates found: {dup_stats['dup_count']:,} files, "
              f"{human(dup_stats['dup_bytes'])} reclaimable")


if __name__ == "__main__":
    main()