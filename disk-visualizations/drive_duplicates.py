#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly>=5.20,<7.0",
# ]
# ///
"""
drive_duplicates.py — Find duplicate files and visualize reclaimable space.

READ-ONLY: this script never renames, moves, or deletes any file. It only
reads file contents to compute hashes and writes an HTML report.

Algorithm (two-stage, fast):
    1. Group all files by exact size.
    2. For groups with 2+ files, hash each (BLAKE2b, 1MB chunks).
    3. Files sharing both size AND hash form a duplicate cluster.
    4. Reclaimable bytes per cluster = (N - 1) * file_size, where N is the
       number of identical copies (you'd keep one, delete the rest — but
       this script does NOT perform that deletion).

Usage:
    uv run drive_duplicates.py /Users/me/Desktop/Petros
    uv run drive_duplicates.py /path --min-size 10K --out dups.html

Views:
    1. Summary cards (clusters, dup files, reclaimable bytes, % of drive)
    2. Top-50 duplicate clusters by reclaimable size (table)
    3. Reclaimable bytes by file-type family (horizontal bar)
    4. Treemap of folders where duplicates concentrate
    5. Size bucket distribution of duplicate clusters
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Shared constants (kept in sync with drive_visualize.py) ----------
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

FAMILY_COLORS = {
    "image":    "#4C9AFF", "video":    "#FF5630", "audio":    "#FFAB00",
    "document": "#36B37E", "sheet":    "#00B8D9", "slides":   "#6554C0",
    "archive":  "#8777D9", "code":     "#998DD9", "email":    "#FF8B00",
    "design":   "#FF7452", "data":     "#00C7E6", "other":    "#A5ADBA",
}

HASH_CHUNK = 1024 * 1024  # 1 MB


def human(n: float) -> str:
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


def parse_size(s: str) -> int:
    """Parse human-friendly size like '10K', '1.5M', '100' (bytes)."""
    m = re.match(r"^([\d.]+)\s*([KMGT]?)B?$", s.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"can't parse size: {s}")
    n = float(m.group(1))
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[m.group(2).upper()]
    return int(n * mult)


# ---------- File walking (same logic as drive_visualize.py) ----------

def walk_filesystem(root: Path, follow_symlinks: bool = False):
    """Yield (Path, size, mtime, dev, inode) for every regular file under root."""
    skip_dirs = {".Spotlight-V100", ".Trashes", ".fseventsd", ".TemporaryItems"}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if name == ".DS_Store":
                continue
            p = Path(dirpath) / name
            try:
                st = p.lstat()
                if (st.st_mode & 0o170000) != 0o100000:
                    continue
                yield p, st.st_size, st.st_mtime, st.st_dev, st.st_ino
            except (OSError, PermissionError):
                continue


# ---------- Hashing ----------

def hash_file(path: Path) -> str | None:
    """BLAKE2b-128 of file contents, or None if unreadable."""
    h = hashlib.blake2b(digest_size=16)
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(HASH_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError):
        return None


def find_duplicates(records, min_size: int, max_size: int | None = None):
    """Two-stage duplicate detection.

    Returns:
        clusters: list of lists, each inner list is [(path, size, mtime), ...]
                  with 2+ entries that share identical content.
        stats: dict with counters about the scan.
    """
    by_size: dict[int, list] = defaultdict(list)
    skipped_small = 0
    skipped_large = 0
    skipped_hardlink = 0
    seen_inodes: set[tuple[int, int]] = set()

    for path, size, mtime, dev, ino in records:
        if size < min_size:
            skipped_small += 1
            continue
        if max_size is not None and size > max_size:
            skipped_large += 1
            continue
        # Hardlinks point to the same on-disk bytes — they always hash identically
        # but deleting one wouldn't actually reclaim space. Drop all-but-one per inode.
        key = (dev, ino)
        if key in seen_inodes:
            skipped_hardlink += 1
            continue
        seen_inodes.add(key)
        by_size[size].append((path, mtime))

    candidate_groups = [g for g in by_size.values() if len(g) >= 2]
    n_candidates = sum(len(g) for g in candidate_groups)
    print(f"Stage 1: {len(by_size):,} unique sizes, "
          f"{len(candidate_groups):,} size-collisions = "
          f"{n_candidates:,} candidate files to hash")
    if skipped_small:
        print(f"  (skipped {skipped_small:,} files below min-size)")
    if skipped_hardlink:
        print(f"  (skipped {skipped_hardlink:,} hardlinks)")

    by_hash: dict[str, list] = defaultdict(list)
    hashed = 0
    failed = 0
    for size, group in by_size.items():
        if len(group) < 2:
            continue
        for path, mtime in group:
            h = hash_file(path)
            hashed += 1
            if h is None:
                failed += 1
                continue
            by_hash[(h, size)].append((path, size, mtime))
            if hashed % 200 == 0:
                print(f"  hashed {hashed:,} / {n_candidates:,}")

    clusters = [files for files in by_hash.values() if len(files) >= 2]
    # Sort by reclaimable bytes (descending)
    clusters.sort(key=lambda c: -((len(c) - 1) * c[0][1]))

    stats = {
        "n_files_scanned": len(records),
        "n_candidates": n_candidates,
        "n_hashed": hashed,
        "n_hash_failed": failed,
        "n_clusters": len(clusters),
        "n_dup_files": sum(len(c) for c in clusters),
        "reclaimable_bytes": sum((len(c) - 1) * c[0][1] for c in clusters),
        "skipped_small": skipped_small,
        "skipped_large": skipped_large,
        "skipped_hardlink": skipped_hardlink,
    }
    return clusters, stats


# ---------- Views ----------

def build_cluster_table(clusters, n: int = 50) -> go.Figure:
    """Top-N clusters by reclaimable bytes."""
    if not clusters:
        fig = go.Figure()
        fig.update_layout(title="No duplicate clusters found", height=200)
        return fig

    rank, copies, file_size, reclaim, family, ext, sample_paths = [], [], [], [], [], [], []
    for i, cluster in enumerate(clusters[:n], 1):
        size = cluster[0][1]
        recl = (len(cluster) - 1) * size
        ex = cluster[0][0].suffix.lower() or "(none)"
        fam = EXT_TO_FAMILY.get(ex, "other")
        # Show up to 4 sample paths in the cell
        paths_str = "<br>".join(str(p) for p, _, _ in cluster[:4])
        if len(cluster) > 4:
            paths_str += f"<br>… +{len(cluster) - 4} more"
        rank.append(i)
        copies.append(len(cluster))
        file_size.append(human(size))
        reclaim.append(human(recl))
        family.append(fam)
        ext.append(ex)
        sample_paths.append(paths_str)

    fig = go.Figure(go.Table(
        header=dict(
            values=["#", "Copies", "File size", "Reclaimable", "Family", "Ext", "Sample paths"],
            fill_color="#22272E",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[rank, copies, file_size, reclaim, family, ext, sample_paths],
            align=["right", "right", "right", "right", "center", "center", "left"],
            height=None,  # autosize for multi-line paths
            font=dict(size=11),
        ),
        columnwidth=[30, 50, 80, 100, 70, 50, 700],
    ))
    fig.update_layout(
        title=f"Top {min(n, len(clusters))} duplicate clusters by reclaimable space",
        height=min(800, 100 + 50 * min(n, len(clusters))),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return fig


def build_family_view(clusters) -> go.Figure:
    """Reclaimable bytes by file-type family."""
    family_reclaim: Counter[str] = Counter()
    family_count: Counter[str] = Counter()
    for cluster in clusters:
        size = cluster[0][1]
        recl = (len(cluster) - 1) * size
        ex = cluster[0][0].suffix.lower()
        fam = EXT_TO_FAMILY.get(ex, "other")
        family_reclaim[fam] += recl
        family_count[fam] += len(cluster) - 1  # number of removable copies

    if not family_reclaim:
        fig = go.Figure()
        fig.update_layout(title="No duplicates to summarize by family", height=200)
        return fig

    fams = sorted(family_reclaim.items(), key=lambda x: -x[1])
    labels = [f for f, _ in fams]
    values = [b for _, b in fams]
    colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"]) for f in labels]
    text = [human(b) for b in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, text=text, textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{text}<br>%{customdata:,} removable copies<extra></extra>",
        customdata=[family_count[f] for f in labels],
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Reclaimable bytes")
    fig.update_layout(
        title="Reclaimable bytes by file-type family",
        height=350,
        margin=dict(t=50, l=10, r=10, b=40),
    )
    return fig


def build_concentration_treemap(clusters, root: Path, top_n_dirs: int = 300) -> go.Figure:
    """Treemap of folders sized by 'reclaimable bytes from duplicates within them'.

    Each non-original copy contributes its size to every ancestor directory.
    This surfaces the folders where dedup would free the most space.
    """
    root_str = str(root)
    dir_reclaim: Counter[str] = Counter()
    dir_dup_count: Counter[str] = Counter()

    for cluster in clusters:
        size = cluster[0][1]
        # All but one in cluster are "removable" — each contributes 'size' bytes
        for path, _, _ in cluster[1:]:
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts[:-1]
            for i in range(len(parts) + 1):
                ancestor = root_str if i == 0 else os.path.join(root_str, *parts[:i])
                dir_reclaim[ancestor] += size
                dir_dup_count[ancestor] += 1

    if not dir_reclaim:
        fig = go.Figure()
        fig.update_layout(title="No duplicate concentrations to plot", height=200)
        return fig

    sorted_dirs = sorted(dir_reclaim.items(), key=lambda x: -x[1])
    kept = {root_str}
    for d, _ in sorted_dirs[:top_n_dirs]:
        kept.add(d)

    ids, labels, parents, values, hover = [], [], [], [], []
    for d, b in dir_reclaim.items():
        if d not in kept:
            continue
        ids.append(d)
        labels.append(Path(d).name if d != root_str else (root.name or root_str))
        parent = "" if d == root_str else str(Path(d).parent)
        parents.append("" if d == root_str else (parent if parent in kept else root_str))
        values.append(b)
        hover.append(
            f"<b>{labels[-1]}</b><br>"
            f"Reclaimable: {human(b)}<br>"
            f"Removable copies: {dir_dup_count[d]:,}"
        )

    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=["#FF5630"] * len(ids), line=dict(width=0.5, color="white")),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        maxdepth=3,
    ))
    fig.update_layout(
        title=f"Folders by reclaimable bytes from duplicates (top {top_n_dirs} shown)",
        margin=dict(t=50, l=10, r=10, b=10),
        height=600,
    )
    return fig


def build_cluster_size_buckets(clusters) -> go.Figure:
    """Distribution of duplicate clusters across size buckets — are the wins
    big-file dups (few clusters, lots of bytes) or small-file dups (many
    clusters, few bytes)?"""
    bucket_clusters: Counter[str] = Counter()
    bucket_reclaim: Counter[str] = Counter()
    for cluster in clusters:
        size = cluster[0][1]
        recl = (len(cluster) - 1) * size
        b = bucket_for(size)
        bucket_clusters[b] += 1
        bucket_reclaim[b] += recl

    labels = [b[0] for b in SIZE_BUCKETS]
    counts = [bucket_clusters.get(l, 0) for l in labels]
    bytes_ = [bucket_reclaim.get(l, 0) for l in labels]
    bytes_human = [human(b) for b in bytes_]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Clusters per size bucket",
                                        "Reclaimable bytes per size bucket"))
    fig.add_trace(
        go.Bar(x=labels, y=counts, marker_color="#FF5630",
               hovertemplate="<b>%{x}</b><br>%{y:,} clusters<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=bytes_, marker_color="#FF8B00",
               text=bytes_human, textposition="outside",
               hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"),
        row=1, col=2,
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(title="Duplicate clusters by individual-file size",
                      showlegend=False, height=420,
                      margin=dict(t=60, l=10, r=10, b=80))
    return fig


# ---------- HTML assembly ----------

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Duplicate report — {root}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 24px; background: #F4F5F7; color: #172B4D;
    max-width: 1400px; margin-left: auto; margin-right: auto;
  }}
  h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
  .meta {{ color: #5E6C84; font-size: 13px; margin-bottom: 24px; }}
  .banner {{
    background: #E3FCEF; border-left: 4px solid #00875A;
    padding: 12px 16px; border-radius: 6px; font-size: 13px;
    color: #006644; margin-bottom: 16px;
  }}
  .card {{
    background: white; border-radius: 8px; padding: 16px;
    margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .summary {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .summary div {{ flex: 1; min-width: 180px; }}
  .summary .num {{ font-size: 28px; font-weight: 600; color: #FF5630; }}
  .summary .num.green {{ color: #00875A; }}
  .summary .lbl {{ font-size: 12px; color: #5E6C84; text-transform: uppercase; letter-spacing: 0.5px; }}
  footer {{ color: #97A0AF; font-size: 11px; text-align: center; margin-top: 24px; }}
</style>
</head>
<body>
<h1>Duplicate report — <code>{root}</code></h1>
<div class="meta">Generated {ts} · {n_files:,} files scanned · {n_hashed:,} hashed</div>
<div class="banner">
  <strong>Read-only inspection.</strong> This report identifies duplicate content and
  estimates reclaimable space. Nothing on disk has been changed.
</div>
<div class="card summary">
  <div><div class="num">{n_clusters:,}</div><div class="lbl">Duplicate clusters</div></div>
  <div><div class="num">{n_dup_files:,}</div><div class="lbl">Duplicate files</div></div>
  <div><div class="num green">{reclaimable}</div><div class="lbl">Reclaimable if deduped</div></div>
  <div><div class="num">{pct:.1f}%</div><div class="lbl">% of scanned bytes</div></div>
</div>
"""

HTML_FOOTER = """
<footer>Built with Plotly · read-only · open in any modern browser.</footer>
</body>
</html>
"""


def render_html(figs, out_path: Path, *, root: Path, n_files: int, n_hashed: int,
                n_clusters: int, n_dup_files: int, reclaimable_bytes: int,
                total_scanned_bytes: int, include_plotlyjs: str = "cdn"):
    pct = (reclaimable_bytes / total_scanned_bytes * 100) if total_scanned_bytes else 0
    parts = [HTML_HEADER.format(
        root=str(root),
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_files=n_files,
        n_hashed=n_hashed,
        n_clusters=n_clusters,
        n_dup_files=n_dup_files,
        reclaimable=human(reclaimable_bytes),
        pct=pct,
    )]
    for i, fig in enumerate(figs):
        plotlyjs = include_plotlyjs if i == 0 else False
        parts.append('<div class="card">')
        parts.append(fig.to_html(full_html=False, include_plotlyjs=plotlyjs))
        parts.append("</div>")
    parts.append(HTML_FOOTER)
    out_path.write_text("\n".join(parts))


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("root", type=Path, help="Directory to scan for duplicates")
    ap.add_argument("--out", type=Path, default=Path("duplicates.html"),
                    help="Output HTML path (default: duplicates.html)")
    ap.add_argument("--min-size", type=str, default="1K",
                    help="Ignore files smaller than this. Examples: 1K, 100K, 10M. Default: 1K")
    ap.add_argument("--max-size", type=str, default=None,
                    help="Ignore files larger than this (mostly useful for testing). Default: no limit")
    ap.add_argument("--top-clusters", type=int, default=50,
                    help="How many duplicate clusters to list in the table (default: 50)")
    ap.add_argument("--top-dirs", type=int, default=300,
                    help="Max directories in the concentration treemap (default: 300)")
    ap.add_argument("--inline-plotly", action="store_true",
                    help="Inline plotly.js into the HTML (no internet needed, larger file)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"Path does not exist: {args.root}", file=sys.stderr)
        sys.exit(1)

    min_size = parse_size(args.min_size)
    max_size = parse_size(args.max_size) if args.max_size else None

    print(f"Walking {args.root} ...")
    records = []
    total_scanned_bytes = 0
    for i, rec in enumerate(walk_filesystem(args.root), 1):
        records.append(rec)
        total_scanned_bytes += rec[1]
        if i % 50_000 == 0:
            print(f"  scanned {i:,} files...")
    print(f"Done. {len(records):,} files. Finding duplicates (min-size={human(min_size)})...")

    clusters, stats = find_duplicates(records, min_size=min_size, max_size=max_size)

    # Console summary
    print()
    print(f"  Clusters found:     {stats['n_clusters']:,}")
    print(f"  Duplicate files:    {stats['n_dup_files']:,}")
    print(f"  Reclaimable:        {human(stats['reclaimable_bytes'])}")
    if stats['n_hash_failed']:
        print(f"  Hash failures:      {stats['n_hash_failed']:,}")
    print()

    print("Building views ...")
    figs = [
        build_cluster_table(clusters, n=args.top_clusters),
        build_family_view(clusters),
        build_concentration_treemap(clusters, args.root, top_n_dirs=args.top_dirs),
        build_cluster_size_buckets(clusters),
    ]

    render_html(
        figs, args.out,
        root=args.root,
        n_files=stats["n_files_scanned"],
        n_hashed=stats["n_hashed"],
        n_clusters=stats["n_clusters"],
        n_dup_files=stats["n_dup_files"],
        reclaimable_bytes=stats["reclaimable_bytes"],
        total_scanned_bytes=total_scanned_bytes,
        include_plotlyjs=True if args.inline_plotly else "cdn",
    )
    print(f"Report written to {args.out.resolve()}")
    print(f"Open it with:  open {args.out.resolve()}")


if __name__ == "__main__":
    main()
