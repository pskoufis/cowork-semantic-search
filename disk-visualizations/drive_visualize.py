#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly>=5.20,<7.0",
# ]
# ///
"""
drive_visualize.py — Interactive HTML dashboard for a directory tree.

Usage:
    # Walk fresh, write dashboard.html in cwd
    uv run drive_visualize.py /Volumes/YourDrive

    # Reuse an existing analysis.csv produced by drive_analyze.py
    uv run drive_visualize.py /Volumes/YourDrive --csv analysis.csv

    # Custom output path
    uv run drive_visualize.py /Volumes/YourDrive --out /tmp/drive.html

Views in the dashboard:
    1. Folder treemap (size by total bytes, click to drill in)
    2. File-type family donut + top-20 extensions bars
    3. Size-bucket distribution (count and bytes side-by-side)
    4. Age histogram (files by last-modified year, count and bytes)
    5. Top-100 largest files (sortable table)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Shared constants (kept in sync with drive_analyze.py) ----------
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

# Stable color palette per family (used in donut + treemap coloring)
FAMILY_COLORS = {
    "image":    "#4C9AFF",
    "video":    "#FF5630",
    "audio":    "#FFAB00",
    "document": "#36B37E",
    "sheet":    "#00B8D9",
    "slides":   "#6554C0",
    "archive":  "#8777D9",
    "code":     "#998DD9",
    "email":    "#FF8B00",
    "design":   "#FF7452",
    "data":     "#00C7E6",
    "other":    "#A5ADBA",
}


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


# ---------- Data loading ----------

def walk_filesystem(root: Path, follow_symlinks: bool = False):
    """Yield (Path, size, mtime) for every regular file under root."""
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
                yield p, st.st_size, st.st_mtime
            except (OSError, PermissionError):
                continue


def load_csv(csv_path: Path):
    """Yield (Path, size, mtime) from a drive_analyze.py-produced CSV."""
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                yield Path(row["path"]), int(row["size"]), float(row["mtime"] or 0)
            except (KeyError, ValueError):
                continue


def collect_records(root: Path, csv_path: Path | None, progress_every: int = 50_000):
    records: list[tuple[Path, int, float]] = []
    source = "csv" if csv_path else "walk"
    if csv_path:
        print(f"Loading records from {csv_path} ...")
        for i, rec in enumerate(load_csv(csv_path), 1):
            records.append(rec)
            if i % progress_every == 0:
                print(f"  loaded {i:,} rows...")
    else:
        print(f"Walking {root} ...")
        for i, rec in enumerate(walk_filesystem(root), 1):
            records.append(rec)
            if i % progress_every == 0:
                print(f"  scanned {i:,} files...")
    print(f"Done. {len(records):,} files. ({source})")
    return records


# ---------- Views ----------

def build_treemap(records, root: Path, top_n_dirs: int = 500) -> go.Figure:
    """Folder treemap. Each rectangle = a directory, sized by cumulative bytes.

    Capping to top_n_dirs keeps the HTML small for huge trees; the user can
    still drill in because parent dirs aggregate their dropped children's bytes.
    """
    root_str = str(root)
    dir_bytes: Counter[str] = Counter()
    dir_family_bytes: dict[str, Counter] = defaultdict(Counter)

    for path, size, _ in records:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        ext = path.suffix.lower()
        family = EXT_TO_FAMILY.get(ext, "other")

        parts = rel.parts[:-1]  # exclude filename
        # Every ancestor directory (including root itself)
        for i in range(len(parts) + 1):
            ancestor = root_str if i == 0 else os.path.join(root_str, *parts[:i])
            dir_bytes[ancestor] += size
            dir_family_bytes[ancestor][family] += size

    # Keep root + top-N dirs by size
    sorted_dirs = sorted(dir_bytes.items(), key=lambda x: -x[1])
    kept = {root_str}
    for d, _ in sorted_dirs[:top_n_dirs]:
        kept.add(d)

    ids, labels, parents, values, colors, hover = [], [], [], [], [], []
    for d, b in dir_bytes.items():
        if d not in kept:
            continue
        ids.append(d)
        labels.append(Path(d).name if d != root_str else (root.name or root_str))
        parent = "" if d == root_str else str(Path(d).parent)
        # If parent isn't in kept set, attach to root instead (avoids orphans)
        parents.append("" if d == root_str else (parent if parent in kept else root_str))
        values.append(b)

        # Color by dominant family in this dir
        if dir_family_bytes[d]:
            top_fam = max(dir_family_bytes[d].items(), key=lambda x: x[1])[0]
        else:
            top_fam = "other"
        colors.append(FAMILY_COLORS.get(top_fam, FAMILY_COLORS["other"]))

        # Hover: size + family breakdown
        fam_lines = []
        for fam, fb in sorted(dir_family_bytes[d].items(), key=lambda x: -x[1])[:5]:
            fam_lines.append(f"{fam}: {human(fb)}")
        hover.append(f"<b>{labels[-1]}</b><br>{human(b)}<br>" + "<br>".join(fam_lines))

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(width=0.5, color="white")),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        maxdepth=3,
    ))
    fig.update_layout(
        title=f"Folder treemap — colored by dominant file family (top {top_n_dirs} dirs shown)",
        margin=dict(t=50, l=10, r=10, b=10),
        height=650,
    )
    return fig


def build_type_views(records) -> go.Figure:
    """Donut (families by bytes) + bar (top-20 extensions by bytes), side by side."""
    family_bytes: Counter[str] = Counter()
    ext_bytes: Counter[str] = Counter()
    ext_count: Counter[str] = Counter()

    for path, size, _ in records:
        ext = path.suffix.lower() or "(no ext)"
        family = EXT_TO_FAMILY.get(ext, "other")
        family_bytes[family] += size
        ext_bytes[ext] += size
        ext_count[ext] += 1

    fam_sorted = sorted(family_bytes.items(), key=lambda x: -x[1])
    fam_labels = [f for f, _ in fam_sorted]
    fam_values = [b for _, b in fam_sorted]
    fam_colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"]) for f in fam_labels]

    top_ext = sorted(ext_bytes.items(), key=lambda x: -x[1])[:20]
    ext_labels = [e for e, _ in top_ext]
    ext_values = [b for _, b in top_ext]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=("File-type families (by bytes)", "Top 20 extensions (by bytes)"),
        column_widths=[0.4, 0.6],
    )
    fig.add_trace(
        go.Pie(
            labels=fam_labels,
            values=fam_values,
            hole=0.5,
            marker=dict(colors=fam_colors),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} bytes (%{percent})<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ext_values,
            y=ext_labels,
            orientation="h",
            marker_color="#4C9AFF",
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} bytes<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Bytes", row=1, col=2)
    fig.update_layout(height=450, showlegend=True, margin=dict(t=60, l=10, r=10, b=40))
    return fig


def build_size_buckets(records) -> go.Figure:
    """Side-by-side: file-count per bucket and total-bytes per bucket."""
    buckets_count: Counter[str] = Counter()
    buckets_bytes: Counter[str] = Counter()
    for _, size, _ in records:
        b = bucket_for(size)
        buckets_count[b] += 1
        buckets_bytes[b] += size

    labels = [b[0] for b in SIZE_BUCKETS]
    counts = [buckets_count.get(l, 0) for l in labels]
    bytes_ = [buckets_bytes.get(l, 0) for l in labels]
    bytes_human = [human(b) for b in bytes_]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("By file count", "By total bytes"))
    fig.add_trace(
        go.Bar(x=labels, y=counts, marker_color="#4C9AFF",
               hovertemplate="<b>%{x}</b><br>%{y:,} files<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=bytes_, marker_color="#FF8B00", text=bytes_human, textposition="outside",
               hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"),
        row=1, col=2,
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(title="Size distribution", showlegend=False, height=420,
                      margin=dict(t=60, l=10, r=10, b=80))
    return fig


def build_age_views(records) -> go.Figure:
    """Files by last-modified year — count and bytes side by side."""
    year_count: Counter[int] = Counter()
    year_bytes: Counter[int] = Counter()
    for _, size, mtime in records:
        if not mtime:
            continue
        try:
            y = datetime.fromtimestamp(mtime).year
        except (OSError, ValueError, OverflowError):
            continue
        # Clamp unreasonable years (filesystem corruption can produce 1970 or 2099)
        if y < 1980 or y > datetime.now().year + 1:
            continue
        year_count[y] += 1
        year_bytes[y] += size

    if not year_count:
        # Empty plot fallback
        fig = go.Figure()
        fig.update_layout(title="Age distribution — no valid mtimes found", height=350)
        return fig

    years = sorted(year_count.keys())
    counts = [year_count[y] for y in years]
    bytes_ = [year_bytes[y] for y in years]
    bytes_human = [human(b) for b in bytes_]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("By file count", "By total bytes"))
    fig.add_trace(
        go.Bar(x=years, y=counts, marker_color="#36B37E",
               hovertemplate="<b>%{x}</b><br>%{y:,} files<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=years, y=bytes_, marker_color="#6554C0", text=bytes_human, textposition="outside",
               hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Year")
    fig.update_layout(title="Age distribution (last modified)", showlegend=False, height=420,
                      margin=dict(t=60, l=10, r=10, b=40))
    return fig


def build_largest_table(records, n: int = 100) -> go.Figure:
    sorted_recs = sorted(records, key=lambda x: -x[1])[:n]
    if not sorted_recs:
        fig = go.Figure()
        fig.update_layout(title="No files found", height=200)
        return fig

    rank = list(range(1, len(sorted_recs) + 1))
    sizes = [human(s) for _, s, _ in sorted_recs]
    size_raw = [s for _, s, _ in sorted_recs]
    mtimes = [datetime.fromtimestamp(m).strftime("%Y-%m-%d") if m else "?"
              for _, _, m in sorted_recs]
    exts = [p.suffix.lower() or "(none)" for p, _, _ in sorted_recs]
    paths = [str(p) for p, _, _ in sorted_recs]

    fig = go.Figure(go.Table(
        header=dict(
            values=["#", "Size", "Bytes", "Modified", "Ext", "Path"],
            fill_color="#22272E",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[rank, sizes, size_raw, mtimes, exts, paths],
            align=["right", "right", "right", "center", "center", "left"],
            height=22,
            font=dict(size=11),
        ),
        columnwidth=[30, 70, 80, 80, 50, 600],
    ))
    fig.update_layout(
        title=f"Top {len(sorted_recs)} largest files (click column headers to sort)",
        height=700,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return fig


# ---------- HTML assembly ----------

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Drive visualization — {root}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 24px; background: #F4F5F7; color: #172B4D;
    max-width: 1400px; margin-left: auto; margin-right: auto;
  }}
  h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
  .meta {{ color: #5E6C84; font-size: 13px; margin-bottom: 24px; }}
  .card {{
    background: white; border-radius: 8px; padding: 16px;
    margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .summary {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .summary div {{ flex: 1; min-width: 180px; }}
  .summary .num {{ font-size: 28px; font-weight: 600; color: #0052CC; }}
  .summary .lbl {{ font-size: 12px; color: #5E6C84; text-transform: uppercase; letter-spacing: 0.5px; }}
  footer {{ color: #97A0AF; font-size: 11px; text-align: center; margin-top: 24px; }}
</style>
</head>
<body>
<h1>Drive visualization — <code>{root}</code></h1>
<div class="meta">Generated {ts} · {n_files:,} files · {total_size}</div>
<div class="card summary">
  <div><div class="num">{n_files:,}</div><div class="lbl">Total files</div></div>
  <div><div class="num">{total_size}</div><div class="lbl">Total size</div></div>
  <div><div class="num">{n_dirs:,}</div><div class="lbl">Directories</div></div>
  <div><div class="num">{n_families}</div><div class="lbl">File-type families</div></div>
</div>
"""

HTML_FOOTER = """
<footer>Built with Plotly · open in any modern browser · no server required.</footer>
</body>
</html>
"""


def render_html(figs: list[tuple[str, go.Figure]], out_path: Path, *,
                root: Path, n_files: int, total_size: int, n_dirs: int,
                n_families: int, include_plotlyjs: str = "cdn"):
    parts = [HTML_HEADER.format(
        root=str(root),
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_files=n_files,
        total_size=human(total_size),
        n_dirs=n_dirs,
        n_families=n_families,
    )]
    for i, (title, fig) in enumerate(figs):
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
    ap.add_argument("root", type=Path, help="Directory to visualize")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Reuse an analysis.csv from drive_analyze.py instead of walking")
    ap.add_argument("--out", type=Path, default=Path("dashboard.html"),
                    help="Output HTML path (default: dashboard.html)")
    ap.add_argument("--top-dirs", type=int, default=500,
                    help="Max number of directories to show in the treemap (default: 500)")
    ap.add_argument("--top-files", type=int, default=100,
                    help="How many files to list in the largest-files table (default: 100)")
    ap.add_argument("--inline-plotly", action="store_true",
                    help="Inline plotly.js into the HTML (no internet needed, ~3MB larger file)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"Path does not exist: {args.root}", file=sys.stderr)
        sys.exit(1)
    if args.csv and not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    records = collect_records(args.root, args.csv)
    if not records:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    n_files = len(records)
    total_size = sum(s for _, s, _ in records)
    # Count unique directories under root
    dir_set = set()
    for p, _, _ in records:
        try:
            rel = p.relative_to(args.root)
            for i in range(len(rel.parts)):
                dir_set.add(os.path.join(str(args.root), *rel.parts[:i]))
        except ValueError:
            continue
    n_families = len({EXT_TO_FAMILY.get(p.suffix.lower(), "other") for p, _, _ in records})

    print("Building views ...")
    figs = [
        ("Treemap", build_treemap(records, args.root, top_n_dirs=args.top_dirs)),
        ("File types", build_type_views(records)),
        ("Size buckets", build_size_buckets(records)),
        ("Age", build_age_views(records)),
        ("Largest files", build_largest_table(records, n=args.top_files)),
    ]

    render_html(
        figs, args.out,
        root=args.root,
        n_files=n_files,
        total_size=total_size,
        n_dirs=len(dir_set),
        n_families=n_families,
        include_plotlyjs=True if args.inline_plotly else "cdn",
    )
    print(f"Dashboard written to {args.out.resolve()}")
    print(f"Open it with:  open {args.out.resolve()}")


if __name__ == "__main__":
    main()
