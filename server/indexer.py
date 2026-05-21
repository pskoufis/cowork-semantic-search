"""Document indexing pipeline: discover, parse, chunk, embed, store."""

import hashlib
import os
import time
from pathlib import Path
from typing import Callable

from server.parsers import extract_text, SUPPORTED_EXTENSIONS
from server.chunker import chunk_document
from server.store import VectorStore
from server.paths import to_relative, to_absolute

EXCLUDE_PATTERNS = {"__pycache__", ".git", ".DS_Store", "node_modules", ".venv", "*.tmp"}
BATCH_SIZE = 64
# Chunks are buffered across files and written in one table.add() once the
# buffer reaches this size, instead of a tiny write per file.
FLUSH_CHUNK_THRESHOLD = 1000

# Per-file size cap. Files larger than this are skipped instead of indexed, so
# one multi-GB file cannot OOM the server — every parser reads the whole file
# into memory (a CSV transiently holds ~3x its size). Override via the
# MAX_FILE_SIZE_MB env var; set it to 0 to disable the cap entirely.
try:
    _MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "100"))
except ValueError:
    _MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = _MAX_FILE_SIZE_MB * 1024 * 1024 if _MAX_FILE_SIZE_MB > 0 else 0

# Formats whose parser streams the file from disk instead of reading it whole,
# so the size cap above must not apply. pypff reads a .pst incrementally; real
# Outlook archives are routinely multi-GB and the cap would skip them all.
# (OOM protection for .pst moves to a per-attachment guard inside the parser.)
STREAMING_EXTENSIONS = {".pst"}


def exceeds_size_cap(file_path: Path, size: int) -> bool:
    """True when a file is too large to parse safely.

    Always False when the cap is disabled or the format streams from disk."""
    if not MAX_FILE_SIZE_BYTES:
        return False
    if file_path.suffix.lower() in STREAMING_EXTENSIONS:
        return False
    return size > MAX_FILE_SIZE_BYTES


_model = None


def _select_device() -> str:
    """Pick the best embedding device available: Apple-Silicon MPS, then CUDA,
    else CPU. Defensive about torch builds that lack the mps backend."""
    try:
        import torch
    except Exception:
        return "cpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device=_select_device(),
        )
    return _model


def discover_files(
    folder_path: Path,
    file_types: set[str] | None,
    recursive: bool,
) -> list[Path]:
    extensions = file_types or SUPPORTED_EXTENSIONS
    pattern = "**/*" if recursive else "*"
    files = []
    for path in folder_path.glob(pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            if not any(exc in path.parts for exc in EXCLUDE_PATTERNS):
                files.append(path)
    return sorted(files)


def compute_file_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    model = get_model()
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(embeddings)
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["vector"] = embedding.tolist()
    return chunks


def _finalize_index(
    store: VectorStore, *, content_changed: bool, stats_changed: bool
) -> list[str]:
    """Post-indexing finalize step.

    Compacts the table when anything was written — index content *or* just file
    stats (a drive move refreshes stats without changing content). Rebuilds the
    ANN and FTS indexes only when index content actually changed, so a pure
    drive-move run does not trigger a needless 50GB index rebuild.

    Each step's error is captured as a warning, never raised — a finalize
    hiccup must not fail a multi-hour indexing run.
    """
    warnings: list[str] = []
    steps: list[tuple[str, Callable[[], None]]] = []
    if content_changed or stats_changed:
        steps.append(("compaction", store.optimize_table))
    if content_changed:
        steps.append(("vector index build", store.create_vector_index))
        steps.append(("FTS index build", store.create_fts_index))
    for label, fn in steps:
        try:
            fn()
        except Exception as e:
            warnings.append(f"{label} failed: {e}")
    return warnings


def index_folder(
    folder_path: str,
    file_types: list[str] | None = None,
    recursive: bool = True,
    db_path: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    store = VectorStore(db_dir)
    store.ensure_schema()  # migrate a pre-Tier-2 index in place, if needed
    type_set = set(file_types) if file_types else None
    files = discover_files(folder, type_set, recursive)
    folder_resolved = folder.resolve()
    # One bulk load of {source_rel: {content_hash, mtime_ns, file_size}},
    # instead of a per-file change-detection query.
    file_index = store.get_file_index()

    start = time.time()
    indexed, skipped, deleted, failed, stats_refreshed = 0, 0, 0, 0, 0
    size_skipped = 0
    errors = []
    oversized_files = []  # files skipped for exceeding MAX_FILE_SIZE_BYTES
    current_files = set()  # paths relative to the index directory
    buffer: list[dict] = []  # chunks awaiting a batched write

    for idx, file_path in enumerate(files):
        if progress_callback is not None:
            progress_callback(idx, len(files))
        try:
            source_rel = to_relative(str(file_path), db_dir)
        except ValueError as e:  # file on a different volume than the index
            failed += 1
            errors.append({"file": str(file_path), "error": str(e)})
            continue
        current_files.add(source_rel)

        try:
            st = file_path.stat()
            mtime_ns, size = st.st_mtime_ns, st.st_size

            # Size cap: skip a file too large to parse without risking an OOM.
            # delete_by_file clears stale chunks if it was indexed while smaller;
            # the file stays in current_files so orphan-cleanup does not also
            # count it as deleted. Streaming formats (.pst) are exempt.
            if exceeds_size_cap(file_path, size):
                store.delete_by_file(source_rel)
                size_skipped += 1
                oversized_files.append({
                    "file": str(file_path),
                    "size_mb": round(size / 1024 / 1024, 1),
                })
                continue

            record = file_index.get(source_rel)

            # Fast path: stat unchanged -> file unchanged. No read, no hash.
            if (record
                    and record["mtime_ns"] == mtime_ns
                    and record["file_size"] == size):
                skipped += 1
                continue

            file_hash = compute_file_hash(file_path)

            # Content unchanged (file touched / drive moved): refresh the stored
            # stat so the fast path works next run, and skip re-embedding.
            if record and record["content_hash"] == file_hash:
                store.update_file_stat(source_rel, mtime_ns, size)
                stats_refreshed += 1
                skipped += 1
                continue

            # New file or changed content: (re)index it.
            store.delete_by_file(source_rel)
            parts = extract_text(file_path)
            chunks = chunk_document(parts, source_rel)
            if chunks:
                chunks = embed_chunks(chunks)
                for c in chunks:
                    c["content_hash"] = file_hash
                    c["mtime_ns"] = mtime_ns
                    c["file_size"] = size
                buffer.extend(chunks)
            indexed += 1
        except Exception as e:
            failed += 1
            errors.append({"file": str(file_path), "error": str(e)})

        # Flush the buffer once it is large enough. A failed batched write is a
        # systemic error and is left to propagate (fails the job).
        if len(buffer) >= FLUSH_CHUNK_THRESHOLD:
            store.add_chunks(buffer)
            buffer = []

    # Flush remaining chunks before orphan-cleanup and the final counts so they
    # see every written chunk.
    if buffer:
        store.add_chunks(buffer)
        buffer = []

    if progress_callback is not None:
        progress_callback(len(files), len(files))

    # Clean up chunks for files deleted from within this folder only.
    # Stored paths are relative — resolve to absolute before the scope check.
    for f_rel in store.get_all_files():
        f_abs = Path(to_absolute(f_rel, db_dir))
        in_scope = (
            f_abs.is_relative_to(folder_resolved) if recursive
            else f_abs.parent == folder_resolved
        )
        if in_scope and f_rel not in current_files:
            store.delete_by_file(f_rel)
            deleted += 1

    # Finalize: compact, and rebuild indexes when index content changed.
    finalize_warnings = _finalize_index(
        store,
        content_changed=(indexed > 0 or deleted > 0),
        stats_changed=(stats_refreshed > 0),
    )

    return {
        "status": "completed",
        "folder_path": folder_path,
        "files_indexed": indexed,
        "files_skipped": skipped,
        "files_deleted": deleted,
        "files_failed": failed,
        "files_size_skipped": size_skipped,
        "total_chunks": store.count_chunks(),
        "errors": errors,
        "oversized_files": oversized_files,
        "finalize_warnings": finalize_warnings,
        "duration_seconds": round(time.time() - start, 2),
    }
