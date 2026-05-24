"""Document indexing pipeline: discover, parse, chunk, embed, store."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

from server.parsers import extract_text, SUPPORTED_EXTENSIONS
from server.chunker import chunk_document
from server.exclusions import ExclusionRules
from server.spreadsheets import (
    SPREADSHEET_EXTENSIONS,
    UnreadableSpreadsheetError,
    extract_preview,
    needs_for_preview,
)
from server.store import VectorStore
from server.paths import to_relative, to_absolute

BATCH_SIZE = 64
# Chunks are buffered across files and written in one table.add() once the
# buffer reaches this size, instead of a tiny write per file.
FLUSH_CHUNK_THRESHOLD = 1000

# Per-file size cap. Files larger than this are skipped instead of indexed, so
# one multi-GB file cannot OOM the server — every parser reads the whole file
# into memory (a CSV transiently holds ~3x its size). Override via the
# MAX_FILE_SIZE_MB env var; set it to 0 to disable the cap entirely.
try:
    _MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "1024"))
except ValueError:
    _MAX_FILE_SIZE_MB = 1024
MAX_FILE_SIZE_BYTES = _MAX_FILE_SIZE_MB * 1024 * 1024 if _MAX_FILE_SIZE_MB > 0 else 0

# Formats whose parser retains only a bounded preview, so the size cap above
# must not apply. pypff reads a .pst incrementally; real Outlook archives are
# routinely multi-GB. csv and xlsx/xlsm preview via streaming; xls loads the
# whole BIFF document but is bounded by BIFF's 65,536-row-per-sheet cap. Only
# headers + ~10 sample rows ever land in the preview regardless of file size.
STREAMING_EXTENSIONS = {".pst", ".csv", ".xlsx", ".xlsm", ".xls"}


def exceeds_size_cap(file_path: Path, size: int) -> bool:
    """True when a file is too large to parse safely.

    Always False when the cap is disabled or the format streams from disk."""
    if not MAX_FILE_SIZE_BYTES:
        return False
    if file_path.suffix.lower() in STREAMING_EXTENSIONS:
        return False
    return size > MAX_FILE_SIZE_BYTES


# Qwen3-Embedding-0.6B: 1024-dim native, MRL-supported. We truncate to
# EMBEDDING_DIM (from store.py — currently 256) via Matryoshka — the head
# dimensions are first-class, not a naive slice.
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

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
        from server.store import EMBEDDING_DIM
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=_select_device(),
            # Matryoshka — sentence-transformers truncates *and* L2-renormalises.
            truncate_dim=EMBEDDING_DIM,
            # Qwen3 uses last-token pooling on a decoder; right-padding would
            # make the pooled token a PAD for shorter sequences in a batch and
            # silently corrupt embeddings.
            tokenizer_kwargs={"padding_side": "left"},
        )
        # Fail loudly at load if the model upload ever drops the "query" prompt
        # we rely on in search.py — better than dying on first user query.
        if "query" not in getattr(_model, "prompts", {}):
            raise RuntimeError(
                f"{EMBEDDING_MODEL} did not expose a 'query' prompt; "
                "server/search.py relies on it. Check the model card / upload."
            )
    return _model


def discover_files(
    folder_path: Path,
    file_types: set[str] | None,
    recursive: bool,
    exclusions: ExclusionRules | None = None,
) -> list[Path]:
    """Walk the folder and return files matching `file_types`.

    When `exclusions` is set, directories matching an exclusion rule are
    pruned from the walk (the walker never descends into them), and matched
    files are filtered out.

    Spreadsheet extensions are filtered out unconditionally, even when the
    caller passes them in `file_types`, so the temporary "spreadsheet
    indexing disabled" stance can't be sidestepped by an explicit override.
    Restore by removing the subtraction below and re-adding the extensions
    to `SUPPORTED_EXTENSIONS` in `server/parsers.py`.
    """
    extensions = set(file_types) if file_types else set(SUPPORTED_EXTENSIONS)
    extensions -= SPREADSHEET_EXTENSIONS
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        dpath = Path(dirpath)
        if recursive:
            if exclusions is not None:
                # In-place mutation so os.walk skips the pruned dirs entirely;
                # is_dir=True is required for directory-only patterns to match.
                dirnames[:] = [
                    d for d in dirnames
                    if not exclusions.is_excluded(dpath / d, folder_path, is_dir=True)
                ]
        else:
            dirnames[:] = []
        for name in filenames:
            f = dpath / name
            if f.suffix.lower() not in extensions:
                continue
            if exclusions is not None and exclusions.is_excluded(
                f, folder_path, is_dir=False
            ):
                continue
            files.append(f)
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
    exclude: list[str] | None = None,
) -> dict:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    # Compile exclusion rules early — a bad pattern raises ValueError before
    # any work is done. The MCP layer turns that into a `rejected` response.
    exclusions = ExclusionRules.load(folder, extra_patterns=exclude, db_dir=db_dir)

    store = VectorStore(db_dir)
    store.ensure_schema()  # migrate older indexes in place, if needed
    # One-shot eviction: drop any legacy CSV/XLSX text chunks so they're
    # re-processed via the description path on this run. The returned set
    # tells the per-file loop to bypass the content_hash short-circuit for
    # those files even when the hash hasn't changed.
    forced_reindex: set[str] = store.evict_legacy_spreadsheet_chunks()
    type_set = set(file_types) if file_types else None
    files = discover_files(folder, type_set, recursive, exclusions=exclusions)
    folder_resolved = folder.resolve()
    # One bulk load of {source_rel: {content_hash, mtime_ns, file_size}},
    # instead of a per-file change-detection query.
    file_index = store.get_file_index()

    # Prune pass: drop chunks for already-indexed files *under this folder*
    # that now match an exclusion rule. Scoped by in-scope check so an
    # exclusion in folder B's run cannot wipe folder A's chunks. Runs before
    # the main loop so the orphan-cleanup at the end does not also see these
    # files and double-count them as deletions.
    #
    # `f_abs` from `to_absolute` is not resolved (it just normpaths join), so
    # we resolve it here to match `folder_resolved` — otherwise macOS's
    # `/var` → `/private/var` symlink makes every in_scope check fail.
    files_excluded_pruned = 0
    for f_rel in list(file_index.keys()):
        f_abs_raw = Path(to_absolute(f_rel, db_dir))
        try:
            f_abs = f_abs_raw.resolve()
        except OSError:
            f_abs = f_abs_raw
        in_scope = (
            f_abs.is_relative_to(folder_resolved) if recursive
            else f_abs.parent == folder_resolved
        )
        if not in_scope:
            continue
        if exclusions.is_excluded(f_abs, folder_resolved, is_dir=False):
            store.delete_by_file(f_rel)
            file_index.pop(f_rel, None)
            files_excluded_pruned += 1

    start = time.time()
    indexed, skipped, deleted, failed, stats_refreshed = 0, 0, 0, 0, 0
    size_skipped = 0
    descriptions_queued = 0
    descriptions_sampled = 0  # filled in by the async auto-drainer post-call
    errors = []
    oversized_files = []  # files skipped for exceeding MAX_FILE_SIZE_BYTES
    current_files = set()  # storage form returned by to_relative (relative same-volume, absolute cross-volume)
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
            suffix = file_path.suffix.lower()
            force = source_rel in forced_reindex

            # Spreadsheet path: never index raw cells — extract a preview and
            # enqueue an LLM-description request. Both auto-drain (later, via
            # ctx.sample()) and the manual MCP tools turn queue entries into
            # description chunks. Fast/hash short-circuits still apply, except
            # on the one-shot legacy-eviction pass which bypasses them.
            if suffix in SPREADSHEET_EXTENSIONS:
                if (not force and record
                        and record["mtime_ns"] == mtime_ns
                        and record["file_size"] == size):
                    skipped += 1
                    continue

                file_hash = compute_file_hash(file_path)

                if (not force and record
                        and record["content_hash"] == file_hash):
                    store.update_file_stat(source_rel, mtime_ns, size)
                    stats_refreshed += 1
                    skipped += 1
                    continue

                if store.is_dismissed(source_rel, file_hash):
                    skipped += 1
                    continue

                # New / changed / forced: drop old chunks first so the file
                # is never half-old half-new.
                store.delete_by_file(source_rel)
                try:
                    preview = extract_preview(file_path)
                except UnreadableSpreadsheetError as exc:
                    failed += 1
                    errors.append({
                        "file": str(file_path),
                        "error": f"unreadable spreadsheet: {exc}",
                    })
                    continue

                store.enqueue_pending(
                    file_path=source_rel,
                    needs=needs_for_preview(preview),
                    preview_json=json.dumps(preview),
                    content_hash=file_hash,
                )
                descriptions_queued += 1
                indexed += 1  # counts as work done for the finalize trigger
                forced_reindex.discard(source_rel)
                continue

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
    # Stored paths may be relative (same-volume) or absolute (cross-volume);
    # to_absolute handles both before the scope check.
    # Skip excluded paths so the prune-pass count and this orphan count never
    # claim the same file: an excluded file is already gone from the store by
    # the time we reach here, but the guard makes the intent explicit.
    # Also skip files whose type isn't currently indexable — without this
    # guard, disabling a file type (e.g. spreadsheets) would silently delete
    # every existing chunk of that type on the next run.
    for f_rel in store.get_all_files():
        if Path(f_rel).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        f_abs_raw = Path(to_absolute(f_rel, db_dir))
        try:
            f_abs = f_abs_raw.resolve()
        except OSError:
            f_abs = f_abs_raw
        in_scope = (
            f_abs.is_relative_to(folder_resolved) if recursive
            else f_abs.parent == folder_resolved
        )
        if not in_scope:
            continue
        if exclusions.is_excluded(f_abs, folder_resolved, is_dir=False):
            continue
        if f_rel not in current_files:
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
        "files_excluded_pruned": files_excluded_pruned,
        "total_chunks": store.count_chunks(),
        "errors": errors,
        "oversized_files": oversized_files,
        "exclusion_patterns": exclusions.patterns,
        "descriptions_queued": descriptions_queued,
        "descriptions_sampled": descriptions_sampled,
        "finalize_warnings": finalize_warnings,
        "duration_seconds": round(time.time() - start, 2),
    }
