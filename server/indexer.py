"""Document indexing pipeline: discover, parse, chunk, embed, store."""

import hashlib
import os
import time
from pathlib import Path

from server.parsers import extract_text, SUPPORTED_EXTENSIONS
from server.chunker import chunk_document
from server.store import VectorStore
from server.paths import to_relative, to_absolute

EXCLUDE_PATTERNS = {"__pycache__", ".git", ".DS_Store", "node_modules", ".venv", "*.tmp"}
BATCH_SIZE = 64

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
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


def index_folder(
    folder_path: str,
    file_types: list[str] | None = None,
    recursive: bool = True,
    db_path: str | None = None,
) -> dict:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if db_path is None:
        db_path = os.environ.get("LANCEDB_PATH", "./lancedb")
    db_dir = os.path.abspath(db_path)

    store = VectorStore(db_dir)
    type_set = set(file_types) if file_types else None
    files = discover_files(folder, type_set, recursive)
    folder_resolved = folder.resolve()

    start = time.time()
    indexed, skipped, deleted, failed = 0, 0, 0, 0
    errors = []
    current_files = set()  # paths relative to the index directory

    for file_path in files:
        try:
            source_rel = to_relative(str(file_path), db_dir)
        except ValueError as e:  # file on a different volume than the index
            failed += 1
            errors.append({"file": str(file_path), "error": str(e)})
            continue
        current_files.add(source_rel)
        file_hash = compute_file_hash(file_path)

        existing_hash = store.get_file_hash(source_rel)
        if existing_hash == file_hash:
            skipped += 1
            continue

        try:
            store.delete_by_file(source_rel)
            parts = extract_text(file_path)
            chunks = chunk_document(parts, source_rel)
            if chunks:
                chunks = embed_chunks(chunks)
                for c in chunks:
                    c["content_hash"] = file_hash
                store.add_chunks(chunks)
            indexed += 1
        except Exception as e:
            failed += 1
            errors.append({"file": str(file_path), "error": str(e)})

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

    return {
        "status": "completed",
        "folder_path": folder_path,
        "files_indexed": indexed,
        "files_skipped": skipped,
        "files_deleted": deleted,
        "files_failed": failed,
        "total_chunks": store.count_chunks(),
        "errors": errors,
        "duration_seconds": round(time.time() - start, 2),
    }
