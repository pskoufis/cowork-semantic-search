"""Text chunking with metadata preservation."""

import hashlib
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def chunk_document(extracted_parts: list[dict], source_rel: str) -> list[dict]:
    """Split extracted text into chunks, preserving per-part metadata.

    source_rel is the file path relative to the LanceDB directory; the chunk id
    and all path fields derive from it so they stay stable when the index is
    moved to a new location.

    chunk_index — and the id suffix — is a single counter that runs across
    every part of the file. A per-part counter restarts at 0 for each part, so
    two parts (PDF pages, PPTX slides, or one part per message in a PST
    archive) would produce colliding ids and colliding (source_file,
    chunk_index) pairs — the latter being the key hybrid search dedupes on.
    """
    chunks = []
    file_name = os.path.basename(source_rel)
    file_type = os.path.splitext(source_rel)[1].lower()
    folder_path = os.path.dirname(source_rel) or "."
    file_hash = _short_hash(source_rel)
    chunk_index = 0
    for part in extracted_parts:
        text = part["text"]
        if not text.strip():
            continue
        for chunk_text in splitter.split_text(text):
            chunks.append({
                "id": f"{file_hash}_{chunk_index}",
                "text": chunk_text,
                "source_file": source_rel,
                "file_name": file_name,
                "file_type": file_type,
                "folder_path": folder_path,
                "chunk_index": chunk_index,
                **part.get("metadata", {}),
            })
            chunk_index += 1
    return chunks
