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
    """
    chunks = []
    file_name = os.path.basename(source_rel)
    file_type = os.path.splitext(source_rel)[1].lower()
    folder_path = os.path.dirname(source_rel) or "."
    for part in extracted_parts:
        text = part["text"]
        if not text.strip():
            continue
        splits = splitter.split_text(text)
        page_or_section = part.get("metadata", {}).get("page_number", 0)
        for chunk_idx, chunk_text in enumerate(splits):
            chunks.append({
                "id": f"{_short_hash(source_rel)}_{page_or_section}_{chunk_idx}",
                "text": chunk_text,
                "source_file": source_rel,
                "file_name": file_name,
                "file_type": file_type,
                "folder_path": folder_path,
                "chunk_index": chunk_idx,
                **part.get("metadata", {}),
            })
    return chunks
