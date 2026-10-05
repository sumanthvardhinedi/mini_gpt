"""
Phase 12: Deterministic Text Chunking with Configurable Size and Overlap.

Splits documents into overlapping chunks (`chunk_size=500`, `chunk_overlap=100` by default)
while preserving:
- `chunk_id`
- `source`
- `metadata` (including original document metadata, `chunk_index`, `start_char`, `end_char`)
"""

from typing import Any, Dict, List


class TextChunker:
    """
    Deterministic sliding-window text chunker.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
        if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be a non-negative integer, got {chunk_overlap!r}."
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step = chunk_size - chunk_overlap

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a single document dictionary (`{"text", "source", "metadata"}`) into
        overlapping chunk dictionaries.
        """
        text = str(document.get("text", ""))
        source = str(document.get("source", "unknown"))
        base_metadata = dict(document.get("metadata", {}))

        if len(text) == 0:
            return []

        chunks: List[Dict[str, Any]] = []
        text_len = len(text)
        start = 0
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            slice_text = text[start:end]
            chunk_id = f"{source}::chunk_{chunk_idx}"

            chunk_meta = dict(base_metadata)
            chunk_meta.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "start_char": start,
                    "end_char": end,
                    "source": source,
                }
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": slice_text,
                    "chunk": slice_text,
                    "source": source,
                    "metadata": chunk_meta,
                }
            )

            chunk_idx += 1
            if end >= text_len:
                break
            start += self.step

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a list of document dictionaries in deterministic order."""
        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


def chunk_text(
    text: str,
    source: str = "inline",
    metadata: Dict[str, Any] | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Convenience helper to chunk a raw text string with source & metadata."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document(
        {
            "text": text,
            "source": source,
            "metadata": metadata or {},
        }
    )


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Convenience helper to chunk a list of loaded documents."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(documents)
