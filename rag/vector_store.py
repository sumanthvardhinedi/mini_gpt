"""
Phase 12: Local In-Memory & Persistent Vector Store using Cosine Similarity.

Stores records containing:
- `vector`: `List[float]`
- `chunk` / `text`: `str`
- `source`: `str`
- `chunk_id`: `str`
- `metadata`: `Dict[str, Any]`

Supports:
- `add_documents(chunks, vectors=None)`
- `search(query_vector, top_k=5)`
- `save(path)`
- `load(path)`
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from rag.embeddings import EmbeddingModel


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Compute cosine similarity between two numeric vectors `vec_a` and `vec_b`:
        cos(a, b) = (a · b) / (||a|| * ||b||)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch: len(vec_a)={len(vec_a)} vs len(vec_b)={len(vec_b)}."
        )
    if len(vec_a) == 0:
        raise ValueError("Cannot compute cosine similarity of empty vectors.")

    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


class LocalVectorStore:
    """
    Simple local vector store using exact cosine similarity search and JSON persistence.
    """

    def __init__(self, embedder: Optional[EmbeddingModel] = None) -> None:
        self.embedder = embedder
        self.records: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.records)

    def add_documents(
        self,
        documents: Sequence[Dict[str, Any]],
        vectors: Optional[Sequence[Sequence[float]]] = None,
    ) -> int:
        """
        Insert chunk documents and their embedding vectors into the vector store.

        Args:
            documents: Sequence of chunk dicts (`{"text" or "chunk", "source", "metadata"}`).
            vectors: Optional precomputed vectors corresponding to `documents`. If omitted,
                     uses `doc["vector"]` if present or `self.embedder.embed_documents(...)`.

        Returns:
            Number of documents added.
        """
        doc_list = list(documents)
        if not doc_list:
            return 0

        if vectors is not None:
            vec_list = [list(map(float, v)) for v in vectors]
            if len(vec_list) != len(doc_list):
                raise ValueError(
                    f"Length mismatch: {len(doc_list)} documents vs {len(vec_list)} vectors."
                )
        elif all("vector" in d for d in doc_list):
            vec_list = [list(map(float, d["vector"])) for d in doc_list]
        elif self.embedder is not None:
            texts = [str(d.get("text", d.get("chunk", ""))) for d in doc_list]
            vec_list = self.embedder.embed_documents(texts)
        else:
            raise ValueError(
                "add_documents requires explicit `vectors`, `'vector'` keys on documents, "
                "or a configured `embedder` on LocalVectorStore."
            )

        for idx, (doc, vec) in enumerate(zip(doc_list, vec_list)):
            chunk_text = str(doc.get("chunk", doc.get("text", "")))
            source = str(doc.get("source", "unknown"))
            chunk_id = str(doc.get("chunk_id", f"{source}::chunk_{len(self.records) + idx}"))
            metadata = dict(doc.get("metadata", {}))

            self.records.append(
                {
                    "chunk_id": chunk_id,
                    "chunk": chunk_text,
                    "text": chunk_text,
                    "source": source,
                    "metadata": metadata,
                    "vector": list(vec),
                }
            )
        return len(doc_list)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store for the `top_k` highest cosine-similarity chunks.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")
        if not self.records:
            return []

        scored: List[Dict[str, Any]] = []
        for rec in self.records:
            score = cosine_similarity(query_vector, rec["vector"])
            scored.append(
                {
                    "chunk_id": rec["chunk_id"],
                    "chunk": rec["chunk"],
                    "text": rec["text"],
                    "source": rec["source"],
                    "score": float(score),
                    "metadata": dict(rec["metadata"]),
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["chunk_id"]))
        return scored[:top_k]

    def save(self, path: Union[str, Path]) -> Path:
        """Save all stored vectors, chunks, sources, and metadata to a JSON index file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": self.records}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def load(self, path: Union[str, Path]) -> "LocalVectorStore":
        """Load vector index records from a JSON file into this `LocalVectorStore`."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Vector index file not found: {target}")
        raw = json.loads(target.read_text(encoding="utf-8"))
        self.records = list(raw.get("records", []))
        return self
