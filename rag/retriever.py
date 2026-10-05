"""
Phase 12: Retriever for MiniGPT RAG.

Converts a natural-language `query` into an embedding vector via `EmbeddingModel`
and retrieves the top-k ranked chunks from `LocalVectorStore`.
"""

from typing import Any, Dict, List, Optional

from rag.embeddings import EmbeddingModel, LocalEmbeddingModel
from rag.vector_store import LocalVectorStore


class Retriever:
    """
    Semantic chunk retriever backed by `LocalVectorStore` and `EmbeddingModel`.
    """

    def __init__(
        self,
        vector_store: LocalVectorStore,
        embedder: Optional[EmbeddingModel] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = (
            embedder
            or vector_store.embedder
            or LocalEmbeddingModel()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for `query` ranked by cosine similarity.

        Each returned item contains:
        - `chunk`: str
        - `source`: str
        - `score`: float
        - `metadata`: dict
        """
        if not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("query must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")

        query_vec = self.embedder.embed_text(query)
        results = self.vector_store.search(query_vec, top_k=top_k)

        if min_score is not None:
            results = [r for r in results if r["score"] >= float(min_score)]

        return results
