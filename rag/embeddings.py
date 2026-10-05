"""
Phase 12: Replaceable Local Embedding Interface for MiniGPT RAG.

Provides:
- `EmbeddingModel`: Clean abstract interface supporting `embed_text(text)` and
  `embed_documents(texts)` so stronger local or hosted models can be swapped in later.
- `LocalEmbeddingModel`: Deterministic, zero-API local embedding model combining:
  1. Technical/domain keyword & stem anchor dimensions
  2. Word unigram & bigram feature hashing
  3. Character 3-gram feature hashing for subword/morphological similarity
  followed by L2 unit-vector normalization (`||v||_2 = 1.0`).
"""

import hashlib
import math
import re
from typing import Dict, List, Sequence


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

DOMAIN_ANCHOR_TERMS: List[str] = [
    "token",
    "tokenization",
    "tokenizer",
    "bpe",
    "byte",
    "pair",
    "encoding",
    "subword",
    "merge",
    "merges",
    "unk",
    "vocabulary",
    "regex",
    "embedding",
    "embeddings",
    "tokenembedding",
    "positional",
    "positionalembedding",
    "inputembedding",
    "vector",
    "dense",
    "sequence",
    "position",
    "attention",
    "self",
    "causal",
    "mask",
    "masking",
    "future",
    "query",
    "key",
    "value",
    "softmax",
    "scaled",
    "dot",
    "multi",
    "head",
    "multiheadattention",
    "transformer",
    "transformerblock",
    "block",
    "layernorm",
    "normalization",
    "residual",
    "skip",
    "feedforward",
    "gelu",
    "dropout",
    "minigpt",
    "gpt",
    "decoder",
    "parameters",
    "4304",
    "perplexity",
    "loss",
    "adamw",
    "epochs",
    "greedy",
    "temperature",
    "topk",
    "autoregressive",
]


def _normalize_word(word: str) -> str:
    w = word.lower().strip()
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _stable_hash(feature: str) -> int:
    digest = hashlib.md5(feature.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


class EmbeddingModel:
    """
    Replaceable base interface for text embedding models in the RAG pipeline.
    """

    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement embed_text(text).")

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class LocalEmbeddingModel(EmbeddingModel):
    """
    Lightweight, deterministic local embedding model requiring no external APIs.
    Produces L2-normalized dense vectors of length `embedding_dim`.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim < 16:
            raise ValueError(f"embedding_dim must be an integer >= 16, got {embedding_dim!r}.")
        self.embedding_dim = embedding_dim
        self._anchor_map: Dict[str, int] = {
            term: idx % embedding_dim for idx, term in enumerate(DOMAIN_ANCHOR_TERMS)
        }

    def embed_text(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")

        vec = [0.0 for _ in range(self.embedding_dim)]
        raw_words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        content_words = [w for w in raw_words if w not in STOPWORDS]

        if not content_words and raw_words:
            content_words = raw_words

        # 1. Word unigram & stemmed features + explicit anchor term boost
        for w in content_words:
            stem = _normalize_word(w)
            for token_variant, weight in ((w, 2.5), (stem, 2.0)):
                if token_variant in self._anchor_map:
                    vec[self._anchor_map[token_variant]] += weight * 2.0
                h = _stable_hash(f"word::{token_variant}")
                idx = h % self.embedding_dim
                sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
                vec[idx] += sign * weight

            # 2. Character 3-grams for morphological/subword overlap
            padded = f"<{w}>"
            if len(padded) >= 3:
                for i in range(len(padded) - 2):
                    trigram = padded[i : i + 3]
                    h_tri = _stable_hash(f"tri::{trigram}")
                    idx_tri = h_tri % self.embedding_dim
                    vec[idx_tri] += 0.35

        # 3. Content word bigrams
        for i in range(len(content_words) - 1):
            bg = f"{_normalize_word(content_words[i])}_{_normalize_word(content_words[i + 1])}"
            h_bg = _stable_hash(f"bg::{bg}")
            idx_bg = h_bg % self.embedding_dim
            vec[idx_bg] += 1.5

        # L2 normalize so cosine similarity is a clean dot product in [-1, 1]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not isinstance(texts, (list, tuple)):
            texts = list(texts)
        return [self.embed_text(t) for t in texts]
