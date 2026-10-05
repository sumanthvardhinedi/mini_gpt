"""
Phase 12: Retrieval-Augmented Generation (RAG) Package for MiniGPT.
"""

from .chunking import TextChunker, chunk_documents, chunk_text
from .documents import SUPPORTED_EXTENSIONS, load_document, load_documents
from .embeddings import EmbeddingModel, LocalEmbeddingModel
from .pipeline import (
    DEFAULT_RAG_EVAL_QUESTIONS,
    MiniGPTRAGGenerator,
    RAGGenerator,
    RAGPipeline,
    UNKNOWN_ANSWER_MESSAGE,
    evaluate_rag,
)
from .prompt import DEFAULT_SYSTEM_INSTRUCTIONS, PromptBuilder, build_rag_prompt
from .retriever import Retriever
from .vector_store import LocalVectorStore, cosine_similarity

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "load_document",
    "load_documents",
    "TextChunker",
    "chunk_text",
    "chunk_documents",
    "EmbeddingModel",
    "LocalEmbeddingModel",
    "cosine_similarity",
    "LocalVectorStore",
    "Retriever",
    "DEFAULT_SYSTEM_INSTRUCTIONS",
    "PromptBuilder",
    "build_rag_prompt",
    "UNKNOWN_ANSWER_MESSAGE",
    "RAGGenerator",
    "MiniGPTRAGGenerator",
    "RAGPipeline",
    "DEFAULT_RAG_EVAL_QUESTIONS",
    "evaluate_rag",
]
