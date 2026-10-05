"""
Phase 12: End-to-End Retrieval-Augmented Generation (RAG) Pipeline & Evaluation.

Provides:
- `RAGGenerator`: Replaceable generator interface so either the educational `MiniGPT`
  model or a stronger local/hosted LLM can be plugged in without changing retrieval.
- `MiniGPTRAGGenerator`: Connects the retrieved context and prompt to MiniGPT while
  providing honest, context-grounded sentence extraction + citation formatting (and
  explicitly reporting when a question is not answered by the retrieved context).
- `RAGPipeline`: Full pipeline supporting `.ingest()`, `.retrieve()`, `.generate()`,
  and `.answer()`.
- `evaluate_rag()`: Non-LLM benchmark evaluator measuring retrieval hit rate, top-1
  source accuracy, and answer grounding on factual questions over `rag_data/documents/`.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from generation.generate import generate as minigpt_generate
from rag.chunking import TextChunker
from rag.documents import load_documents
from rag.embeddings import EmbeddingModel, LocalEmbeddingModel, STOPWORDS, _normalize_word
from rag.prompt import PromptBuilder
from rag.retriever import Retriever
from rag.vector_store import LocalVectorStore


UNKNOWN_ANSWER_MESSAGE = "The answer is not contained in the provided context."


class RAGGenerator:
    """
    Replaceable interface for the generation stage of a RAG pipeline.
    """

    def generate(
        self,
        prompt: str,
        question: str,
        retrieved_chunks: Sequence[Dict[str, Any]],
    ) -> str:
        raise NotImplementedError("Subclasses must implement generate(...).")


class MiniGPTRAGGenerator(RAGGenerator):
    """
    Default local RAG generator that pairs grounded context extraction with optional
    autoregressive continuation from the trained educational `MiniGPT` model.

    Note: Because MiniGPT is a tiny 4,304-parameter educational model trained for 5
    epochs on CPU, it cannot synthesize fluent multi-sentence answers on its own.
    This generator selects the highest-overlap sentences directly from the retrieved
    chunks with `[1]` source citations, and optionally appends a short MiniGPT token
    continuation for inspection.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        min_score_threshold: float = 0.18,
        max_new_tokens: int = 8,
        include_minigpt_continuation: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.min_score_threshold = min_score_threshold
        self.max_new_tokens = max_new_tokens
        self.include_minigpt_continuation = include_minigpt_continuation

    def _query_terms(self, question: str) -> List[str]:
        words = re.findall(r"[a-zA-Z0-9_]+", question.lower())
        return [_normalize_word(w) for w in words if w not in STOPWORDS]

    def generate(
        self,
        prompt: str,
        question: str,
        retrieved_chunks: Sequence[Dict[str, Any]],
    ) -> str:
        if not retrieved_chunks:
            return UNKNOWN_ANSWER_MESSAGE

        top_score = float(retrieved_chunks[0].get("score", 0.0))
        if top_score < self.min_score_threshold:
            return UNKNOWN_ANSWER_MESSAGE

        q_terms = set(self._query_terms(question))
        if not q_terms:
            return UNKNOWN_ANSWER_MESSAGE

        # Verify that at least one meaningful query term appears in the top retrieved chunks
        combined_top_text = " ".join(
            str(c.get("chunk", c.get("text", ""))).lower() for c in retrieved_chunks[:2]
        )
        combined_stems = {
            _normalize_word(w) for w in re.findall(r"[a-zA-Z0-9_]+", combined_top_text)
        }
        if len(q_terms.intersection(combined_stems)) == 0:
            return UNKNOWN_ANSWER_MESSAGE

        # Rank individual sentences across top retrieved chunks by query term overlap
        candidate_sentences: List[tuple[float, int, str, str]] = []
        for rank_idx, chunk_item in enumerate(retrieved_chunks[:3], start=1):
            raw_chunk = str(chunk_item.get("chunk", chunk_item.get("text", "")))
            source = str(chunk_item.get("source", "unknown"))
            chunk_score = float(chunk_item.get("score", 0.0))

            # Clean markdown headings and split into sentences/lines
            cleaned_lines = [
                re.sub(r"^#+\s*", "", line).strip()
                for line in raw_chunk.splitlines()
                if line.strip()
            ]
            joined = " ".join(cleaned_lines)
            sentences = re.split(r"(?<=[.!?])\s+", joined)
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 15:
                    continue
                sent_words = {
                    _normalize_word(w) for w in re.findall(r"[a-zA-Z0-9_]+", sent_clean.lower())
                }
                overlap = len(q_terms.intersection(sent_words))
                score = overlap * 2.0 + chunk_score
                candidate_sentences.append((score, rank_idx, source, sent_clean))

        if not candidate_sentences:
            top_chunk = str(retrieved_chunks[0].get("chunk", "")).strip()
            top_source = str(retrieved_chunks[0].get("source", "unknown"))
            base_answer = f"{top_chunk} [1: {top_source}]"
        else:
            candidate_sentences.sort(key=lambda item: (-item[0], item[1]))
            best_sents = candidate_sentences[:4]
            parts = [f"{sent} [{r_idx}]" for _, r_idx, _, sent in best_sents]
            base_answer = " ".join(parts)

        if self.include_minigpt_continuation and self.model is not None and self.tokenizer is not None:
            short_seed = question.strip().split()[0] if question.strip() else "MiniGPT"
            mgpt_out = minigpt_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=short_seed,
                max_new_tokens=self.max_new_tokens,
                greedy=True,
            )
            return f"{base_answer}\n(MiniGPT decoder continuation: {mgpt_out})"

        return base_answer


class RAGPipeline:
    """
    Complete from-scratch Retrieval-Augmented Generation pipeline:
        ingest(documents) -> retrieve(query) -> generate(query) -> answer(query)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedder: Optional[EmbeddingModel] = None,
        vector_store: Optional[LocalVectorStore] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        generator: Optional[RAGGenerator] = None,
        min_score_threshold: float = 0.18,
    ) -> None:
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = embedder or LocalEmbeddingModel()
        self.vector_store = vector_store or LocalVectorStore(embedder=self.embedder)
        self.retriever = Retriever(vector_store=self.vector_store, embedder=self.embedder)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.min_score_threshold = min_score_threshold
        self.generator = generator or MiniGPTRAGGenerator(
            min_score_threshold=min_score_threshold
        )

    def ingest(
        self,
        documents_or_path: Union[str, Path, Sequence[Union[str, Path, Dict[str, Any]]]],
    ) -> int:
        """
        Load (if paths/directory given), chunk, embed, and index documents in the vector store.

        Returns:
            Total number of chunks indexed.
        """
        if isinstance(documents_or_path, (str, Path)):
            docs = load_documents(documents_or_path)
        elif isinstance(documents_or_path, Sequence):
            first = documents_or_path[0] if len(documents_or_path) > 0 else None
            if isinstance(first, dict):
                docs = list(documents_or_path)  # type: ignore[arg-type]
            else:
                docs = load_documents(documents_or_path)  # type: ignore[arg-type]
        else:
            raise TypeError("documents_or_path must be a path or sequence of paths/dicts.")

        chunks = self.chunker.chunk_documents(docs)
        if not chunks:
            return 0

        vectors = self.embedder.embed_documents([c["text"] for c in chunks])
        return self.vector_store.add_documents(chunks, vectors=vectors)

    def save_index(self, path: Union[str, Path]) -> Path:
        """Save the underlying vector index to `path`."""
        return self.vector_store.save(path)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load the underlying vector index from `path`."""
        self.vector_store.load(path)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top-k relevant chunks for `query`."""
        return self.retriever.retrieve(query=query, top_k=top_k)

    def generate(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant chunks, construct a grounded prompt, and generate an answer string."""
        retrieved = self.retrieve(query=query, top_k=top_k)
        prompt = self.prompt_builder.build_prompt(question=query, retrieved_chunks=retrieved)
        return self.generator.generate(
            prompt=prompt,
            question=query,
            retrieved_chunks=retrieved,
        )

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for `query` and return structured answer + source citations.
        """
        retrieved = self.retrieve(query=query, top_k=top_k)
        prompt = self.prompt_builder.build_prompt(question=query, retrieved_chunks=retrieved)
        answer_text = self.generator.generate(
            prompt=prompt,
            question=query,
            retrieved_chunks=retrieved,
        )

        is_unknown = answer_text.strip() == UNKNOWN_ANSWER_MESSAGE
        if is_unknown:
            sources: List[str] = []
        else:
            seen: List[str] = []
            for item in retrieved:
                src = str(item.get("source", "unknown"))
                if src not in seen:
                    seen.append(src)
            sources = seen

        return {
            "question": query,
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved,
            "prompt": prompt,
            "supported": not is_unknown,
        }


DEFAULT_RAG_EVAL_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question": "What is causal masking and how does it prevent looking at future tokens?",
        "expected_source": "attention_notes.md",
        "expected_keyword": "future",
    },
    {
        "question": "How does Byte Pair Encoding (BPE) build its subword vocabulary from characters?",
        "expected_source": "tokenization_notes.md",
        "expected_keyword": "merge",
    },
    {
        "question": "Why does MiniGPT add positional embeddings to token embeddings?",
        "expected_source": "embeddings_notes.md",
        "expected_keyword": "order",
    },
    {
        "question": "What activation function and normalization order does the TransformerBlock use?",
        "expected_source": "transformer_notes.md",
        "expected_keyword": "gelu",
    },
    {
        "question": "How many trainable parameters does MiniGPT have and what validation perplexity did it reach?",
        "expected_source": "minigpt_notes.txt",
        "expected_keyword": "4,304",
    },
]


def evaluate_rag(
    pipeline: RAGPipeline,
    eval_questions: Optional[Sequence[Dict[str, Any]]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate retrieval hit rate, top-1 source accuracy, and answer grounding without an LLM judge.
    """
    questions = list(eval_questions) if eval_questions is not None else list(DEFAULT_RAG_EVAL_QUESTIONS)
    if not questions:
        raise ValueError("eval_questions cannot be empty.")

    top_k_hits = 0
    top_1_hits = 0
    grounded_hits = 0
    details: List[Dict[str, Any]] = []

    for item in questions:
        q = str(item["question"])
        expected_source = str(item["expected_source"])
        expected_keyword = str(item.get("expected_keyword", "")).lower()

        result = pipeline.answer(q, top_k=top_k)
        retrieved_sources = [str(c["source"]) for c in result["retrieved_chunks"]]

        hit_top_k = expected_source in retrieved_sources
        hit_top_1 = len(retrieved_sources) > 0 and retrieved_sources[0] == expected_source
        answer_lower = str(result["answer"]).lower()
        is_grounded = (
            hit_top_k
            and expected_source in result["sources"]
            and (not expected_keyword or expected_keyword in answer_lower)
        )

        if hit_top_k:
            top_k_hits += 1
        if hit_top_1:
            top_1_hits += 1
        if is_grounded:
            grounded_hits += 1

        details.append(
            {
                "question": q,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "top_1_match": hit_top_1,
                "top_k_hit": hit_top_k,
                "answer_grounded": is_grounded,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        )

    n = len(questions)
    return {
        "num_questions": n,
        "top_k": top_k,
        "retrieval_hit_rate": top_k_hits / n,
        "top_1_accuracy": top_1_hits / n,
        "answer_grounding_rate": grounded_hits / n,
        "details": details,
    }
