"""
Pytest test suite for MiniGPT Phase 12: Retrieval-Augmented Generation (RAG).
Covers all 15 required test scenarios:
1. Document loading
2. Chunking
3. Chunk overlap
4. Embedding generation
5. Vector similarity
6. Vector store insertion
7. Vector store persistence
8. Top-k retrieval
9. Retriever ranking
10. Prompt construction
11. Source preservation
12. RAG pipeline
13. Unknown-question behavior
14. Evaluation metrics
15. End-to-end RAG demo
"""

from pathlib import Path
import tempfile
from rag.chunking import TextChunker, chunk_text
from rag.documents import load_document, load_documents
from rag.embeddings import LocalEmbeddingModel
from rag.pipeline import (
    RAGPipeline,
    UNKNOWN_ANSWER_MESSAGE,
    evaluate_rag,
)
from rag.prompt import build_rag_prompt
from rag.vector_store import LocalVectorStore, cosine_similarity
from scripts.rag_demo import run_demo


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DOCS_DIR = PROJECT_ROOT / "rag_data" / "documents"


def test_document_loading() -> None:
    """1. Verify .txt and .md files load with text, source, and metadata."""
    docs = load_documents(RAG_DOCS_DIR)
    assert len(docs) == 5

    sources = [d["source"] for d in docs]
    assert "attention_notes.md" in sources
    assert "minigpt_notes.txt" in sources

    for doc in docs:
        assert isinstance(doc["text"], str) and len(doc["text"]) > 50
        assert isinstance(doc["source"], str)
        assert isinstance(doc["metadata"], dict)
        assert doc["metadata"]["extension"] in {".txt", ".md"}


def test_chunking() -> None:
    """2. Verify deterministic chunking splits documents within chunk_size and preserves metadata."""
    doc = load_document(RAG_DOCS_DIR / "attention_notes.md")
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for idx, ch in enumerate(chunks):
        assert len(ch["text"]) <= 300
        assert ch["source"] == "attention_notes.md"
        assert ch["chunk_id"] == f"attention_notes.md::chunk_{idx}"
        assert ch["metadata"]["chunk_index"] == idx


def test_chunk_overlap() -> None:
    """3. Verify adjacent chunks share exact overlapping text of length chunk_overlap."""
    sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(
        sample,
        source="alphabet.txt",
        chunk_size=20,
        chunk_overlap=6,
    )
    assert len(chunks) >= 3
    # End of chunk 0 (last 6 chars) must equal start of chunk 1 (first 6 chars)
    assert chunks[0]["text"][-6:] == chunks[1]["text"][:6]
    assert chunks[1]["text"][-6:] == chunks[2]["text"][:6]


def test_embedding_generation() -> None:
    """4. Verify embed_text and embed_documents return deterministic L2-normalized vectors."""
    embedder = LocalEmbeddingModel(embedding_dim=64)
    v1 = embedder.embed_text("Causal masking blocks future tokens in self-attention.")
    v2 = embedder.embed_text("Causal masking blocks future tokens in self-attention.")
    assert len(v1) == 64
    assert v1 == v2

    batch = embedder.embed_documents(["Tokenization with BPE", "Positional embeddings"])
    assert len(batch) == 2
    assert len(batch[0]) == 64
    assert len(batch[1]) == 64


def test_vector_similarity() -> None:
    """5. Verify cosine_similarity scores identical vectors at 1.0 and related texts higher than unrelated."""
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6

    embedder = LocalEmbeddingModel(embedding_dim=128)
    q_vec = embedder.embed_text("What is causal masking in self-attention?")
    rel_vec = embedder.embed_text("Causal masking prevents attending to future tokens in self-attention.")
    unrel_vec = embedder.embed_text("Byte Pair Encoding merges adjacent character pairs into subwords.")

    sim_rel = cosine_similarity(q_vec, rel_vec)
    sim_unrel = cosine_similarity(q_vec, unrel_vec)
    assert sim_rel > sim_unrel
    assert sim_rel > 0.35


def test_vector_store_insertion() -> None:
    """6. Verify LocalVectorStore.add_documents stores vectors, chunks, sources, and metadata."""
    embedder = LocalEmbeddingModel(embedding_dim=64)
    store = LocalVectorStore(embedder=embedder)

    chunks = chunk_text("MiniGPT uses pre-norm Transformer blocks.", source="notes.md", chunk_size=100, chunk_overlap=20)
    added = store.add_documents(chunks)
    assert added == 1
    assert len(store) == 1
    assert store.records[0]["source"] == "notes.md"
    assert len(store.records[0]["vector"]) == 64


def test_vector_store_persistence() -> None:
    """7. Verify LocalVectorStore saves to JSON and loads back with identical search results."""
    embedder = LocalEmbeddingModel(embedding_dim=64)
    store_a = LocalVectorStore(embedder=embedder)
    docs = load_documents(RAG_DOCS_DIR)
    chunks = TextChunker(chunk_size=400, chunk_overlap=80).chunk_documents(docs)
    store_a.add_documents(chunks)

    q_vec = embedder.embed_text("What is causal masking?")
    res_before = store_a.search(q_vec, top_k=2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        idx_file = Path(tmp_dir) / "index.json"
        store_a.save(idx_file)
        assert idx_file.exists()

        store_b = LocalVectorStore(embedder=embedder)
        store_b.load(idx_file)
        assert len(store_b) == len(store_a)
        res_after = store_b.search(q_vec, top_k=2)
        assert res_before == res_after


def test_top_k_retrieval() -> None:
    """8. Verify Retriever.retrieve returns exactly top_k items when enough chunks exist."""
    pipeline = RAGPipeline(chunk_size=300, chunk_overlap=50)
    pipeline.ingest(RAG_DOCS_DIR)

    res_2 = pipeline.retrieve("How does BPE tokenization work?", top_k=2)
    res_4 = pipeline.retrieve("How does BPE tokenization work?", top_k=4)
    assert len(res_2) == 2
    assert len(res_4) == 4


def test_retriever_ranking() -> None:
    """9. Verify Retriever returns chunks sorted in descending order of cosine similarity score."""
    pipeline = RAGPipeline(chunk_size=400, chunk_overlap=80)
    pipeline.ingest(RAG_DOCS_DIR)

    results = pipeline.retrieve("What is causal masking in self-attention?", top_k=4)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["source"] == "attention_notes.md"


def test_prompt_construction() -> None:
    """10. Verify build_rag_prompt includes instructions, retrieved context, source references, and question."""
    sample_chunks = [
        {
            "chunk": "Causal masking sets future positions to -inf before softmax.",
            "source": "attention_notes.md",
            "score": 0.89,
            "metadata": {},
        }
    ]
    prompt = build_rag_prompt("What is causal masking?", sample_chunks)
    assert "using ONLY the provided context" in prompt
    assert "Do not invent unsupported facts" in prompt
    assert "The answer is not contained in the provided context." in prompt
    assert "[1] Source: attention_notes.md" in prompt
    assert "What is causal masking?" in prompt


def test_source_preservation() -> None:
    """11. Verify source document names and metadata survive from loading through retrieval and answer."""
    pipeline = RAGPipeline(chunk_size=450, chunk_overlap=90)
    pipeline.ingest(RAG_DOCS_DIR)

    ans = pipeline.answer("How many trainable parameters does MiniGPT have?", top_k=2)
    assert "minigpt_notes.txt" in ans["sources"]
    for ch in ans["retrieved_chunks"]:
        assert "path" in ch["metadata"]
        assert "chunk_id" in ch["metadata"]


def test_rag_pipeline() -> None:
    """12. Verify RAGPipeline.ingest, .retrieve, .generate, and .answer work end-to-end."""
    pipeline = RAGPipeline(chunk_size=500, chunk_overlap=100)
    count = pipeline.ingest(RAG_DOCS_DIR)
    assert count >= 5

    retrieved = pipeline.retrieve("What is causal masking?", top_k=2)
    assert len(retrieved) == 2

    gen_str = pipeline.generate("What is causal masking?", top_k=2)
    assert isinstance(gen_str, str) and len(gen_str) > 10

    ans_obj = pipeline.answer("What is causal masking?", top_k=2)
    assert ans_obj["supported"] is True
    assert "attention_notes.md" in ans_obj["sources"]
    assert "[1]" in ans_obj["answer"]


def test_unknown_question_behavior() -> None:
    """13. Verify out-of-domain questions not in the documents return the explicit unsupported message."""
    pipeline = RAGPipeline(chunk_size=500, chunk_overlap=100)
    pipeline.ingest(RAG_DOCS_DIR)

    ans = pipeline.answer("What is the atmospheric pressure on the surface of Venus in kilopascals?")
    assert ans["supported"] is False
    assert ans["answer"] == UNKNOWN_ANSWER_MESSAGE
    assert ans["sources"] == []


def test_evaluation_metrics() -> None:
    """14. Verify evaluate_rag computes hit rate, top-1 accuracy, and answer grounding without an LLM judge."""
    pipeline = RAGPipeline(chunk_size=500, chunk_overlap=100)
    pipeline.ingest(RAG_DOCS_DIR)

    metrics = evaluate_rag(pipeline, top_k=3)
    assert metrics["num_questions"] == 5
    assert metrics["retrieval_hit_rate"] == 1.0
    assert metrics["top_1_accuracy"] >= 0.8
    assert metrics["answer_grounding_rate"] >= 0.8


def test_end_to_end_rag_demo() -> None:
    """15. Verify scripts/rag_demo.py runs end-to-end, saves the vector index, and returns grounded answers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_file = Path(tmp_dir) / "vector_index.json"
        demo_out = run_demo(
            question="What is causal masking?",
            top_k=2,
            docs_dir=RAG_DOCS_DIR,
            index_path=index_file,
            base_dir=PROJECT_ROOT,
        )
        assert index_file.exists()
        assert demo_out["num_chunks"] >= 5
        assert demo_out["result"]["supported"] is True
        assert "attention_notes.md" in demo_out["result"]["sources"]
