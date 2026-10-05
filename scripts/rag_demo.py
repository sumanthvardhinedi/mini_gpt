"""
MiniGPT Phase 12: Retrieval-Augmented Generation (RAG) CLI Demo.

Usage from the `mini-gpt` directory:
    python scripts/rag_demo.py
    python scripts/rag_demo.py --question "What is causal masking?" --top-k 2
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import RAGPipeline, evaluate_rag  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the MiniGPT RAG demo."""
    parser = argparse.ArgumentParser(
        description="Run the MiniGPT Phase 12 Retrieval-Augmented Generation (RAG) demo."
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="rag_data/documents",
        help="Directory containing local .txt and .md knowledge base files.",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="rag_data/index/vector_index.json",
        help="Path to save the built local vector index JSON.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is causal masking?",
        help="Question to query against the RAG pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of top chunks to retrieve (default: 2).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters (default: 500).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap in characters (default: 100).",
    )
    return parser.parse_args(argv)


def run_demo(
    question: str = "What is causal masking?",
    top_k: int = 2,
    docs_dir: Path | str = "rag_data/documents",
    index_path: Path | str = "rag_data/index/vector_index.json",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute the end-to-end RAG ingestion, index persistence, query answering,
    and retrieval benchmark evaluation.
    """
    root = base_dir if base_dir is not None else PROJECT_ROOT
    resolved_docs = Path(docs_dir) if Path(docs_dir).is_absolute() else root / docs_dir
    resolved_index = Path(index_path) if Path(index_path).is_absolute() else root / index_path

    pipeline = RAGPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    num_chunks = pipeline.ingest(resolved_docs)
    pipeline.save_index(resolved_index)

    result = pipeline.answer(question, top_k=top_k)
    eval_metrics = evaluate_rag(pipeline, top_k=3)

    return {
        "num_chunks": num_chunks,
        "index_path": str(resolved_index),
        "result": result,
        "eval_metrics": eval_metrics,
    }


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the CLI RAG demo and print retrieved sources, grounded answer, citations, and metrics."""
    args = parse_args(argv)
    demo_out = run_demo(
        question=args.question,
        top_k=args.top_k,
        docs_dir=args.docs_dir,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        base_dir=PROJECT_ROOT,
    )

    res = demo_out["result"]
    eval_metrics = demo_out["eval_metrics"]

    print("Question:")
    print(res["question"])
    print("\nRetrieved Sources:")
    for idx, chunk in enumerate(res["retrieved_chunks"], start=1):
        print(f"{idx}. {chunk['source']} — score: {chunk['score']:.2f}")

    print("\nAnswer:")
    print(res["answer"])

    print("\nSources:")
    if res["sources"]:
        for idx, src in enumerate(res["sources"], start=1):
            print(f"[{idx}] {src}")
    else:
        print("None (answer not contained in context)")

    print("\n--- Retrieval Benchmark Evaluation (5 Factual Questions) ---")
    print(f"Top-3 Retrieval Hit Rate : {eval_metrics['retrieval_hit_rate'] * 100:.1f}%")
    print(f"Top-1 Source Accuracy    : {eval_metrics['top_1_accuracy'] * 100:.1f}%")
    print(f"Answer Grounding Rate    : {eval_metrics['answer_grounding_rate'] * 100:.1f}%")
    return demo_out


if __name__ == "__main__":
    main()
