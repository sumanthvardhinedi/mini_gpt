"""
Phase 12: Grounded RAG Prompt Construction.

Combines:
- Explicit grounding instructions (answer using the provided context, do not invent
  unsupported facts, state when the answer is not contained in the context, include
  source references)
- Numbered retrieved context passages with source filenames
- The user question
"""

from typing import Any, Dict, Sequence


DEFAULT_SYSTEM_INSTRUCTIONS = (
    "Instructions:\n"
    "1. Answer the user question using ONLY the provided context below.\n"
    "2. Do not invent unsupported facts or extrapolate beyond the context.\n"
    "3. If the answer is not contained in the provided context, state clearly: "
    "\"The answer is not contained in the provided context.\"\n"
    "4. Include source references (such as [1] source_name) for every factual claim."
)


class PromptBuilder:
    """
    Builds structured RAG prompts combining instructions, retrieved context chunks,
    and the user's question.
    """

    def __init__(self, instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS) -> None:
        self.instructions = instructions

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: Sequence[Dict[str, Any]],
    ) -> str:
        if not isinstance(question, str) or len(question.strip()) == 0:
            raise ValueError("question must be a non-empty string.")

        if not retrieved_chunks:
            context_block = "No relevant context chunks were retrieved."
        else:
            formatted_passages = []
            for idx, item in enumerate(retrieved_chunks, start=1):
                source = item.get("source", "unknown_source")
                score = item.get("score", 0.0)
                chunk_text = str(item.get("chunk", item.get("text", ""))).strip()
                formatted_passages.append(
                    f"[{idx}] Source: {source} (similarity: {score:.2f})\n{chunk_text}"
                )
            context_block = "\n\n".join(formatted_passages)

        return (
            f"{self.instructions}\n\n"
            f"Retrieved Context:\n"
            f"{context_block}\n\n"
            f"User Question:\n"
            f"{question.strip()}\n\n"
            f"Answer:"
        )


def build_rag_prompt(
    question: str,
    retrieved_chunks: Sequence[Dict[str, Any]],
    instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
) -> str:
    """Convenience function to construct a grounded RAG prompt."""
    return PromptBuilder(instructions=instructions).build_prompt(question, retrieved_chunks)
