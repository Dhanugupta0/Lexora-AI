"""Prompt builder — injects retrieved chunks into the LLM context window."""
from typing import List

from app.retrieval.vector_store import SearchResult

SYSTEM_PROMPT = """\
You are LexoraAI, a precise and helpful document assistant.
Answer the user's question using ONLY the context excerpts provided below.
If the answer cannot be found in the context, say "I don't have enough information \
in the provided documents to answer that question."
Do not fabricate information. Cite the source document ID when relevant.
"""


def build_prompt(question: str, results: List[SearchResult]) -> tuple[str, str]:
    """Build a (system_prompt, user_prompt) pair from the retrieved chunks.

    Args:
        question: The user's raw question.
        results: Ordered list of retrieved :class:`SearchResult` objects.

    Returns:
        A 2-tuple ``(system_prompt, user_prompt)`` ready to pass to any LLM provider.
    """
    context_blocks: List[str] = []
    for i, r in enumerate(results, start=1):
        page_info = f", page {r.page_number}" if r.page_number else ""
        context_blocks.append(
            f"[{i}] Document ID: {r.document_id}{page_info}\n"
            f"Relevance score: {1 - r.score:.3f}\n"
            f"---\n{r.text}"
        )

    context_section = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    user_prompt = (
        f"Context from uploaded documents:\n\n"
        f"{context_section}\n\n"
        f"---\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return SYSTEM_PROMPT, user_prompt
