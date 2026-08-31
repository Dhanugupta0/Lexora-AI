"""Turn retrieved passages into a cited answer.

Three generation paths, chosen by intent and by whether the LLM is reachable:

    grounded    normal path -- numbered passages in, [S#]-cited answer out
    chitchat    greetings and small talk, no retrieval, no citations
    extractive  the LLM is unreachable: return the best passages verbatim with
                their sources. Strictly worse than a written answer, but it is
                still true, still cited, and still useful -- which beats a 502.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag import prompts
from app.rag.llm import LLMUnavailable, get_llm
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Generation:
    answer: str
    model: str = ""
    mode: str = "grounded"           # grounded | chitchat | no_context | extractive
    degraded: bool = False
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    notes: List[str] = field(default_factory=list)


def build_context(chunks: Sequence[RetrievedChunk], token_budget: int = 6000) -> tuple[str, List[str]]:
    """Numbered passage block, trimmed to fit the context budget.

    Returns the rendered block and the plain passage texts in the same order, so
    `[S3]` in the answer always maps to `passages[2]` during verification.
    """
    from app.rag.chunker import count_tokens

    blocks: List[str] = []
    passages: List[str] = []
    used = 0

    for i, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        cost = count_tokens(text) + 30
        if used + cost > token_budget and passages:
            break
        blocks.append(f"[S{i}] Source: {chunk.label}\n{text}")
        passages.append(text)
        used += cost

    return "\n\n".join(blocks), passages


class Generator:
    def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        intent: str = "document_qa",
        history: Optional[Sequence[Dict[str, str]]] = None,
        document_hint: str = "",
    ) -> tuple[Generation, List[str]]:
        """Returns the generation plus the passage texts used (for grounding)."""
        started = time.time()
        llm = get_llm()

        if intent in ("chitchat", "meta"):
            return self._chitchat(question, history, document_hint, started), []

        if not chunks:
            return self._no_context(question, started), []

        context, passages = build_context(chunks)
        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": prompts.ANSWER_USER.format(
                context=context, question=question)},
        ]

        try:
            result = llm.complete(messages, tag="answer")
            return Generation(
                answer=result.text,
                model=result.model,
                mode="grounded",
                degraded=result.degraded,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                notes=["answered by a fallback model"] if result.degraded else [],
            ), passages
        except LLMUnavailable as exc:
            logger.warning("Generation unavailable (%s) -- serving extractive answer", exc)
            return self._extractive(chunks, str(exc), started), passages

    # ------------------------------------------------------------- variants --
    def _chitchat(self, question, history, document_hint, started) -> Generation:
        llm = get_llm()
        hint = f"\n\nContext for you only: {document_hint}" if document_hint else ""
        messages = [{"role": "system", "content": prompts.CHITCHAT_SYSTEM.format(doc_hint=hint)}]
        for turn in list(history or [])[-4:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = str(turn.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content[:600]})
        messages.append({"role": "user", "content": question})

        try:
            result = llm.complete(messages, tag="chitchat", max_tokens=400,
                                  temperature=0.5, reasoning_effort="low")
            return Generation(result.text, result.model, "chitchat",
                              result.degraded, result.latency_ms,
                              result.prompt_tokens, result.completion_tokens)
        except LLMUnavailable:
            fallback = ("Hi! I'm LexoraAI. Upload a PDF, DOCX or text file and I'll answer "
                        "questions about it with citations back to the exact page.")
            if document_hint:
                fallback = f"Hi! I'm LexoraAI. {document_hint} Ask me anything about them."
            return Generation(fallback, "", "chitchat", degraded=True,
                              latency_ms=int((time.time() - started) * 1000),
                              notes=["LLM unavailable -- canned reply"])

    def _no_context(self, question, started) -> Generation:
        llm = get_llm()
        try:
            result = llm.complete(
                [{"role": "system", "content": prompts.NO_CONTEXT_SYSTEM},
                 {"role": "user", "content": question}],
                tag="no_context", max_tokens=300, reasoning_effort="low",
            )
            return Generation(result.text, result.model, "no_context",
                              result.degraded, result.latency_ms,
                              result.prompt_tokens, result.completion_tokens)
        except LLMUnavailable:
            return Generation(
                "I couldn't find anything about that in your uploaded documents. "
                "Try rephrasing with wording closer to the document, or upload the file that covers it.",
                "", "no_context", degraded=True,
                latency_ms=int((time.time() - started) * 1000),
                notes=["LLM unavailable -- canned reply"],
            )

    def _extractive(self, chunks: Sequence[RetrievedChunk], reason: str, started) -> Generation:
        """No LLM: hand back the evidence itself rather than nothing."""
        lines = [
            "The answer model is temporarily unavailable, so here are the most relevant "
            "passages from your documents, highest match first:",
            "",
        ]
        for i, chunk in enumerate(chunks[:4], start=1):
            snippet = re.sub(r"\s+", " ", chunk.text).strip()
            if len(snippet) > 600:
                snippet = snippet[:600].rsplit(" ", 1)[0] + " ..."
            lines.append(f"**[S{i}] {chunk.label}** (match {chunk.final_score:.0%})")
            lines.append(f"> {snippet}")
            lines.append("")

        return Generation(
            answer="\n".join(lines).strip(),
            model="",
            mode="extractive",
            degraded=True,
            latency_ms=int((time.time() - started) * 1000),
            notes=[f"LLM unavailable: {reason}"],
        )


_generator: Optional[Generator] = None


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator
