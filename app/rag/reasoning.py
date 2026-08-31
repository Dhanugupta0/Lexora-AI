"""Query understanding: route, rewrite, decompose, hypothesise.

Retrieval quality is capped by the query you hand it. A raw user message is
usually a bad search query -- it carries pronouns that only resolve against the
chat history ("what about the second one?"), it bundles two questions into one,
and it is phrased as a question while the passage that answers it is phrased as
a statement.

One LLM call fixes all of that at once and produces a `QueryPlan`:

    intent               routes the request -- chitchat and meta questions skip
                         retrieval entirely instead of hallucinating over noise
    standalone_question  history-resolved, self-contained
    search_queries       decomposed into keyword-shaped sub-queries
    hypothetical_answer  a HyDE probe: an invented answer embeds far closer to
                         the real passage than the question does

Two safety nets: obvious small talk is caught by a regex fast-path before any
network call, and if the planner call fails the pipeline falls back to a plan
built from the raw question, so a planner outage degrades quality but never
breaks answering.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag import prompts
from app.rag.llm import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()

INTENTS = ("chitchat", "document_qa", "summarize", "compare", "meta")

# Words that carry no topic on their own -- a message made only of these is a
# pure follow-up and has nothing worth searching for verbatim.
_FUNCTION_WORDS = {
    "about", "and", "what", "how", "the", "it", "that", "this", "those", "these",
    "one", "ones", "second", "third", "first", "last", "next", "other", "another",
    "of", "for", "is", "are", "was", "were", "do", "does", "did", "so", "then",
    "also", "too", "much", "many", "more", "why", "when", "where", "who", "which",
    "me", "my", "i", "you", "your", "we", "our", "they", "them", "there", "here",
}


# Back-references that only a previous turn can resolve.
_ANAPHORA = re.compile(
    r"\b(it|its|it's|that|this|these|those|they|them|their|he|she|him|her|his|hers|"
    r"same|one|ones|above|former|latter|previous|earlier|there)\b",
    re.I,
)


def _has_content_words(text: str) -> bool:
    from app.rag.keyword import tokenize
    return any(token not in _FUNCTION_WORDS for token in tokenize(text))


def needs_history_resolution(question: str) -> bool:
    """True when the question cannot be understood without the previous turns.

    Used to decide whether the LLM's rewrite should override the user's own
    words. "and when is it due?" must be rewritten; "what about expenses?"
    names its own subject and must not be, because a rewrite that fuses the old
    topic back in ("the expense policy under STD-441") sends the whole pipeline
    after a question nobody asked.
    """
    text = (question or "").strip()
    if not text:
        return False
    return bool(_ANAPHORA.search(text)) or not _has_content_words(text)


@dataclass
class QueryPlan:
    original_question: str
    standalone_question: str
    intent: str = "document_qa"
    search_queries: List[str] = field(default_factory=list)
    hypothetical_answer: str = ""
    needs_retrieval: bool = True
    planner: str = "heuristic"          # heuristic | llm | fallback
    latency_ms: int = 0

    @property
    def resolved_question(self) -> str:
        """The question the rest of the pipeline should actually answer.

        The rewrite wins only for genuine back-references; otherwise the user's
        own wording is authoritative, because rewriting is the step most likely
        to quietly change what was asked.
        """
        if not self.standalone_question:
            return self.original_question
        if self.planner != "llm":
            return self.standalone_question
        return self.standalone_question if needs_history_resolution(self.original_question) \
            else self.original_question

    @property
    def retrieval_queries(self) -> List[str]:
        """Extra probes handed to the retriever alongside the main question.

        Both phrasings of the question are searched -- the user's own wording and
        the rewrite -- so whichever one was right can win the fusion. Rewriting is
        the most failure-prone step in the pipeline, and this guarantees a bad
        rewrite can never delete the terms the user actually typed.
        """
        seen = {self.resolved_question.strip().lower()}
        extras: List[str] = []

        for alternative in (self.original_question.strip(), self.standalone_question.strip()):
            key = alternative.lower()
            if alternative and key not in seen and _has_content_words(alternative):
                extras.append(alternative)
                seen.add(key)

        for query in self.search_queries:
            key = query.strip().lower()
            if key and key not in seen:
                extras.append(query)
                seen.add(key)

        if settings.ENABLE_HYDE and self.hypothetical_answer:
            extras.append(self.hypothetical_answer)
        return extras[: settings.MAX_SUB_QUERIES + 2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "standalone_question": self.standalone_question,
            "resolved_question": self.resolved_question,
            "search_queries": self.search_queries,
            "hyde_used": bool(settings.ENABLE_HYDE and self.hypothetical_answer),
            "needs_retrieval": self.needs_retrieval,
            "planner": self.planner,
            "latency_ms": self.latency_ms,
        }


# --------------------------------------------------------------------------- #
# Fast paths -- avoid a network round trip for the obvious cases
# --------------------------------------------------------------------------- #
_GREETING = re.compile(
    r"^\s*(hi|hey+|hello+|yo|sup|howdy|hola|namaste|good\s+(morning|afternoon|evening|day)|"
    r"how\s+(are|r)\s+(you|u)|what'?s\s+up|thanks?|thank\s+you|thx|ty|ok(ay)?|cool|nice|great|"
    r"bye|goodbye|see\s+ya|who\s+are\s+you|what\s+can\s+you\s+do|help)\b"
    r"(\s+(there|all|again|team|guys|folks|everyone|mate|buddy|lexora|bot|so\s+much))?"
    r"[\s!.?,]*$",
    re.I,
)
_SELF_INTRO = re.compile(r"^\s*(i\s*am|i'm|my\s+name\s+is|this\s+is)\s+[\w\s.]{1,40}[\s!.?]*$", re.I)
_META = re.compile(
    r"\b(what|which|how\s+many)\b.{0,30}\b(documents?|files?|pdfs?|uploads?)\b.{0,30}"
    r"\b(do\s+i\s+have|are\s+(there|uploaded|loaded)|have\s+i\s+uploaded|did\s+i\s+upload)\b",
    re.I,
)


def _fast_path(question: str) -> Optional[QueryPlan]:
    text = question.strip()
    if not text:
        return None
    if _META.search(text):
        return QueryPlan(text, text, intent="meta", needs_retrieval=False, planner="heuristic")
    if len(text) <= 60 and (_GREETING.match(text) or _SELF_INTRO.match(text)):
        return QueryPlan(text, text, intent="chitchat", needs_retrieval=False, planner="heuristic")
    return None


def _format_history(history: Optional[Sequence[Dict[str, str]]]) -> str:
    if not history:
        return "(no previous messages)"
    turns = list(history)[-settings.MAX_HISTORY_TURNS :]
    lines = []
    for turn in turns:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = re.sub(r"\s+", " ", str(turn.get("content", ""))).strip()
        if content:
            lines.append(f"{role}: {content[:400]}")
    return "\n".join(lines) or "(no previous messages)"


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
def plan_query(question: str, history: Optional[Sequence[Dict[str, str]]] = None) -> QueryPlan:
    """Build a retrieval plan. Never raises -- always returns a usable plan."""
    question = (question or "").strip()

    fast = _fast_path(question)
    if fast is not None:
        return fast

    if not settings.ENABLE_QUERY_PLANNING:
        return QueryPlan(question, question, search_queries=[question], planner="heuristic")

    llm = get_llm()
    if not llm.available:
        return _fallback_plan(question, reason="llm unavailable")

    started = time.time()
    payload = llm.complete_json(
        [
            {"role": "system", "content": prompts.PLANNER_SYSTEM},
            {"role": "user", "content": prompts.PLANNER_USER.format(
                history=_format_history(history), question=question)},
        ],
        temperature=0.0,
        max_tokens=900,
        reasoning_effort="low",
        tag="planner",
        default={},
    )
    if not payload:
        return _fallback_plan(question, reason="planner returned nothing")

    intent = str(payload.get("intent", "document_qa")).strip().lower()
    if intent not in INTENTS:
        intent = "document_qa"

    standalone = str(payload.get("standalone_question") or "").strip() or question

    raw_queries = payload.get("search_queries") or []
    queries = [str(q).strip() for q in raw_queries if isinstance(q, (str, int, float)) and str(q).strip()]
    queries = queries[: settings.MAX_SUB_QUERIES] or [standalone]

    hypothetical = str(payload.get("hypothetical_answer") or "").strip()[:600]

    needs_retrieval = payload.get("needs_retrieval")
    if not isinstance(needs_retrieval, bool):
        needs_retrieval = intent not in ("chitchat", "meta")
    if intent in ("chitchat", "meta"):
        needs_retrieval = False

    return QueryPlan(
        original_question=question,
        standalone_question=standalone,
        intent=intent,
        search_queries=queries,
        hypothetical_answer=hypothetical if intent not in ("chitchat", "meta") else "",
        needs_retrieval=needs_retrieval,
        planner="llm",
        latency_ms=int((time.time() - started) * 1000),
    )


def _fallback_plan(question: str, reason: str) -> QueryPlan:
    logger.info("Query planner degraded (%s) -- using the raw question", reason)
    return QueryPlan(
        original_question=question,
        standalone_question=question,
        search_queries=[question],
        planner="fallback",
    )
