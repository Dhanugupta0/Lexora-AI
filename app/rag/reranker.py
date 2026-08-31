"""Precision pass over the retrieved candidates.

Retrieval optimises recall -- pull 20-40 plausible chunks. Reranking optimises
precision -- decide which 5 actually answer *this* question, using a model that
sees the query and passage together instead of comparing two independent
vectors. Fewer, better passages in the prompt is the cheapest hallucination
reduction available.

Three modes, each degrading into the next if it is unavailable:

    llm            listwise scoring by the Groq model  (no extra RAM, ~300ms)
    cross_encoder  local ONNX cross-encoder            (offline, ~90MB RAM)
    heuristic      lexical overlap + coverage + prior  (always works, no I/O)
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag import prompts
from app.rag.keyword import tokenize
from app.rag.llm import get_llm
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class Reranker:
    def __init__(self) -> None:
        self._cross_encoder = None
        self._lock = threading.Lock()
        self._cross_encoder_failed = False

    # ------------------------------------------------------------------ API --
    def rerank(
        self,
        question: str,
        candidates: List[RetrievedChunk],
        top_k: int,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode = (mode or settings.RERANKER_MODE).lower()
        if not candidates:
            return {"chunks": [], "mode": "none", "latency_ms": 0}
        if mode == "off" or len(candidates) <= 1:
            return {"chunks": candidates[:top_k], "mode": "off", "latency_ms": 0}

        pool = candidates[: settings.RERANK_INPUT_SIZE]
        started = time.time()
        applied = mode

        try:
            if mode == "llm":
                scores = self._llm_scores(question, pool)
            elif mode == "cross_encoder":
                scores = self._cross_encoder_scores(question, pool)
            else:
                scores = self._heuristic_scores(question, pool)
        except Exception as exc:                             # noqa: BLE001
            logger.warning("Reranker '%s' failed (%s) -- falling back to heuristic", mode, exc)
            scores = self._heuristic_scores(question, pool)
            applied = "heuristic"

        if not scores:
            scores = self._heuristic_scores(question, pool)
            applied = "heuristic"

        for chunk in pool:
            chunk.rerank_score = round(scores.get(chunk.chunk_id, 0.0), 4)

        # Blend in the retrieval score so reranker noise can't fully override a
        # candidate both retrievers strongly agreed on.
        ordered = sorted(pool, key=lambda c: (0.85 * (c.rerank_score or 0.0) + 0.15 * c.score), reverse=True)
        return {
            "chunks": ordered[:top_k],
            "mode": applied,
            "scored": len(pool),
            "latency_ms": int((time.time() - started) * 1000),
        }

    # ------------------------------------------------------------- llm mode --
    def _llm_scores(self, question: str, candidates: List[RetrievedChunk]) -> Dict[str, float]:
        llm = get_llm()
        if not llm.available:
            raise RuntimeError("LLM unavailable for reranking")

        passages = "\n\n".join(
            f"[{i}] ({c.label})\n{_clip(c.text, 700)}" for i, c in enumerate(candidates, start=1)
        )
        payload = llm.complete_json(
            [
                {"role": "system", "content": prompts.RERANK_SYSTEM},
                {"role": "user", "content": prompts.RERANK_USER.format(
                    question=question, passages=passages)},
            ],
            temperature=0.0,
            max_tokens=1400,
            reasoning_effort="low",
            tag="rerank",
            default={},
        )

        entries = payload.get("scores")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("reranker returned no scores")

        scores: Dict[str, float] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("id", 0)) - 1
                value = float(entry.get("score", 0))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates):
                scores[candidates[index].chunk_id] = max(0.0, min(10.0, value)) / 10.0
        return scores

    # --------------------------------------------------- cross-encoder mode --
    def _cross_encoder_scores(self, question: str, candidates: List[RetrievedChunk]) -> Dict[str, float]:
        model = self._load_cross_encoder()
        raw = list(model.rerank(question, [_clip(c.text, 900) for c in candidates]))
        # cross-encoder logits are unbounded -- squash to 0..1
        return {c.chunk_id: 1 / (1 + math.exp(-float(s))) for c, s in zip(candidates, raw)}

    def _load_cross_encoder(self):
        if self._cross_encoder_failed:
            raise RuntimeError("cross-encoder previously failed to load")
        if self._cross_encoder is None:
            with self._lock:
                if self._cross_encoder is None:
                    try:
                        from fastembed.rerank.cross_encoder import TextCrossEncoder
                        self._cross_encoder = TextCrossEncoder(model_name=settings.RERANKER_MODEL)
                        logger.info("Cross-encoder ready: %s", settings.RERANKER_MODEL)
                    except Exception as exc:                 # noqa: BLE001
                        self._cross_encoder_failed = True
                        raise RuntimeError(f"cross-encoder unavailable: {exc}") from exc
        return self._cross_encoder

    # ------------------------------------------------------- heuristic mode --
    @staticmethod
    def _heuristic_scores(question: str, candidates: List[RetrievedChunk]) -> Dict[str, float]:
        """Zero-dependency lexical scorer: term coverage + IDF weight + density."""
        query_terms = set(tokenize(question))
        if not query_terms:
            return {c.chunk_id: c.score for c in candidates}

        # crude IDF over the candidate pool -- rare terms matter more
        doc_freq: Dict[str, int] = {}
        tokenized = []
        for candidate in candidates:
            tokens = tokenize(candidate.text)
            tokenized.append(tokens)
            for term in set(tokens) & query_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        total = len(candidates)
        scores: Dict[str, float] = {}
        for candidate, tokens in zip(candidates, tokenized):
            token_set = set(tokens)
            matched = token_set & query_terms
            if not matched:
                scores[candidate.chunk_id] = 0.05 * candidate.score
                continue
            idf = sum(math.log(1 + total / (1 + doc_freq.get(term, 0))) for term in matched)
            idf_max = sum(math.log(1 + total / 1.0) for _ in query_terms) or 1.0
            coverage = len(matched) / len(query_terms)
            density = min(1.0, sum(tokens.count(t) for t in matched) / max(len(tokens), 1) * 12)
            scores[candidate.chunk_id] = round(
                0.5 * coverage + 0.3 * (idf / idf_max) + 0.2 * density, 4
            )
        return scores

    def snapshot(self) -> Dict[str, Any]:
        return {
            "mode": settings.RERANKER_MODE,
            "cross_encoder_model": settings.RERANKER_MODEL,
            "cross_encoder_loaded": self._cross_encoder is not None,
            "input_size": settings.RERANK_INPUT_SIZE,
        }


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
