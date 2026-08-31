"""Hybrid retrieval: dense + sparse -> RRF fusion -> MMR diversification.

Pipeline for one question:

    query (+ sub-queries, + HyDE probe)
        |-- dense  : embed -> ChromaDB cosine ANN        (semantic recall)
        |-- sparse : BM25 over the chunk corpus          (exact-token recall)
        v
    Reciprocal Rank Fusion  -- rank-based, so the two score scales never need
                               calibrating against each other
        v
    MMR                     -- trades a little relevance for coverage so five
                               near-duplicate chunks don't crowd out the answer
        v
    top-k candidates for the reranker

Each retriever is independently fault-tolerant: if the vector store is down the
BM25 arm still answers, and vice versa.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag.embedder import get_embedder
from app.rag.keyword import get_keyword_index
from app.rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    section: str = ""
    doc_title: str = ""
    score: float = 0.0                  # fused score, 0..1
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    sources: List[str] = field(default_factory=list)     # which retrievers found it

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

    @property
    def label(self) -> str:
        bits = [self.doc_title or self.document_id[:8]]
        if self.section:
            bits.append(self.section)
        bits.append(f"p.{self.page_number}")
        return " > ".join(bits)


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(rankings: Dict[str, List[str]], k: int) -> Dict[str, float]:
    """RRF: score(d) = sum over lists of 1 / (k + rank(d)).

    Rank-based fusion is robust precisely because it ignores the raw scores --
    a cosine of 0.71 and a BM25 of 14.2 are not comparable, but "3rd place" and
    "1st place" always are.
    """
    fused: Dict[str, float] = {}
    for ids in rankings.values():
        for rank, chunk_id in enumerate(ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def maximal_marginal_relevance(
    query_vector: Sequence[float],
    candidates: List[RetrievedChunk],
    vectors: Dict[str, List[float]],
    top_k: int,
    lambda_: float,
) -> List[RetrievedChunk]:
    """Greedy MMR -- pick the next chunk that is relevant *and* not redundant."""
    pool = [c for c in candidates if c.chunk_id in vectors]
    if len(pool) <= 1:
        return candidates[:top_k]

    selected: List[RetrievedChunk] = []
    remaining = list(pool)

    while remaining and len(selected) < top_k:
        best, best_score = None, -float("inf")
        for candidate in remaining:
            relevance = _cosine(query_vector, vectors[candidate.chunk_id])
            redundancy = max(
                (_cosine(vectors[candidate.chunk_id], vectors[s.chunk_id]) for s in selected),
                default=0.0,
            )
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = candidate, score
        selected.append(best)
        remaining.remove(best)

    # anything without a stored vector keeps its fused ordering at the tail
    leftovers = [c for c in candidates if c not in selected]
    return (selected + leftovers)[:top_k]


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class HybridRetriever:
    def __init__(self) -> None:
        self.vectors = get_vector_store()
        self.keywords = get_keyword_index()

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        document_ids: Optional[Sequence[str]] = None,
        extra_queries: Optional[Sequence[str]] = None,
        mode: Optional[str] = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.DEFAULT_TOP_K
        mode = (mode or settings.RETRIEVAL_MODE).lower()
        pool = max(settings.CANDIDATE_POOL, top_k * 4)
        # sub-queries and the HyDE probe widen recall; the primary query leads
        queries = [query] + [q for q in (extra_queries or []) if q and q.strip() != query.strip()]

        started = time.time()
        rankings: Dict[str, List[str]] = {}
        registry: Dict[str, RetrievedChunk] = {}
        notes: List[str] = []
        query_vector: Optional[List[float]] = None
        dense_ok = sparse_ok = False

        # -- dense arm ------------------------------------------------------ #
        if mode in ("hybrid", "dense"):
            try:
                embedder = get_embedder()
                for i, sub_query in enumerate(queries):
                    vector = embedder.embed_query(sub_query)
                    if i == 0:
                        query_vector = vector
                    hits = self.vectors.search(vector, pool, document_ids)
                    rankings[f"dense::{i}"] = [h.chunk_id for h in hits]
                    for hit in hits:
                        entry = registry.setdefault(hit.chunk_id, _from_dense(hit))
                        entry.dense_score = max(entry.dense_score or 0.0, hit.score)
                        if "dense" not in entry.sources:
                            entry.sources.append("dense")
                dense_ok = True
            except Exception as exc:                         # noqa: BLE001
                logger.warning("Dense retrieval failed (%s) -- continuing with sparse only", exc)
                notes.append(f"dense retrieval unavailable: {exc}")

        # -- sparse arm ----------------------------------------------------- #
        if mode in ("hybrid", "sparse"):
            try:
                for i, sub_query in enumerate(queries):
                    hits = self.keywords.search(sub_query, pool, document_ids)
                    rankings[f"sparse::{i}"] = [h.chunk_id for h in hits]
                    for hit in hits:
                        entry = registry.setdefault(hit.chunk_id, _from_sparse(hit))
                        entry.sparse_score = max(entry.sparse_score or 0.0, hit.score)
                        if "sparse" not in entry.sources:
                            entry.sources.append("sparse")
                sparse_ok = True
            except Exception as exc:                         # noqa: BLE001
                logger.warning("Sparse retrieval failed (%s) -- continuing with dense only", exc)
                notes.append(f"sparse retrieval unavailable: {exc}")

        if not registry:
            return RetrievalResult(
                chunks=[], degraded=not (dense_ok or sparse_ok), notes=notes,
                stats={"mode": mode, "queries": len(queries), "candidates": 0,
                       "latency_ms": int((time.time() - started) * 1000)},
            )

        # -- fuse ----------------------------------------------------------- #
        fused = reciprocal_rank_fusion(rankings, settings.RRF_K)
        peak = max(fused.values()) or 1.0
        for chunk_id, score in fused.items():
            if chunk_id in registry:
                registry[chunk_id].score = round(score / peak, 4)

        candidates = sorted(registry.values(), key=lambda c: c.score, reverse=True)
        kept = [c for c in candidates if c.score >= settings.MIN_RELEVANCE] or candidates[:top_k]
        candidates_before_mmr = len(kept)

        # -- diversify ------------------------------------------------------ #
        depth = max(top_k, settings.RERANK_INPUT_SIZE)
        if settings.MMR_ENABLED and query_vector and len(kept) > top_k:
            try:
                shortlist = kept[: max(depth * 2, top_k * 4)]
                vectors = self.vectors.get_embeddings([c.chunk_id for c in shortlist])
                if vectors:
                    kept = maximal_marginal_relevance(
                        query_vector, shortlist, vectors, depth, settings.MMR_LAMBDA
                    )
                else:
                    kept = kept[:depth]
            except Exception as exc:                         # noqa: BLE001
                logger.warning("MMR failed (%s) -- falling back to fused order", exc)
                kept = kept[:depth]
        else:
            kept = kept[:depth]

        return RetrievalResult(
            chunks=kept,
            degraded=not (dense_ok and sparse_ok) if mode == "hybrid" else False,
            notes=notes,
            stats={
                "mode": mode,
                "queries": len(queries),
                "dense_ok": dense_ok,
                "sparse_ok": sparse_ok,
                "candidates": candidates_before_mmr,
                "returned": len(kept),
                "latency_ms": int((time.time() - started) * 1000),
            },
        )


def _from_dense(hit) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=hit.chunk_id, document_id=hit.document_id, chunk_index=hit.chunk_index,
        page_number=hit.page_number, text=hit.text, section=hit.section, doc_title=hit.doc_title,
    )


def _from_sparse(hit) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=hit.chunk_id, document_id=hit.document_id, chunk_index=hit.chunk_index,
        page_number=hit.page_number, text=hit.text, section=hit.section, doc_title=hit.doc_title,
    )


_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
