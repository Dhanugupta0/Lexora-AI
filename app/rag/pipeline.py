"""The orchestrator -- the only module the API layer talks to.

Ingestion
    parse -> chunk -> embed -> index (dense + sparse), with per-stage status on
    the document row so a failure tells you exactly where it broke.

Answering
    plan -> retrieve -> rerank -> generate -> verify

Every stage is timed and reported back in `trace`, so the API response shows how
an answer was produced rather than asking you to trust it.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag.chunker import Chunk, chunk_blocks
from app.rag.embedder import get_embedder
from app.rag.generator import get_generator
from app.rag.grounding import apply_grounding
from app.rag.keyword import get_keyword_index
from app.rag.llm import get_llm
from app.rag.parser import ParsedDocument, parse_document
from app.rag.reasoning import plan_query
from app.rag.reranker import get_reranker
from app.rag.retriever import RetrievedChunk, get_retriever
from app.rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
class Trace:
    """Records how long each stage took and what it decided."""

    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, Any]] = {}
        self._started = time.time()

    @contextmanager
    def stage(self, name: str):
        start = time.time()
        entry: Dict[str, Any] = {}
        self.stages[name] = entry
        try:
            yield entry
        finally:
            entry["ms"] = int((time.time() - start) * 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {"total_ms": int((time.time() - self._started) * 1000), "stages": self.stages}


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
@dataclass
class IngestResult:
    document_id: str
    title: str
    page_count: int
    chunk_count: int
    token_count: int
    seconds: float
    trace: Dict[str, Any] = field(default_factory=dict)


def ingest_document(
    document_id: str,
    file_bytes: bytes,
    file_type: str,
    filename: str = "",
    on_stage: Optional[Any] = None,
) -> IngestResult:
    """Parse, chunk, embed and index one document.

    `on_stage(name)` is called before each stage so the caller can persist
    progress. Raises on failure -- the caller records the error.
    """
    trace = Trace()
    started = time.time()

    def announce(name: str) -> None:
        if on_stage:
            try:
                on_stage(name)
            except Exception:                                # noqa: BLE001
                logger.debug("stage callback failed for %s", name, exc_info=True)

    # -- parse -------------------------------------------------------------- #
    announce("parsing")
    with trace.stage("parse") as entry:
        parsed: ParsedDocument = parse_document(file_bytes, file_type, filename)
        entry.update(blocks=len(parsed.blocks), pages=parsed.page_count,
                     chars=parsed.char_count, title=parsed.title[:80])

    # -- chunk -------------------------------------------------------------- #
    announce("chunking")
    with trace.stage("chunk") as entry:
        chunks: List[Chunk] = chunk_blocks(parsed.blocks, doc_title=parsed.title)
        if not chunks:
            raise ValueError("Document produced no chunks -- it may be empty or unreadable.")
        entry.update(chunks=len(chunks), strategy=settings.CHUNK_STRATEGY,
                     avg_tokens=round(sum(c.token_count for c in chunks) / len(chunks), 1))

    # -- embed -------------------------------------------------------------- #
    announce("embedding")
    with trace.stage("embed") as entry:
        embedder = get_embedder()
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
        entry.update(vectors=len(vectors), provider=embedder.provider_name,
                     dimension=embedder.dimension)

    # -- index -------------------------------------------------------------- #
    announce("indexing")
    with trace.stage("index") as entry:
        get_vector_store().upsert(document_id, chunks, vectors, filename=filename)
        _persist_chunks(document_id, chunks, filename)
        get_keyword_index().invalidate()
        entry.update(dense=True, sparse=True)

    elapsed = time.time() - started
    logger.info(
        "Ingested %s (%s): %d chunks, %d pages in %.2fs",
        document_id, filename or "?", len(chunks), parsed.page_count, elapsed,
    )
    return IngestResult(
        document_id=document_id,
        title=parsed.title,
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        token_count=sum(c.token_count for c in chunks),
        seconds=round(elapsed, 2),
        trace=trace.to_dict(),
    )


def _persist_chunks(document_id: str, chunks: Sequence[Chunk], filename: str) -> None:
    """Mirror chunk text into SQL so BM25 can be rebuilt after a restart."""
    from app.database import Chunk as ChunkRow, SessionLocal

    db = SessionLocal()
    try:
        db.query(ChunkRow).filter(ChunkRow.document_id == document_id).delete()
        db.bulk_save_objects([
            ChunkRow(
                id=f"{document_id}::{c.chunk_index}",
                document_id=document_id,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                section=(c.section or "")[:512],
                doc_title=(c.doc_title or "")[:512],
                filename=(filename or "")[:255],
                kind=c.kind,
                token_count=c.token_count,
                text=c.text,
            )
            for c in chunks
        ])
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_document(document_id: str) -> None:
    """Remove a document from both indexes. Best-effort, never raises."""
    try:
        get_vector_store().delete_document(document_id)
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("Could not delete vectors for %s: %s", document_id, exc)

    from app.database import Chunk as ChunkRow, SessionLocal
    db = SessionLocal()
    try:
        db.query(ChunkRow).filter(ChunkRow.document_id == document_id).delete()
        db.commit()
    except Exception as exc:                                 # noqa: BLE001
        db.rollback()
        logger.warning("Could not delete chunk rows for %s: %s", document_id, exc)
    finally:
        db.close()

    get_keyword_index().invalidate()


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    document_id: str
    filename: str
    doc_title: str
    section: str
    page_number: int
    text: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrievers: List[str] = field(default_factory=list)
    cited: bool = False


@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: List[Source] = field(default_factory=list)
    intent: str = "document_qa"
    confidence: float = 0.0
    grounded: bool = False
    abstained: bool = False
    degraded: bool = False
    model: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    grounding: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def answer_question(
    question: str,
    *,
    top_k: Optional[int] = None,
    document_ids: Optional[Sequence[str]] = None,
    history: Optional[Sequence[Dict[str, str]]] = None,
    verify_with_llm: bool = False,
    include_trace: bool = True,
) -> AnswerResult:
    """Run the full RAG pipeline for one question."""
    top_k = top_k or settings.DEFAULT_TOP_K
    trace = Trace()
    warnings: List[str] = []

    # -- 1. plan ------------------------------------------------------------ #
    with trace.stage("plan") as entry:
        plan = plan_query(question, history)
        entry.update(plan.to_dict())

    # -- 2. retrieve -------------------------------------------------------- #
    retrieved: List[RetrievedChunk] = []
    retrieval_stats: Dict[str, Any] = {"skipped": True}

    if plan.needs_retrieval:
        with trace.stage("retrieve") as entry:
            result = get_retriever().retrieve(
                plan.resolved_question,
                top_k=top_k,
                document_ids=document_ids,
                extra_queries=plan.retrieval_queries,
            )
            retrieved = result.chunks
            retrieval_stats = result.stats
            warnings.extend(result.notes)
            entry.update(result.stats)

    # -- 3. rerank ---------------------------------------------------------- #
    if retrieved:
        with trace.stage("rerank") as entry:
            outcome = get_reranker().rerank(plan.resolved_question, retrieved, top_k)
            retrieved = outcome["chunks"]
            entry.update({k: v for k, v in outcome.items() if k != "chunks"})

    # -- 4. generate -------------------------------------------------------- #
    with trace.stage("generate") as entry:
        generation, passages = get_generator().generate(
            plan.resolved_question,
            retrieved,
            intent=plan.intent,
            history=history,
            document_hint=_document_hint() if plan.intent in ("chitchat", "meta") else "",
        )
        entry.update(mode=generation.mode, model=generation.model, degraded=generation.degraded,
                     prompt_tokens=generation.prompt_tokens,
                     completion_tokens=generation.completion_tokens)
        warnings.extend(generation.notes)

    answer = generation.answer

    # -- 5. verify ---------------------------------------------------------- #
    grounding: Dict[str, Any] = {"method": "skipped"}
    confidence = 0.0
    grounded = False
    abstained = False

    if generation.mode == "grounded" and passages:
        with trace.stage("verify") as entry:
            answer, report = apply_grounding(answer, passages, use_llm=verify_with_llm)
            grounding = report.to_dict()
            confidence = report.confidence
            grounded = report.verified
            abstained = report.abstained
            warnings.extend(report.notes)
            entry.update(verified=report.verified, confidence=report.confidence,
                         abstained=report.abstained)

        cited = set(_cited_indices(answer))
        for i, chunk in enumerate(retrieved[: len(passages)], start=1):
            chunk_cited = i in cited
            setattr(chunk, "_cited", chunk_cited)
    elif generation.mode == "extractive":
        confidence = 0.35                       # verbatim passages, no synthesis
        warnings.append("answered without the LLM -- passages returned verbatim")
    elif generation.mode in ("chitchat", "no_context"):
        confidence = 1.0 if generation.mode == "chitchat" else 0.0

    sources = [
        Source(
            document_id=c.document_id,
            filename=_filename_for(c),
            doc_title=c.doc_title,
            section=c.section,
            page_number=c.page_number,
            text=c.text,
            score=round(c.final_score, 4),
            dense_score=c.dense_score,
            sparse_score=c.sparse_score,
            rerank_score=c.rerank_score,
            retrievers=c.sources,
            cited=getattr(c, "_cited", False),
        )
        for c in retrieved
    ]

    return AnswerResult(
        question=question,
        answer=answer,
        sources=sources,
        intent=plan.intent,
        confidence=round(confidence, 3),
        grounded=grounded,
        abstained=abstained,
        degraded=generation.degraded or bool(retrieval_stats.get("dense_ok") is False),
        model=generation.model,
        plan=plan.to_dict(),
        retrieval=retrieval_stats,
        grounding=grounding,
        trace=trace.to_dict() if include_trace else {},
        warnings=[w for w in dict.fromkeys(warnings) if w],
    )


def _cited_indices(answer: str) -> List[int]:
    from app.rag.grounding import parse_citations
    return parse_citations(answer)


_filename_cache: Dict[str, str] = {}


def _filename_for(chunk: RetrievedChunk) -> str:
    if chunk.document_id in _filename_cache:
        return _filename_cache[chunk.document_id]
    from app.database import Document, SessionLocal
    db = SessionLocal()
    try:
        doc = db.get(Document, chunk.document_id)
        name = doc.filename if doc else ""
    except Exception:                                        # noqa: BLE001
        name = ""
    finally:
        db.close()
    _filename_cache[chunk.document_id] = name
    return name


def forget_filename(document_id: str) -> None:
    _filename_cache.pop(document_id, None)


def _document_hint() -> str:
    """A one-line summary of the library, so small talk can reference real files."""
    from app.database import Document, DocumentStatus, SessionLocal
    db = SessionLocal()
    try:
        docs = (db.query(Document)
                  .filter(Document.status == DocumentStatus.READY)
                  .order_by(Document.upload_time.desc()).limit(5).all())
        total = db.query(Document).filter(Document.status == DocumentStatus.READY).count()
    except Exception:                                        # noqa: BLE001
        return ""
    finally:
        db.close()

    if not docs:
        return "The user has not uploaded any documents yet."
    names = ", ".join(d.filename for d in docs)
    suffix = f" (and {total - len(docs)} more)" if total > len(docs) else ""
    return f"The user has {total} document(s) ready: {names}{suffix}."


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def health_snapshot(deep: bool = False) -> Dict[str, Any]:
    """Component-level health -- used by /health and /stats."""
    snapshot: Dict[str, Any] = {
        "llm": get_llm().snapshot(),
        "vector_store": get_vector_store().snapshot(),
        "keyword_index": get_keyword_index().snapshot(),
        "reranker": get_reranker().snapshot(),
        "config": {
            "chunk_strategy": settings.CHUNK_STRATEGY,
            "chunk_size_tokens": settings.CHUNK_SIZE_TOKENS,
            "retrieval_mode": settings.RETRIEVAL_MODE,
            "reranker_mode": settings.RERANKER_MODE,
            "query_planning": settings.ENABLE_QUERY_PLANNING,
            "hyde": settings.ENABLE_HYDE,
            "grounding_check": settings.ENABLE_GROUNDING_CHECK,
            "default_top_k": settings.DEFAULT_TOP_K,
        },
    }
    if deep:
        snapshot["embedder"] = get_embedder().snapshot()
    else:
        snapshot["embedder"] = {"provider": settings.EMBEDDING_PROVIDER,
                                "model": settings.EMBEDDING_MODEL}

    healthy = (
        snapshot["vector_store"].get("healthy", False)
        and snapshot["llm"]["circuit"]["state"] != "open"
        and snapshot["llm"]["configured"]
    )
    snapshot["status"] = "ok" if healthy else "degraded"
    return snapshot
