"""Question answering endpoint."""

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag.pipeline import answer_question

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=20)
    document_ids: Optional[List[str]] = Field(
        default=None, description="Restrict the search to specific documents (omit = search all)."
    )
    history: Optional[List[Turn]] = Field(
        default=None, description="Prior turns, so follow-up questions resolve their pronouns."
    )
    verify_with_llm: bool = Field(
        default=False,
        description="Run an extra NLI-style pass over borderline claims. Slower, stricter.",
    )
    include_trace: bool = Field(default=True, description="Return per-stage timings and decisions.")


class SourceOut(BaseModel):
    document_id: str
    filename: str
    doc_title: str
    section: str
    page_number: int
    text_preview: str
    relevance_score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrievers: List[str] = []
    cited: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceOut]
    intent: str
    confidence: float = Field(..., description="0-1 faithfulness of the answer to its sources.")
    grounded: bool = Field(..., description="True when every claim was verified against a source.")
    abstained: bool = Field(..., description="True when support was too weak and the model declined.")
    degraded: bool = Field(..., description="True when a fallback path produced this answer.")
    model: str
    plan: Dict[str, Any] = {}
    retrieval: Dict[str, Any] = {}
    grounding: Dict[str, Any] = {}
    trace: Dict[str, Any] = {}
    warnings: List[str] = []


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    try:
        result = answer_question(
            body.question,
            top_k=body.top_k,
            document_ids=body.document_ids,
            history=[turn.model_dump() for turn in body.history] if body.history else None,
            verify_with_llm=body.verify_with_llm,
            include_trace=body.include_trace,
        )
    except Exception as exc:                                 # noqa: BLE001
        logger.exception("Query pipeline failed")
        raise HTTPException(500, f"Query failed: {exc}") from exc

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceOut(
                document_id=s.document_id,
                filename=s.filename,
                doc_title=s.doc_title,
                section=s.section,
                page_number=s.page_number,
                text_preview=s.text[:600],
                relevance_score=s.score,
                dense_score=s.dense_score,
                sparse_score=s.sparse_score,
                rerank_score=s.rerank_score,
                retrievers=s.retrievers,
                cited=s.cited,
            )
            for s in result.sources
        ],
        intent=result.intent,
        confidence=result.confidence,
        grounded=result.grounded,
        abstained=result.abstained,
        degraded=result.degraded,
        model=result.model,
        plan=result.plan,
        retrieval=result.retrieval,
        grounding=result.grounding,
        trace=result.trace,
        warnings=result.warnings,
    )
