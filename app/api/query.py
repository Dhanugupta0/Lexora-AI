import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.pipeline import generate_answer, search_chunks

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: Optional[List[str]] = Field(
        default=None, description="Filter to specific documents (omit = search all)"
    )


class Source(BaseModel):
    document_id: str
    page_number: int
    text_preview: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    results = search_chunks(
        question=body.question,
        top_k=body.top_k,
        document_ids=body.document_ids,
    )

    try:
        answer = generate_answer(body.question, results)
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(502, f"LLM error: {exc}")

    sources = [
        Source(
            document_id=r.document_id,
            page_number=r.page_number,
            text_preview=r.text[:400],
            relevance_score=r.score,
        )
        for r in results
    ]

    return QueryResponse(question=body.question, answer=answer, sources=sources)
