"""Query endpoint — POST /api/v1/query."""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import DBSession, Embedder, LLM, VS
from app.config import get_settings
from app.retrieval.prompt_builder import build_prompt

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to these document IDs (omit to search all)",
    )


class SourceChunk(BaseModel):
    document_id: str
    chunk_index: int
    page_number: Optional[int]
    text: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    question: str


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG pipeline",
)
async def query(
    body: QueryRequest,
    embedder: Embedder,
    vector_store: VS,
    llm: LLM,
    db: DBSession,
):
    """
    Embed the question, retrieve the most relevant chunks, build a prompt,
    and return the LLM-generated answer with source attribution.
    """
    # 1. Embed query
    try:
        query_embedding = await embedder.embed_query(body.question)
    except Exception as exc:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=502, detail=f"Embedding service error: {exc}") from exc

    # 2. Vector search
    try:
        results = await vector_store.search(
            query_embedding=query_embedding,
            top_k=body.top_k,
            document_ids=body.document_ids,
        )
    except Exception as exc:
        logger.exception("Vector search failed")
        raise HTTPException(status_code=502, detail=f"Vector store error: {exc}") from exc

    if not results:
        return QueryResponse(
            question=body.question,
            answer="No relevant documents found. Please upload documents first.",
            sources=[],
        )

    # 3. Build prompt and call LLM
    system_prompt, user_prompt = build_prompt(body.question, results)
    try:
        answer = await llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}") from exc

    sources = [
        SourceChunk(
            document_id=r.document_id,
            chunk_index=r.chunk_index,
            page_number=r.page_number,
            text=r.text[:500],  # truncate for response brevity
            relevance_score=round(1 - r.score, 4),
        )
        for r in results
    ]

    return QueryResponse(question=body.question, answer=answer, sources=sources)
