"""Documents endpoints — GET /api/v1/documents, GET /documents/{id}, DELETE /documents/{id}."""
import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DBSession, VS
from app.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Response schemas ──────────────────────────────────────────────────────────

class ChunkPreview(BaseModel):
    chunk_index: int
    page_number: Optional[int]
    text_preview: str
    token_count: int


class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    file_type: str
    file_size: int
    page_count: Optional[int]
    chunk_count: Optional[int]
    status: DocumentStatus
    error_message: Optional[str]
    upload_time: str
    processed_time: Optional[str]


class DocumentDetailResponse(DocumentResponse):
    chunks_preview: List[ChunkPreview]


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_to_response(doc: Document) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        original_filename=doc.original_filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        status=doc.status,
        error_message=doc.error_message,
        upload_time=doc.upload_time.isoformat(),
        processed_time=doc.processed_time.isoformat() if doc.processed_time else None,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(
    db: DBSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: Optional[DocumentStatus] = Query(default=None, alias="status"),
):
    """Return paginated document metadata."""
    query = select(Document).order_by(Document.upload_time.desc())
    if status_filter:
        query = query.where(Document.status == status_filter)

    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar_one()

    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    docs = result.scalars().all()

    return DocumentListResponse(
        documents=[_doc_to_response(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document detail with chunk preview",
)
async def get_document(document_id: str, db: DBSession):
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    previews = [
        ChunkPreview(
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            text_preview=c.text[:300],
            token_count=c.token_count,
        )
        for c in sorted(doc.chunks, key=lambda x: x.chunk_index)[:10]  # first 10 chunks
    ]

    base = _doc_to_response(doc)
    return DocumentDetailResponse(**base.model_dump(), chunks_preview=previews)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document and its embeddings",
)
async def delete_document(document_id: str, db: DBSession, vector_store: VS):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    # Remove from ChromaDB
    try:
        await vector_store.delete_by_document_id(document_id)
    except Exception as exc:
        logger.warning("Could not delete vectors for %s: %s", document_id, exc)

    # Remove raw file
    file_path = Path(settings.UPLOAD_DIR) / doc.filename
    if file_path.exists():
        os.remove(file_path)

    # Cascade-deletes chunks via ORM relationship
    await db.delete(doc)
    await db.commit()

    return {"message": f"Document '{document_id}' deleted successfully."}
