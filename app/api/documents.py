"""Document management endpoints."""

import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import Document, DocumentStatus, get_db
from app.rag.parser import SUPPORTED_TYPES
from app.rag.pipeline import delete_document as purge_document, forget_filename

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentOut(BaseModel):
    id: str
    filename: str
    title: Optional[str] = None
    file_type: str
    file_size: int
    page_count: Optional[int]
    chunk_count: Optional[int]
    token_count: Optional[int] = None
    status: DocumentStatus
    stage: Optional[str] = None
    error_message: Optional[str]
    ingest_seconds: Optional[float] = None
    upload_time: str
    processed_time: Optional[str]


@router.get("/documents", response_model=List[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    status: Optional[DocumentStatus] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    q = db.query(Document).order_by(Document.upload_time.desc())
    if status:
        q = q.filter(Document.status == status)
    return [_to_out(d) for d in q.offset(offset).limit(limit).all()]


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, f"Document '{document_id}' not found.")
    return _to_out(doc)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, f"Document '{document_id}' not found.")

    purge_document(document_id)          # vectors + chunk rows + BM25 invalidation
    forget_filename(document_id)

    for ext in SUPPORTED_TYPES:
        path = Path(settings.UPLOAD_DIR) / f"{document_id}.{ext}"
        if path.exists():
            try:
                os.remove(path)
            except OSError as exc:
                logger.warning("Could not remove %s: %s", path, exc)
            break

    db.delete(doc)
    return {"message": f"Document '{document_id}' deleted."}


def _to_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        title=doc.title,
        file_type=doc.file_type,
        file_size=doc.file_size,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        token_count=doc.token_count,
        status=doc.status,
        stage=doc.stage,
        error_message=doc.error_message,
        ingest_seconds=doc.ingest_seconds,
        upload_time=doc.upload_time.isoformat(),
        processed_time=doc.processed_time.isoformat() if doc.processed_time else None,
    )
