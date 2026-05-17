import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import Document, DocumentStatus, get_db
from app.pipeline import delete_document_vectors

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    page_count: Optional[int]
    chunk_count: Optional[int]
    status: DocumentStatus
    error_message: Optional[str]
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
    docs = q.offset(offset).limit(limit).all()
    return [_to_out(d) for d in docs]


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

    try:
        delete_document_vectors(document_id)
    except Exception as exc:
        logger.warning("Could not remove ChromaDB vectors for %s: %s", document_id, exc)

    for ext in ("pdf", "docx", "txt"):
        path = Path(settings.UPLOAD_DIR) / f"{document_id}.{ext}"
        if path.exists():
            os.remove(path)
            break

    db.delete(doc)
    return {"message": f"Document '{document_id}' deleted."}


def _to_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        status=doc.status,
        error_message=doc.error_message,
        upload_time=doc.upload_time.isoformat(),
        processed_time=doc.processed_time.isoformat() if doc.processed_time else None,
    )
