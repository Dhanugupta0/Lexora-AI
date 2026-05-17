import logging
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import Document, DocumentStatus, get_db
from app.pipeline import (
    chunk_text, delete_document_vectors, get_embeddings,
    parse_document, store_chunks,
)

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

ALLOWED = {"pdf", "docx", "txt"}


@router.post("/upload", status_code=202)
def upload_documents(
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(400, "No files provided.")
    if len(files) > settings.MAX_DOCUMENTS:
        raise HTTPException(400, f"Max {settings.MAX_DOCUMENTS} files per request.")

    accepted = []

    for upload in files:
        ext = Path(upload.filename or "").suffix.lower().lstrip(".")
        if ext not in ALLOWED:
            raise HTTPException(415, f"'{upload.filename}' — unsupported type '{ext}'. Allowed: {ALLOWED}")

        file_bytes = upload.file.read()
        if len(file_bytes) > settings.max_file_size_bytes:
            raise HTTPException(413, f"'{upload.filename}' exceeds {settings.MAX_FILE_SIZE_MB} MB limit.")

        doc_id = str(uuid.uuid4())

        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = Path(settings.UPLOAD_DIR) / f"{doc_id}.{ext}"
        file_path.write_bytes(file_bytes)

        doc = Document(
            id=doc_id,
            filename=upload.filename or f"{doc_id}.{ext}",
            file_type=ext,
            file_size=len(file_bytes),
            status=DocumentStatus.PENDING,
        )
        db.add(doc)
        db.commit()

        accepted.append({"document_id": doc_id, "filename": upload.filename})

        background_tasks.add_task(_ingest, doc_id=doc_id, file_bytes=file_bytes, file_type=ext)

    return {"accepted": accepted, "message": "Accepted. Processing in background."}


def _ingest(doc_id: str, file_bytes: bytes, file_type: str) -> None:
    from datetime import datetime
    from app.database import SessionLocal

    db: Session = SessionLocal()
    doc = None
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            logger.error("_ingest: document %s not found in DB", doc_id)
            return

        doc.status = DocumentStatus.PROCESSING
        db.commit()

        parsed = parse_document(file_bytes, file_type)
        doc.page_count = parsed.page_count

        chunks = chunk_text(parsed.pages)

        embeddings = get_embeddings([c.text for c in chunks])

        store_chunks(doc_id, chunks, embeddings)

        doc.chunk_count = len(chunks)
        doc.status = DocumentStatus.READY
        doc.processed_time = datetime.utcnow()
        db.commit()
        logger.info("Ingested document %s — %d chunks", doc_id, len(chunks))

    except Exception as exc:
        logger.exception("Failed to ingest document %s: %s", doc_id, exc)
        if doc:
            doc.status = DocumentStatus.ERROR
            doc.error_message = str(exc)
            db.commit()
    finally:
        db.close()
