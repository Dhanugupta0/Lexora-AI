"""Ingestion endpoint. Files are accepted fast and processed in the background."""

import logging
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import Document, DocumentStatus, get_db, utcnow
from app.rag.parser import SUPPORTED_TYPES
from app.rag.pipeline import ingest_document

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


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
        if ext not in SUPPORTED_TYPES:
            raise HTTPException(
                415,
                f"'{upload.filename}' -- unsupported type '{ext}'. "
                f"Allowed: {', '.join(sorted(SUPPORTED_TYPES))}",
            )

        file_bytes = upload.file.read()
        if not file_bytes:
            raise HTTPException(400, f"'{upload.filename}' is empty.")
        if len(file_bytes) > settings.max_file_size_bytes:
            raise HTTPException(413, f"'{upload.filename}' exceeds the {settings.MAX_FILE_SIZE_MB} MB limit.")

        doc_id = str(uuid.uuid4())
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        (Path(settings.UPLOAD_DIR) / f"{doc_id}.{ext}").write_bytes(file_bytes)

        db.add(Document(
            id=doc_id,
            filename=upload.filename or f"{doc_id}.{ext}",
            file_type=ext,
            file_size=len(file_bytes),
            status=DocumentStatus.PENDING,
        ))
        db.commit()

        accepted.append({"document_id": doc_id, "filename": upload.filename})
        background_tasks.add_task(
            _ingest, doc_id=doc_id, file_bytes=file_bytes, file_type=ext,
            filename=upload.filename or "",
        )

    return {
        "accepted": accepted,
        "count": len(accepted),
        "message": "Accepted. Track progress with GET /api/v1/documents/{document_id}.",
    }


def _ingest(doc_id: str, file_bytes: bytes, file_type: str, filename: str) -> None:
    """Background worker. Records the failing stage so errors are diagnosable."""
    from app.database import SessionLocal

    db: Session = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            logger.error("_ingest: document %s vanished before processing", doc_id)
            return

        doc.status = DocumentStatus.PROCESSING
        doc.stage = "starting"
        doc.error_message = None
        db.commit()

        def on_stage(name: str) -> None:
            doc.stage = name
            db.commit()

        result = ingest_document(doc_id, file_bytes, file_type, filename, on_stage=on_stage)

        doc.title = result.title[:512]
        doc.page_count = result.page_count
        doc.chunk_count = result.chunk_count
        doc.token_count = result.token_count
        doc.ingest_seconds = result.seconds
        doc.status = DocumentStatus.READY
        doc.stage = "done"
        doc.processed_time = utcnow()
        db.commit()

    except Exception as exc:                                 # noqa: BLE001
        logger.exception("Ingestion failed for %s", doc_id)
        db.rollback()
        try:
            doc = db.get(Document, doc_id)
            if doc:
                doc.status = DocumentStatus.ERROR
                doc.error_message = str(exc)[:1000]
                db.commit()
        except Exception:                                    # noqa: BLE001
            logger.exception("Could not record the ingestion error for %s", doc_id)
    finally:
        db.close()
