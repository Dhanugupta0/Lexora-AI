"""Upload endpoint — POST /api/v1/upload."""
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DBSession, Embedder, VS
from app.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import get_db_context
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import BaseEmbedder
from app.ingestion.parser import ParserError, parse_document
from app.retrieval.vector_store import VectorStore

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def _file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload documents for ingestion",
    response_description="List of accepted document IDs",
)
async def upload_documents(
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
    db: DBSession,
    embedder: Embedder,
    vector_store: VS,
):
    """
    Upload 1–20 documents (PDF, DOCX, TXT). Each file is accepted immediately
    and processed asynchronously. Poll ``GET /documents/{id}`` to check status.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > settings.MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Max {settings.MAX_DOCUMENTS} files per request.",
        )

    accepted: List[dict] = []

    for upload in files:
        ext = _file_extension(upload.filename or "")
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"'{upload.filename}' has unsupported type '{ext}'. "
                       f"Allowed: {ALLOWED_EXTENSIONS}",
            )

        # Read and validate size
        file_bytes = await upload.read()
        if len(file_bytes) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"'{upload.filename}' exceeds {settings.MAX_FILE_SIZE_MB} MB limit.",
            )

        doc_id = str(uuid.uuid4())
        stored_name = f"{doc_id}.{ext}"
        upload_path = Path(settings.UPLOAD_DIR) / stored_name

        # Persist file to disk
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        async with aiofiles.open(upload_path, "wb") as f:
            await f.write(file_bytes)

        # Create DB record
        document = Document(
            id=doc_id,
            filename=stored_name,
            original_filename=upload.filename or stored_name,
            file_type=ext,
            file_size=len(file_bytes),
            status=DocumentStatus.PENDING,
        )
        db.add(document)
        accepted.append({"document_id": doc_id, "filename": upload.filename})

        # Schedule background ingestion
        background_tasks.add_task(
            _ingest_document,
            doc_id=doc_id,
            file_bytes=file_bytes,
            file_type=ext,
            embedder=embedder,
            vector_store=vector_store,
        )

    await db.commit()
    return {"accepted": accepted, "message": "Documents accepted for processing."}


async def _ingest_document(
    doc_id: str,
    file_bytes: bytes,
    file_type: str,
    embedder: BaseEmbedder,
    vector_store: VectorStore,
) -> None:
    """Background task: parse → chunk → embed → store."""
    async with get_db_context() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            logger.error("Document %s not found for ingestion", doc_id)
            return

        try:
            doc.status = DocumentStatus.PROCESSING
            await db.commit()

            # 1. Parse
            parsed = parse_document(file_bytes, file_type)
            doc.page_count = parsed.page_count

            # 2. Chunk
            chunks = chunk_text(parsed.pages)

            # 3. Embed
            texts = [c.text for c in chunks]
            embeddings = await embedder.embed(texts)

            # 4. Store in ChromaDB
            chroma_ids = await vector_store.upsert_chunks(
                document_id=doc_id,
                chunk_texts=texts,
                embeddings=embeddings,
                chunk_indices=[c.chunk_index for c in chunks],
                page_numbers=[c.page_number for c in chunks],
            )

            # 5. Persist chunk metadata to PostgreSQL
            for c, chroma_id in zip(chunks, chroma_ids):
                db.add(
                    Chunk(
                        document_id=doc_id,
                        chunk_index=c.chunk_index,
                        text=c.text,
                        token_count=c.token_count,
                        chroma_id=chroma_id,
                        page_number=c.page_number,
                    )
                )

            doc.chunk_count = len(chunks)
            doc.status = DocumentStatus.READY
            doc.processed_time = datetime.utcnow()

        except ParserError as exc:
            logger.warning("Parse error for document %s: %s", doc_id, exc)
            doc.status = DocumentStatus.ERROR
            doc.error_message = str(exc)
        except Exception as exc:
            logger.exception("Unexpected error ingesting document %s", doc_id)
            doc.status = DocumentStatus.ERROR
            doc.error_message = f"Internal error: {exc}"
