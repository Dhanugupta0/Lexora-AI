"""Health and introspection endpoints.

`/health` is a fast liveness probe. `/health/deep` loads the embedding model and
reports every component, and `/stats` summarises the corpus -- both are useful
when the deployed app misbehaves and you cannot attach a debugger.
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import Chunk, Document, DocumentStatus, get_db
from app.rag.pipeline import health_snapshot

router = APIRouter()
settings = get_settings()


@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}


@router.get("/health/deep")
def health_deep() -> Dict[str, Any]:
    snapshot = health_snapshot(deep=True)
    snapshot.update(app=settings.APP_NAME, version=settings.APP_VERSION)
    return snapshot


@router.get("/stats")
def stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    by_status = dict(
        db.query(Document.status, func.count(Document.id)).group_by(Document.status).all()
    )
    return {
        "documents": {
            "total": sum(by_status.values()),
            **{s.value: by_status.get(s, 0) for s in DocumentStatus},
        },
        "chunks": db.query(func.count(Chunk.id)).scalar() or 0,
        "tokens_indexed": db.query(func.coalesce(func.sum(Chunk.token_count), 0)).scalar() or 0,
        "pipeline": health_snapshot(deep=False),
    }
