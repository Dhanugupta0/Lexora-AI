"""API router — aggregates all sub-routers under /api/v1."""
from fastapi import APIRouter

from app.api import documents, query, upload

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(upload.router, tags=["ingestion"])
api_router.include_router(query.router, tags=["query"])
api_router.include_router(documents.router, tags=["documents"])
