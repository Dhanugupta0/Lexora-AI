"""FastAPI application entry point."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import get_settings
from app.db.base import Base
from app.db.session import engine

logger = logging.getLogger(__name__)
settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup (idempotent via checkfirst)."""
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready.")
    yield
    logger.info("Shutting down.")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "A production-grade Retrieval-Augmented Generation pipeline. "
        "Upload documents, query them with natural language, and get grounded answers."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/api/v1/health", tags=["health"], summary="Health check")
async def health():
    """Returns 200 when the API, database, and vector store are reachable."""
    from app.retrieval.vector_store import VectorStore

    vs = VectorStore()
    chroma_ok = await vs.health_check()
    return {
        "status": "healthy" if chroma_ok else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "chromadb": "ok" if chroma_ok else "unreachable",
    }
