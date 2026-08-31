"""FastAPI entry point."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import get_settings
from app.database import init_db

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("lexora")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(application: FastAPI):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PATH, exist_ok=True)
    init_db()

    # Warm the sparse index so the first query isn't slowed by a cold rebuild.
    try:
        from app.rag.keyword import get_keyword_index
        get_keyword_index().invalidate()
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("Keyword index warm-up skipped: %s", exc)

    logger.info(
        "%s %s ready | llm=%s | embeddings=%s | retrieval=%s | reranker=%s",
        settings.APP_NAME, settings.APP_VERSION, settings.GROQ_MODEL,
        settings.EMBEDDING_MODEL, settings.RETRIEVAL_MODE, settings.RERANKER_MODE,
    )
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not set -- answering will run in extractive fallback mode.")
    if settings.EMBEDDING_PROVIDER.lower() == "jina" and not settings.JINA_API_KEY:
        logger.warning("JINA_API_KEY is not set -- embeddings will fall back to the local ONNX model.")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Hybrid-retrieval RAG with query planning, reranking and grounded, "
        "citation-verified answers."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a stack trace to the client; always log the full one."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
