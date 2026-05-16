"""Shared FastAPI dependencies."""
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.ingestion.embedder import BaseEmbedder, get_embedder
from app.llm.base import BaseLLMProvider
from app.llm.factory import get_llm_provider
from app.retrieval.vector_store import VectorStore

# ── Database session ──────────────────────────────────────────────────────────

DBSession = Annotated[AsyncSession, Depends(get_db)]

# ── Singletons injected into route handlers ───────────────────────────────────
# These are module-level singletons so they're only constructed once per process.

_embedder: BaseEmbedder | None = None
_vector_store: VectorStore | None = None
_llm_provider: BaseLLMProvider | None = None


def _get_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_llm() -> BaseLLMProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


Embedder = Annotated[BaseEmbedder, Depends(_get_embedder)]
VS = Annotated[VectorStore, Depends(_get_vector_store)]
LLM = Annotated[BaseLLMProvider, Depends(_get_llm)]
