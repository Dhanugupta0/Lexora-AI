"""Shared pytest fixtures for unit and integration tests."""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.base import Base
from app.db.models import Document, DocumentStatus

# ── Test settings override ─────────────────────────────────────────────────────

TEST_SETTINGS = Settings(
    DATABASE_URL="sqlite+aiosqlite:///./test.db",
    DATABASE_SYNC_URL="sqlite:///./test.db",
    CHROMA_HOST="localhost",
    CHROMA_PORT=8000,
    EMBEDDING_PROVIDER="huggingface",
    LLM_PROVIDER="openai",
    OPENAI_API_KEY="test-key",
    UPLOAD_DIR=tempfile.mkdtemp(prefix="lexora_test_"),
)

# ── In-memory SQLite engine for unit/integration tests ────────────────────────

test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
TestAsyncSession = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh in-memory DB for each test function."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with TestAsyncSession() as session:
        yield session
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ── Mock vector store ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.upsert_chunks = AsyncMock(return_value=["chunk_id_0", "chunk_id_1"])
    vs.search = AsyncMock(return_value=[])
    vs.delete_by_document_id = AsyncMock()
    vs.health_check = AsyncMock(return_value=True)
    vs.collection_count = AsyncMock(return_value=0)
    return vs


# ── Mock embedder ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.embed = AsyncMock(return_value=[[0.1] * 384])
    emb.embed_query = AsyncMock(return_value=[0.1] * 384)
    return emb


# ── Mock LLM ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="Test answer from LLM.")
    return llm


# ── Sample file fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_bytes() -> bytes:
    return b"This is a test document.\nIt has multiple lines.\nUsed for unit tests."


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Minimal valid PDF for testing the parser."""
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 700 Td (Hello World) Tj ET\nendstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n0000000000 65535 f\n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )


# ── FastAPI test client ────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def async_client(mock_vector_store, mock_embedder, mock_llm) -> AsyncGenerator:
    """Async HTTP client with all external services mocked."""
    from app.main import app
    import app.api.deps as deps

    deps._vector_store = mock_vector_store
    deps._embedder = mock_embedder
    deps._llm_provider = mock_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    # Reset singletons
    deps._vector_store = None
    deps._embedder = None
    deps._llm_provider = None
