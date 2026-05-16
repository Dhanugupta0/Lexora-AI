"""Unit tests for app/ingestion/embedder.py."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.embedder import HuggingFaceEmbedder, OpenAIEmbedder, get_embedder
from app.config import Settings


# ── OpenAIEmbedder ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openai_embedder_returns_vectors():
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    with patch("openai.AsyncOpenAI") as mock_cls:
        instance = mock_cls.return_value
        instance.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAIEmbedder(api_key="fake", model="text-embedding-3-small")
        embedder._client = instance

        result = await embedder.embed(["hello world"])
        assert result == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_openai_embedder_batches_large_input():
    """Inputs over 100 items should be split into batches (100 + 50)."""
    # Mock returns 50 results per call to match each batch size (first=100, second=50)
    responses = [
        MagicMock(data=[MagicMock(embedding=[0.0] * 10) for _ in range(100)]),
        MagicMock(data=[MagicMock(embedding=[0.0] * 10) for _ in range(50)]),
    ]

    with patch("openai.AsyncOpenAI") as mock_cls:
        instance = mock_cls.return_value
        instance.embeddings.create = AsyncMock(side_effect=responses)

        embedder = OpenAIEmbedder(api_key="fake")
        embedder._client = instance

        texts = [f"text {i}" for i in range(150)]
        result = await embedder.embed(texts)
        # Should have called create twice (100 + 50)
        assert instance.embeddings.create.call_count == 2
        assert len(result) == 150


@pytest.mark.asyncio
async def test_openai_embedder_empty_input():
    embedder = OpenAIEmbedder(api_key="fake")
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector():
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.5, 0.6])]

    with patch("openai.AsyncOpenAI") as mock_cls:
        instance = mock_cls.return_value
        instance.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAIEmbedder(api_key="fake")
        embedder._client = instance

        result = await embedder.embed_query("single question")
        assert result == [0.5, 0.6]


# ── Factory ───────────────────────────────────────────────────────────────────

def test_get_embedder_openai():
    settings = Settings(
        EMBEDDING_PROVIDER="openai",
        OPENAI_API_KEY="sk-test",
    )
    embedder = get_embedder(settings)
    assert isinstance(embedder, OpenAIEmbedder)


def test_get_embedder_missing_key_raises():
    settings = Settings(EMBEDDING_PROVIDER="openai", OPENAI_API_KEY="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_embedder(settings)
