"""Embedding abstraction — OpenAI and HuggingFace backends."""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedder interface."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts, returning a list of float vectors."""
        ...

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (convenience wrapper)."""
        result = await self.embed([text])
        return result[0]


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small (or configurable) embedder."""

    _BATCH_SIZE = 100  # OpenAI max inputs per request

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            response = await self._client.embeddings.create(input=batch, model=self._model)
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings


class HuggingFaceEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder — no API key required."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        logger.info("Loading HuggingFace embedding model: %s", model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, self._model.encode, texts)
        return [v.tolist() for v in vectors]


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Factory — returns the embedder configured in *settings*."""
    settings = settings or get_settings()
    if settings.EMBEDDING_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return OpenAIEmbedder(api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL_OPENAI)
    return HuggingFaceEmbedder(model_name=settings.EMBEDDING_MODEL_HF)
