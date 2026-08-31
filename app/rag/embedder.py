"""Text -> vectors.

Groq does not serve an embeddings API, so embeddings come from **Jina AI**
(`jina-embeddings-v4`, hosted) with local ONNX models kept as offline fallbacks.
Providers are tried in order so a missing key or dependency degrades instead of
crashing:

    jina (jina-embeddings-v4)  ->  fastembed (bge-small)  ->  chroma MiniLM  ->  openai

Two details matter more than the provider choice:

**Asymmetry.** A question and the passage that answers it are worded
differently, so they should not be embedded the same way. Every call therefore
carries a *role*:

    query       the user's question, or a HyDE probe        (retrieval.query)
    passage     an indexed chunk                            (retrieval.passage)
    similarity  two texts compared to each other, as in     (text-matching)
                grounding's claim-vs-evidence check

Jina maps the role onto its typed `task` parameter; bge maps it onto its
instruction prefix; the ONNX fallbacks ignore it. The role is part of the cache
key, so the same sentence embedded as a query and as a passage never collides.

**Caching.** A content-addressed LRU sits in front of every provider, so
repeated text -- re-uploads, HyDE probes, MMR passes, grounding claims -- never
pays for inference (or an API call) twice.
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

Vector = List[float]

# The three ways a piece of text can be used. See the module docstring.
ROLES = ("query", "passage", "similarity")


# --------------------------------------------------------------------------- #
# Providers
#
# Each exposes: name, model_name, dimension, encode(texts, role) -> vectors.
# --------------------------------------------------------------------------- #
class _JinaProvider:
    """Hosted embeddings from https://api.jina.ai/v1/embeddings.

    `jina-embeddings-v4` is multilingual, multimodal and Matryoshka-trained: the
    2048-dim vector can be truncated to any shorter prefix and stays valid, so
    `JINA_EMBED_DIMENSIONS` trades index size against a little accuracy without
    changing models.
    """

    name = "jina"

    # role -> the model's typed task. Getting this wrong silently costs recall.
    _TASKS = {
        "query": "retrieval.query",
        "passage": "retrieval.passage",
        "similarity": "text-matching",
    }

    def __init__(self, model_name: str):
        import httpx

        if not settings.JINA_API_KEY:
            raise RuntimeError("JINA_API_KEY is not set")

        # EMBEDDING_MODEL wins when it names a Jina model; otherwise fall back
        # to JINA_EMBED_MODEL, so `EMBEDDING_PROVIDER=jina` alone is enough.
        self.model_name = model_name if model_name.startswith("jina-") else settings.JINA_EMBED_MODEL
        self._dimensions = settings.JINA_EMBED_DIMENSIONS if settings.JINA_EMBED_DIMENSIONS > 0 else None
        self._client = httpx.Client(
            timeout=settings.JINA_TIMEOUT_SECONDS,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        # Doubles as a credential check: a bad key fails here, at start-up,
        # rather than halfway through someone's first upload.
        self.dimension = len(self.encode(["dimension probe"], "passage")[0])

    def _payload(self, texts: Sequence[str], role: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "task": self._TASKS.get(role, "retrieval.passage"),
            # v4 is multimodal, so inputs are typed objects rather than strings.
            "input": [{"text": text} for text in texts],
            "truncate": True,          # over-long input is clipped, not rejected
        }
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        return payload

    def encode(self, texts: Sequence[str], role: str = "passage") -> List[Vector]:
        import httpx

        payload = self._payload(texts, role)
        last_error: Optional[Exception] = None

        for attempt in range(1, max(1, settings.JINA_MAX_RETRIES) + 1):
            try:
                response = self._client.post(settings.JINA_API_URL, json=payload)
                # 429 and 5xx are transient; 4xx otherwise is our bug, so stop.
                if response.status_code == 429 or response.status_code >= 500:
                    raise _Retryable(f"HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                data = response.json().get("data") or []
                if len(data) != len(texts):
                    raise _Retryable(f"expected {len(texts)} embeddings, got {len(data)}")
                # The API may return results out of order; `index` is authoritative.
                ordered = sorted(data, key=lambda item: item.get("index", 0))
                return [[float(x) for x in item["embedding"]] for item in ordered]
            except (_Retryable, httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= max(1, settings.JINA_MAX_RETRIES):
                    break
                delay = settings.JINA_BACKOFF_BASE * (2 ** (attempt - 1))
                delay += random.uniform(0, delay * 0.25)      # jitter: avoid lockstep retries
                logger.warning("Jina embeddings attempt %d failed (%s) -- retrying in %.1fs",
                               attempt, exc, delay)
                time.sleep(delay)

        raise RuntimeError(f"Jina embeddings failed after {settings.JINA_MAX_RETRIES} attempts: {last_error}")


class _Retryable(RuntimeError):
    """A Jina failure worth trying again (rate limit, 5xx, truncated response)."""


class _FastEmbedProvider:
    """Offline fallback: quantised ONNX bge-small, ~130 MB, CPU-only, no PyTorch."""

    name = "fastembed"

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        # A Jina model name means nothing to fastembed -- use its own default.
        self.model_name = model_name if not model_name.startswith("jina-") else "BAAI/bge-small-en-v1.5"
        self._model = TextEmbedding(model_name=self.model_name)
        self.dimension = len(next(iter(self._model.embed(["dimension probe"]))))

    def encode(self, texts: Sequence[str], role: str = "passage") -> List[Vector]:
        # bge wants its instruction prefix on the query side only.
        if role == "query" and settings.EMBEDDING_QUERY_PREFIX:
            texts = [f"{settings.EMBEDDING_QUERY_PREFIX}{text}" for text in texts]
        return [vec.tolist() for vec in self._model.embed(list(texts))]


class _ChromaOnnxProvider:
    """Last-resort fallback: the all-MiniLM-L6-v2 ONNX model bundled with chromadb."""

    name = "chroma-minilm"
    model_name = "all-MiniLM-L6-v2"

    def __init__(self, _model_name: str):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._fn = ONNXMiniLM_L6_V2()
        self.dimension = len(self._fn(["dimension probe"])[0])

    def encode(self, texts: Sequence[str], role: str = "passage") -> List[Vector]:
        return [list(map(float, vec)) for vec in self._fn(list(texts))]


class _OpenAIProvider:
    name = "openai"

    def __init__(self, model_name: str):
        from openai import OpenAI
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model_name = model_name if model_name.startswith("text-embedding") else "text-embedding-3-small"
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.dimension = len(self.encode(["dimension probe"])[0])

    def encode(self, texts: Sequence[str], role: str = "passage") -> List[Vector]:
        response = self._client.embeddings.create(model=self.model_name, input=list(texts))
        return [item.embedding for item in response.data]


_PROVIDERS = {
    "jina": _JinaProvider,
    "fastembed": _FastEmbedProvider,
    "chroma": _ChromaOnnxProvider,
    "openai": _OpenAIProvider,
}
_FALLBACK_ORDER = ("jina", "fastembed", "chroma", "openai")


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #
class Embedder:
    def __init__(self) -> None:
        self._provider: Any = None
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, Vector]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "batches": 0}

    # -- lazy provider chain ------------------------------------------------ #
    def _load(self):
        if self._provider is not None:
            return self._provider
        with self._lock:
            if self._provider is not None:
                return self._provider

            preferred = settings.EMBEDDING_PROVIDER.lower()
            order = [preferred] + [p for p in _FALLBACK_ORDER if p != preferred]
            errors = []
            for key in order:
                factory = _PROVIDERS.get(key)
                if factory is None:
                    continue
                try:
                    provider = factory(settings.EMBEDDING_MODEL)
                    logger.info(
                        "Embedder ready: %s (%s, %d dims)",
                        provider.name, getattr(provider, "model_name", "?"), provider.dimension,
                    )
                    self._provider = provider
                    return provider
                except Exception as exc:                        # noqa: BLE001
                    errors.append(f"{key}: {exc}")
                    logger.warning("Embedding provider '%s' unavailable -- %s", key, exc)
            raise RuntimeError("No embedding provider could be loaded. Tried -> " + " | ".join(errors))

    @property
    def dimension(self) -> int:
        return self._load().dimension

    @property
    def provider_name(self) -> str:
        return self._load().name

    @property
    def model_name(self) -> str:
        return getattr(self._load(), "model_name", "?")

    # -- cache -------------------------------------------------------------- #
    def _key(self, text: str, role: str) -> str:
        """Role and model are part of the key: the same sentence embedded as a
        query and as a passage is two different vectors."""
        seed = f"{self.provider_name}|{self.model_name}|{role}|{text}"
        return hashlib.sha1(seed.encode("utf-8", "ignore")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Vector]:
        with self._cache_lock:
            vector = self._cache.get(key)
            if vector is not None:
                self._cache.move_to_end(key)
                self._stats["hits"] += 1
            return vector

    def _cache_put(self, key: str, vector: Vector) -> None:
        with self._cache_lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > settings.EMBEDDING_CACHE_SIZE:
                self._cache.popitem(last=False)

    # -- public API --------------------------------------------------------- #
    def embed(self, texts: Sequence[str], role: str = "passage") -> List[Vector]:
        """Embed a batch in one role, using the cache and honouring the batch size."""
        if not texts:
            return []
        if role not in ROLES:
            role = "passage"

        provider = self._load()
        results: List[Optional[Vector]] = [None] * len(texts)
        pending: List[int] = []

        for i, text in enumerate(texts):
            cached = self._cache_get(self._key(text, role))
            if cached is not None:
                results[i] = cached
            else:
                pending.append(i)
                self._stats["misses"] += 1

        size = max(1, settings.EMBEDDING_BATCH_SIZE)
        for start in range(0, len(pending), size):
            window = pending[start : start + size]
            batch = [texts[i] for i in window]
            vectors = self._encode_resilient(provider, batch, role)
            self._stats["batches"] += 1
            for i, vector in zip(window, vectors):
                results[i] = vector
                self._cache_put(self._key(texts[i], role), vector)

        return [vector if vector is not None else [0.0] * provider.dimension for vector in results]

    def _encode_resilient(self, provider: Any, batch: List[str], role: str) -> List[Vector]:
        """Bisect on failure so a single bad input can't sink a whole batch."""
        try:
            return provider.encode(batch, role)
        except Exception as exc:                                # noqa: BLE001
            if len(batch) == 1:
                logger.error("Embedding failed for a single input (%s) -- using zero vector", exc)
                return [[0.0] * provider.dimension]
            logger.warning("Embedding batch of %d failed (%s) -- splitting", len(batch), exc)
            mid = len(batch) // 2
            return (self._encode_resilient(provider, batch[:mid], role)
                    + self._encode_resilient(provider, batch[mid:], role))

    def embed_query(self, query: str) -> Vector:
        return self.embed([query], "query")[0]

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        return self.embed(texts, "passage")

    def embed_similarity(self, texts: Sequence[str]) -> List[Vector]:
        """Symmetric comparison -- both sides are the same kind of text."""
        return self.embed(texts, "similarity")

    def snapshot(self) -> Dict[str, Any]:
        try:
            provider = self._load()
            info = {"provider": provider.name, "model": getattr(provider, "model_name", "?"),
                    "dimension": provider.dimension, "loaded": True}
        except Exception as exc:                                # noqa: BLE001
            info = {"provider": settings.EMBEDDING_PROVIDER, "loaded": False, "error": str(exc)}
        info["cache"] = {**self._stats, "size": len(self._cache)}
        return info


_embedder: Optional[Embedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder
