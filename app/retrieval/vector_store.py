"""ChromaDB vector store wrapper."""
import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chroma_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float  # cosine distance (lower = more similar)
    page_number: Optional[int]


class VectorStore:
    """Thin async wrapper around a ChromaDB HTTP collection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any = None
        self._collection: Any = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is None:
            import chromadb

            self._client = chromadb.HttpClient(
                host=self._settings.CHROMA_HOST,
                port=self._settings.CHROMA_PORT,
            )
        return self._client

    def _get_collection(self) -> Any:
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self._settings.CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def _run(self, fn, *args, **kwargs):
        """Run a synchronous ChromaDB call in a thread-pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ── Write operations ──────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        document_id: str,
        chunk_texts: List[str],
        embeddings: List[List[float]],
        chunk_indices: List[int],
        page_numbers: List[Optional[int]],
    ) -> List[str]:
        """Insert or update chunk vectors. Returns the generated chroma_ids."""
        collection = self._get_collection()
        chroma_ids = [f"{document_id}_{idx}" for idx in chunk_indices]
        metadatas = [
            {
                "document_id": document_id,
                "chunk_index": ci,
                "page_number": pn if pn is not None else -1,
            }
            for ci, pn in zip(chunk_indices, page_numbers)
        ]
        await self._run(
            collection.upsert,
            ids=chroma_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
        )
        return chroma_ids

    async def delete_by_document_id(self, document_id: str) -> None:
        """Remove all vectors belonging to *document_id*."""
        collection = self._get_collection()
        await self._run(
            collection.delete,
            where={"document_id": document_id},
        )

    # ── Read operations ───────────────────────────────────────────────────────

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Return the *top_k* most similar chunks, optionally scoped to *document_ids*."""
        collection = self._get_collection()
        where: Optional[Dict] = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": document_ids}}
            )

        raw = await self._run(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=min(top_k, await self._run(collection.count) or top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: List[SearchResult] = []
        for chroma_id, doc, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(
                SearchResult(
                    chroma_id=chroma_id,
                    document_id=meta.get("document_id", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    text=doc,
                    score=dist,
                    page_number=meta.get("page_number") if meta.get("page_number", -1) != -1 else None,
                )
            )
        return results

    async def collection_count(self) -> int:
        collection = self._get_collection()
        return await self._run(collection.count)

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            await self._run(client.heartbeat)
            return True
        except Exception:
            return False
