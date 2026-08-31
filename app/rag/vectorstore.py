"""Dense vector index (ChromaDB).

Thin, defensive wrapper so the rest of the pipeline never imports chromadb
directly and never has to care which chroma major version is installed.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag.chunker import Chunk

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class VectorHit:
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    section: str
    doc_title: str
    score: float                       # cosine similarity, 0..1


class VectorStore:
    def __init__(self) -> None:
        self._collection = None
        self._lock = threading.Lock()
        self._fingerprint_checked = False

    def _get(self):
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    import chromadb
                    os.makedirs(settings.CHROMA_PATH, exist_ok=True)
                    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
                    try:
                        self._collection = client.get_or_create_collection(
                            name=settings.CHROMA_COLLECTION,
                            metadata={"hnsw:space": "cosine"},
                        )
                    except Exception:                        # noqa: BLE001 -- chroma 1.x config shape
                        self._collection = client.get_or_create_collection(
                            name=settings.CHROMA_COLLECTION,
                            configuration={"hnsw": {"space": "cosine"}},
                        )
                    logger.info("ChromaDB ready at %s", settings.CHROMA_PATH)
        return self._collection

    # -- writes ------------------------------------------------------------- #
    def upsert(self, document_id: str, chunks: Sequence[Chunk],
               embeddings: Sequence[Sequence[float]], filename: str = "") -> List[str]:
        if not chunks:
            return []
        collection = self._get()
        self._check_fingerprint(len(embeddings[0]) if embeddings else 0)
        ids = [f"{document_id}::{c.chunk_index}" for c in chunks]
        collection.upsert(
            ids=ids,
            embeddings=[list(map(float, e)) for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[{
                "document_id": document_id,
                "filename": filename,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "section": c.section or "",
                "doc_title": c.doc_title or "",
                "kind": c.kind,
                "token_count": c.token_count,
            } for c in chunks],
        )
        return ids

    def _check_fingerprint(self, dimension: int) -> None:
        """Warn when the collection was built with a different embedding model.

        Two models can share a vector width while placing text in completely
        different spaces, so a silent swap degrades retrieval without any error.
        The fingerprint is written on first write and compared on every later one.
        """
        if not dimension or self._fingerprint_checked:
            return
        self._fingerprint_checked = True

        fingerprint = f"{settings.EMBEDDING_PROVIDER}:{settings.EMBEDDING_MODEL}:{dimension}"
        path = os.path.join(settings.CHROMA_PATH, ".embedding_fingerprint")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    stored = handle.read().strip()
                if stored and stored != fingerprint:
                    logger.error(
                        "Embedding model changed (%s -> %s). Existing vectors were built with the "
                        "old model and will retrieve poorly -- or fail outright if the dimension "
                        "also changed. Re-embed from the stored chunk text with "
                        "`PYTHONPATH=. python scripts/reindex.py --drop` (no re-upload needed), "
                        "or delete %s.",
                        stored, fingerprint, settings.CHROMA_PATH,
                    )
                    return
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(fingerprint)
        except OSError as exc:
            logger.debug("Could not record the embedding fingerprint: %s", exc)

    def delete_document(self, document_id: str) -> None:
        self._get().delete(where={"document_id": document_id})

    # -- reads -------------------------------------------------------------- #
    def search(self, embedding: Sequence[float], top_k: int,
               document_ids: Optional[Sequence[str]] = None) -> List[VectorHit]:
        collection = self._get()
        total = collection.count()
        if total == 0:
            return []

        raw = collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=max(1, min(top_k, total)),
            where=_where(document_ids),
            include=["documents", "metadatas", "distances"],
        )
        if not raw.get("ids") or not raw["ids"][0]:
            return []

        hits: List[VectorHit] = []
        for cid, text, meta, distance in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            meta = meta or {}
            hits.append(VectorHit(
                chunk_id=cid,
                document_id=str(meta.get("document_id", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
                page_number=int(meta.get("page_number", 0)),
                text=text or "",
                section=str(meta.get("section", "")),
                doc_title=str(meta.get("doc_title", "")),
                # chroma returns cosine *distance* in [0, 2]
                score=round(max(0.0, 1.0 - float(distance)), 4),
            ))
        return hits

    def get_embeddings(self, chunk_ids: Sequence[str]) -> Dict[str, List[float]]:
        """Fetch stored vectors -- used by MMR so we never re-embed candidates."""
        if not chunk_ids:
            return {}
        try:
            raw = self._get().get(ids=list(chunk_ids), include=["embeddings"])
        except Exception as exc:                             # noqa: BLE001
            logger.warning("Could not fetch stored embeddings: %s", exc)
            return {}
        vectors = raw.get("embeddings")
        if vectors is None:
            return {}
        return {cid: list(map(float, vec)) for cid, vec in zip(raw.get("ids", []), vectors) if vec is not None}

    def count(self) -> int:
        try:
            return self._get().count()
        except Exception:                                    # noqa: BLE001
            return 0

    def snapshot(self) -> Dict[str, Any]:
        try:
            return {"backend": "chromadb", "path": settings.CHROMA_PATH,
                    "collection": settings.CHROMA_COLLECTION, "vectors": self.count(), "healthy": True}
        except Exception as exc:                             # noqa: BLE001
            return {"backend": "chromadb", "healthy": False, "error": str(exc)}


def _where(document_ids: Optional[Sequence[str]]):
    if not document_ids:
        return None
    ids = list(document_ids)
    return {"document_id": ids[0]} if len(ids) == 1 else {"document_id": {"$in": ids}}


_store: Optional[VectorStore] = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store
