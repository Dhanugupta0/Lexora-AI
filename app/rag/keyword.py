"""Sparse lexical index (BM25).

Dense retrieval is great at paraphrase and terrible at exact tokens -- error
codes, product SKUs, acronyms, surnames, "Section 4.2". BM25 is the opposite.
Running both and fusing them is why the hybrid retriever beats either alone.

The corpus lives in the `chunks` SQL table, so the index survives restarts; it
is rebuilt in memory on first use and invalidated whenever documents change.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# A small stopword list keeps BM25 focused on content words without pulling NLTK.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your",
}
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._/-]*")


def tokenize(text: str) -> List[str]:
    tokens = [t.strip("._-/") for t in _TOKEN.findall(text.lower())]
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


@dataclass
class KeywordHit:
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    section: str
    doc_title: str
    score: float                       # normalised 0..1


class KeywordIndex:
    def __init__(self) -> None:
        self._bm25 = None
        self._ids: List[str] = []
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._dirty = True
        self._lock = threading.RLock()

    def invalidate(self) -> None:
        """Called whenever chunks are added or removed."""
        with self._lock:
            self._dirty = True

    def _rebuild(self) -> None:
        from app.database import Chunk as ChunkRow, SessionLocal

        db = SessionLocal()
        try:
            rows = db.query(ChunkRow).all()
            corpus, ids, meta = [], [], {}
            for row in rows:
                tokens = tokenize(f"{row.doc_title or ''} {row.section or ''} {row.text}")
                if not tokens:
                    continue
                corpus.append(tokens)
                ids.append(row.id)
                meta[row.id] = {
                    "document_id": row.document_id,
                    "chunk_index": row.chunk_index,
                    "page_number": row.page_number or 0,
                    "text": row.text,
                    "section": row.section or "",
                    "doc_title": row.doc_title or "",
                }

            if corpus:
                from rank_bm25 import BM25Okapi
                self._bm25 = BM25Okapi(corpus)
            else:
                self._bm25 = None

            self._ids, self._meta, self._dirty = ids, meta, False
            logger.info("BM25 index rebuilt over %d chunks", len(ids))
        except Exception as exc:                             # noqa: BLE001
            logger.warning("BM25 rebuild failed (%s) -- sparse retrieval disabled", exc)
            self._bm25, self._ids, self._meta, self._dirty = None, [], {}, False
        finally:
            db.close()

    def _ready(self) -> bool:
        with self._lock:
            if self._dirty:
                self._rebuild()
            return self._bm25 is not None

    def search(self, query: str, top_k: int,
               document_ids: Optional[Sequence[str]] = None) -> List[KeywordHit]:
        if not self._ready():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        with self._lock:
            try:
                scores = self._bm25.get_scores(tokens)
            except Exception as exc:                         # noqa: BLE001
                logger.warning("BM25 scoring failed: %s", exc)
                return []
            ids, meta = list(self._ids), self._meta

        allowed = set(document_ids) if document_ids else None
        ranked: List[Tuple[str, float]] = []
        for chunk_id, score in zip(ids, scores):
            if score <= 0:
                continue
            if allowed and meta[chunk_id]["document_id"] not in allowed:
                continue
            ranked.append((chunk_id, float(score)))

        if not ranked:
            return []
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        ranked = ranked[:top_k]

        peak = ranked[0][1] or 1.0
        return [
            KeywordHit(
                chunk_id=chunk_id,
                document_id=meta[chunk_id]["document_id"],
                chunk_index=meta[chunk_id]["chunk_index"],
                page_number=meta[chunk_id]["page_number"],
                text=meta[chunk_id]["text"],
                section=meta[chunk_id]["section"],
                doc_title=meta[chunk_id]["doc_title"],
                score=round(score / peak, 4),
            )
            for chunk_id, score in ranked
        ]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"backend": "bm25-okapi", "documents_indexed": len(self._ids),
                    "stale": self._dirty, "healthy": self._bm25 is not None or not self._ids}


_index: Optional[KeywordIndex] = None
_index_lock = threading.Lock()


def get_keyword_index() -> KeywordIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = KeywordIndex()
    return _index
