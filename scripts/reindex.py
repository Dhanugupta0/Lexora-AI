#!/usr/bin/env python3
"""Rebuild the dense index with the currently configured embedding model.

Vectors are only comparable to other vectors from the same model. Switching
embedding models -- bge-small to `jina-embeddings-v4`, say -- leaves ChromaDB
holding vectors from the old space, which either errors on a dimension mismatch
or, worse, silently returns nonsense at the same dimension.

Re-uploading every document would work but wastes the parse and chunk stages.
Chunk text is mirrored into the `chunks` SQL table precisely so it does not have
to: this script reads it back, re-embeds it with whatever `EMBEDDING_PROVIDER`
now points at, and rewrites the collection.

Usage
    PYTHONPATH=. python scripts/reindex.py                # re-embed everything
    PYTHONPATH=. python scripts/reindex.py --dry-run      # just report the plan
    PYTHONPATH=. python scripts/reindex.py --doc <id>     # one document only
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings                                  # noqa: E402
from app.database import Chunk as ChunkRow, SessionLocal, init_db     # noqa: E402
from app.rag.chunker import Chunk, build_context_header               # noqa: E402
from app.rag.embedder import get_embedder                             # noqa: E402
from app.rag.keyword import get_keyword_index                         # noqa: E402
from app.rag.vectorstore import get_vector_store                      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("reindex")
settings = get_settings()


def load_chunks(document_id: str = "") -> Dict[str, List[ChunkRow]]:
    """Every stored chunk, grouped by document and ordered as it was written."""
    db = SessionLocal()
    try:
        query = db.query(ChunkRow)
        if document_id:
            query = query.filter(ChunkRow.document_id == document_id)
        rows = query.order_by(ChunkRow.document_id, ChunkRow.chunk_index).all()
    finally:
        db.close()

    grouped: Dict[str, List[ChunkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.document_id].append(row)
    return grouped


def to_chunk(row: ChunkRow) -> Chunk:
    """SQL row -> the dataclass the vector store writes.

    `embed_text` is rebuilt rather than stored: contextual headers are a
    property of the *current* config, so a config change is picked up here.
    """
    section = row.section or ""
    doc_title = row.doc_title or ""
    page = row.page_number or 0
    header = build_context_header(doc_title, section, page)
    return Chunk(
        text=row.text,
        embed_text=f"{header}\n{row.text}" if settings.CONTEXTUAL_HEADERS else row.text,
        token_count=row.token_count or 0,
        chunk_index=row.chunk_index,
        page_number=page,
        section=section,
        doc_title=doc_title,
        kind=row.kind or "body",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", default="", help="reindex a single document id")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--drop", action="store_true",
                        help="delete the whole Chroma directory first (needed when the "
                             "vector dimension changed)")
    args = parser.parse_args()

    init_db()
    grouped = load_chunks(args.doc)
    total = sum(len(rows) for rows in grouped.values())
    if not total:
        logger.warning("No chunks found in %s -- nothing to reindex.", settings.DATABASE_URL)
        return 1

    embedder = get_embedder()
    logger.info("Target: %s / %s (%d dims)", embedder.provider_name, embedder.model_name,
                embedder.dimension)
    logger.info("Plan: %d chunks across %d document(s)", total, len(grouped))
    if args.dry_run:
        for document_id, rows in grouped.items():
            logger.info("  %s  %-40s %4d chunks", document_id, rows[0].filename or "?", len(rows))
        return 0

    if args.drop and not args.doc:
        # A dimension change cannot be fixed by upserting over the old vectors.
        fingerprint = os.path.join(settings.CHROMA_PATH, ".embedding_fingerprint")
        logger.info("Dropping %s", settings.CHROMA_PATH)
        shutil.rmtree(settings.CHROMA_PATH, ignore_errors=True)
        if os.path.exists(fingerprint):
            os.remove(fingerprint)
        # The cached collection handle now points at a deleted directory.
        import app.rag.vectorstore as vectorstore
        vectorstore._store = None

    store = get_vector_store()
    started = time.time()
    done = 0

    for document_id, rows in grouped.items():
        chunks = [to_chunk(row) for row in rows]
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
        store.upsert(document_id, chunks, vectors, filename=rows[0].filename or "")
        done += len(chunks)
        logger.info("  %s  %4d chunks  (%d/%d)", document_id, len(chunks), done, total)

    get_keyword_index().invalidate()
    logger.info("Reindexed %d chunks in %.1fs. Vectors now in the collection: %d",
                done, time.time() - started, store.count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
