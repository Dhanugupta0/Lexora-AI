"""Relational metadata store.

`documents` tracks ingestion state. `chunks` mirrors every chunk's text so the
BM25 sparse index can be rebuilt from durable storage on restart -- a vector
store alone cannot serve lexical search.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Column, DateTime, Enum, Float, ForeignKey, Index, Integer,
    String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class Document(Base):
    __tablename__ = "documents"

    id              = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename        = Column(String(255), nullable=False)
    title           = Column(String(512), nullable=True)
    file_type       = Column(String(10),  nullable=False)
    file_size       = Column(BigInteger,  nullable=False)
    page_count      = Column(Integer,     nullable=True)
    chunk_count     = Column(Integer,     nullable=True)
    token_count     = Column(Integer,     nullable=True)
    status          = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    stage           = Column(String(32),  nullable=True)     # parsing | chunking | embedding | indexing
    error_message   = Column(Text,        nullable=True)
    ingest_seconds  = Column(Float,       nullable=True)
    upload_time     = Column(DateTime,    default=utcnow, nullable=False)
    processed_time  = Column(DateTime,    nullable=True)


class Chunk(Base):
    """Durable copy of chunk text -- powers BM25 and source snippets."""

    __tablename__ = "chunks"

    id           = Column(String(80), primary_key=True)          # "<document_id>::<index>"
    document_id  = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    chunk_index  = Column(Integer, nullable=False)
    page_number  = Column(Integer, nullable=False, default=0)
    section      = Column(String(512), nullable=True)
    doc_title    = Column(String(512), nullable=True)
    filename     = Column(String(255), nullable=True)
    kind         = Column(String(16), nullable=True, default="body")
    token_count  = Column(Integer, nullable=True)
    text         = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=utcnow, nullable=False)


Index("ix_chunks_document_index", Chunk.document_id, Chunk.chunk_index)


_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=not _is_sqlite,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
