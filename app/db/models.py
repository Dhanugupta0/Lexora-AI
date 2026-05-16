"""ORM models for Document and Chunk."""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .base import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    """Metadata record for every uploaded document."""

    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=_uuid)
    filename = Column(String(255), nullable=False)           # sanitised stored name
    original_filename = Column(String(255), nullable=False)  # as supplied by user
    file_type = Column(String(10), nullable=False)           # pdf | docx | txt
    file_size = Column(BigInteger, nullable=False)           # bytes
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    status = Column(
        Enum(DocumentStatus),
        default=DocumentStatus.PENDING,
        nullable=False,
    )
    error_message = Column(Text, nullable=True)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_time = Column(DateTime, nullable=True)

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "error_message": self.error_message,
            "upload_time": self.upload_time.isoformat() if self.upload_time else None,
            "processed_time": self.processed_time.isoformat() if self.processed_time else None,
        }


class Chunk(Base):
    """Individual text chunk derived from a document."""

    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True, default=_uuid)
    document_id = Column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    chroma_id = Column(String(100), nullable=False, unique=True)
    page_number = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="chunks")
