import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger, Column, DateTime, Enum, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()


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
    file_type       = Column(String(10),  nullable=False)
    file_size       = Column(BigInteger,  nullable=False)
    page_count      = Column(Integer,     nullable=True)
    chunk_count     = Column(Integer,     nullable=True)
    status          = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    error_message   = Column(Text,        nullable=True)
    upload_time     = Column(DateTime,    default=datetime.utcnow, nullable=False)
    processed_time  = Column(DateTime,    nullable=True)


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
