"""Application configuration via Pydantic Settings."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "LexoraAI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://lexora:lexora_secret@postgres:5432/lexora"
    DATABASE_SYNC_URL: str = "postgresql+psycopg2://lexora:lexora_secret@postgres:5432/lexora"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "lexora_chunks"

    # ── File storage ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "/uploads"
    MAX_FILE_SIZE_MB: int = 50
    MAX_DOCUMENTS: int = 20

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: Literal["openai", "huggingface"] = "openai"
    EMBEDDING_MODEL_OPENAI: str = "text-embedding-3-small"
    EMBEDDING_MODEL_HF: str = "all-MiniLM-L6-v2"

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["openai", "gemini"] = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 5

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()
