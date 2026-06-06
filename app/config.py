from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    APP_NAME: str = "LexoraAI"
    APP_VERSION: str = "1.0.0"

    DATABASE_URL: str = "sqlite:///./local_data/lexora.db"

    CHROMA_PATH: str = "./local_data/chroma"
    CHROMA_COLLECTION: str = "lexora_chunks"

    UPLOAD_DIR: str = "./local_data/uploads"
    MAX_FILE_SIZE_MB: int = 50
    MAX_DOCUMENTS: int = 20

    EMBEDDING_MODEL: str = "text-embedding-3-small"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50
    DEFAULT_TOP_K: int = 5

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()
