"""Central configuration. Every knob in the RAG pipeline is env-driven."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # ---------------------------------------------------------------- app ---
    APP_NAME: str = "LexoraAI"
    APP_VERSION: str = "2.0.0"
    LOG_LEVEL: str = "INFO"

    # ----------------------------------------------------------- storage ---
    DATABASE_URL: str = "sqlite:///./local_data/lexora.db"
    CHROMA_PATH: str = "./local_data/chroma"
    CHROMA_COLLECTION: str = "lexora_chunks"
    UPLOAD_DIR: str = "./local_data/uploads"
    MAX_FILE_SIZE_MB: int = 50
    MAX_DOCUMENTS: int = 20

    # --------------------------------------------------------------- llm ---
    # Groq only serves chat completions -- embeddings come from Jina (see below).
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    # Comma separated. Tried in order when the primary model fails.
    GROQ_FALLBACK_MODELS: str = "openai/gpt-oss-120b,qwen/qwen3.6-27b"
    LLM_TEMPERATURE: float = 0.15
    LLM_MAX_TOKENS: int = 1800
    LLM_REASONING_EFFORT: str = "medium"   # low | medium | high  (gpt-oss)
    LLM_TIMEOUT_SECONDS: float = 45.0
    LLM_MAX_RETRIES: int = 3
    LLM_BACKOFF_BASE: float = 0.6
    CIRCUIT_FAIL_THRESHOLD: int = 5
    CIRCUIT_COOLDOWN_SECONDS: float = 45.0

    # -------------------------------------------------------- embeddings ---
    EMBEDDING_PROVIDER: str = "jina"               # jina | fastembed | chroma | openai
    EMBEDDING_MODEL: str = "jina-embeddings-v4"
    EMBEDDING_BATCH_SIZE: int = 32                 # Jina is a network call: smaller batches
    EMBEDDING_CACHE_SIZE: int = 8192
    # bge-family models want an instruction prefix on the *query* side only.
    # Jina uses typed `task` values instead, so this is ignored there.
    EMBEDDING_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
    OPENAI_API_KEY: str = ""                        # optional legacy provider

    # -- Jina AI (hosted embeddings) --------------------------------------- #
    # Get a free key at https://jina.ai/embeddings
    JINA_API_KEY: str = ""
    JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"
    JINA_EMBED_MODEL: str = "jina-embeddings-v4"
    # v4 is a Matryoshka model: any prefix of the 2048-dim vector is a valid
    # embedding. 1024 keeps ~99% of retrieval quality at half the index size.
    JINA_EMBED_DIMENSIONS: int = 1024
    JINA_TIMEOUT_SECONDS: float = 60.0
    JINA_MAX_RETRIES: int = 3
    JINA_BACKOFF_BASE: float = 0.8

    # ---------------------------------------------------------- chunking ---
    CHUNK_STRATEGY: str = "semantic"               # semantic | recursive
    CHUNK_SIZE_TOKENS: int = 420
    CHUNK_OVERLAP_TOKENS: int = 64
    CHUNK_MIN_TOKENS: int = 48
    SEMANTIC_BREAKPOINT_PERCENTILE: int = 88
    CONTEXTUAL_HEADERS: bool = True                # prepend doc/section to embedded text

    # --------------------------------------------------------- retrieval ---
    RETRIEVAL_MODE: str = "hybrid"                 # hybrid | dense | sparse
    DEFAULT_TOP_K: int = 6
    CANDIDATE_POOL: int = 40                       # per-retriever candidate depth
    RRF_K: int = 60                                # reciprocal-rank-fusion constant
    MMR_ENABLED: bool = True
    MMR_LAMBDA: float = 0.65                       # 1.0 = pure relevance, 0 = pure diversity
    MIN_RELEVANCE: float = 0.15                    # drop candidates below this fused score

    # ---------------------------------------------------------- reranker ---
    RERANKER_MODE: str = "llm"                     # llm | cross_encoder | heuristic | off
    RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    RERANK_INPUT_SIZE: int = 20                    # candidates fed into the reranker

    # --------------------------------------------------------- reasoning ---
    ENABLE_QUERY_PLANNING: bool = True             # rewrite + decompose + route
    ENABLE_HYDE: bool = True                       # hypothetical-document embeddings
    MAX_SUB_QUERIES: int = 3
    MAX_HISTORY_TURNS: int = 6

    # --------------------------------------------------------- grounding ---
    ENABLE_GROUNDING_CHECK: bool = True
    GROUNDING_MIN_SUPPORT: float = 0.42            # per-claim support threshold
    ABSTAIN_THRESHOLD: float = 0.25                # answer-level confidence floor
    STRIP_INVALID_CITATIONS: bool = True

    # ---------------------------------------------------------- helpers ---
    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def model_chain(self) -> List[str]:
        """Primary model first, then each configured fallback (deduped)."""
        chain = [self.GROQ_MODEL]
        for name in self.GROQ_FALLBACK_MODELS.split(","):
            name = name.strip()
            if name and name not in chain:
                chain.append(name)
        return chain


@lru_cache()
def get_settings() -> Settings:
    return Settings()
