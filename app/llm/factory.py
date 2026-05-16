"""LLM provider factory."""
from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.base import BaseLLMProvider


@lru_cache()
def get_llm_provider(settings: Settings | None = None) -> BaseLLMProvider:
    """Return a cached LLM provider instance based on settings."""
    settings = settings or get_settings()
    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from app.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)

    if settings.LLM_PROVIDER == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        from app.llm.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=settings.GEMINI_API_KEY, model=settings.GEMINI_MODEL)

    raise ValueError(f"Unknown LLM_PROVIDER: '{settings.LLM_PROVIDER}'")
