"""Google Gemini LLM provider."""
import asyncio
import logging

from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Wraps Google Generative AI (Gemini) API."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model_name = model
        self._genai = genai

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        logger.debug("Calling Gemini model=%s", self._model_name)
        combined = f"{system_prompt}\n\n{user_prompt}"
        loop = asyncio.get_event_loop()
        model = self._genai.GenerativeModel(self._model_name)
        response = await loop.run_in_executor(None, model.generate_content, combined)
        return response.text or ""
