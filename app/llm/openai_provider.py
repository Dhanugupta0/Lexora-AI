"""OpenAI chat-completions LLM provider."""
import logging

from openai import AsyncOpenAI

from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """Wraps OpenAI's chat completions API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        logger.debug("Calling OpenAI model=%s", self._model)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
