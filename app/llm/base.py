"""Abstract LLM provider interface."""
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """All LLM providers implement this interface."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the model and return the text response."""
        ...
