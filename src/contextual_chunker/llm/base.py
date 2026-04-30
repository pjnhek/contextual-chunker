from abc import ABC, abstractmethod


class BaseContextLLM(ABC):
    """
    Minimal async LLM interface used by ContextualChunker.

    Implementations only need to provide a single async text-in / text-out call.
    See README for an example of how to plug in your own provider.
    """

    @abstractmethod
    async def generate_simple_async(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Return the model's text response to `prompt`."""
        ...
