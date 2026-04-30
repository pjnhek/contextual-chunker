import logging
import os
from typing import Optional

from google import genai
from google.genai.types import GenerateContentConfig

from contextual_chunker.llm.base import BaseContextLLM


class GeminiContextLLM(BaseContextLLM):
    """
    Gemini adapter using google-genai's native async interface.

    Set GOOGLE_API_KEY in your environment, or pass api_key directly.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing. Set it in .env or pass api_key= explicitly."
            )
        self.model_name = model_name
        self.client = genai.Client(api_key=key)
        self.logger = logging.getLogger(__name__)

    async def generate_simple_async(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None)
        if text is None:
            # Surface why the response was empty (safety filter, finish reason, etc.)
            candidates = getattr(response, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            safety = getattr(candidates[0], "safety_ratings", None) if candidates else None
            self.logger.warning(
                f"Gemini returned None text. finish_reason={finish_reason}, "
                f"safety_ratings={safety}"
            )

        return (text or "").strip()
