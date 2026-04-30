import logging
import os
from typing import Optional

from openai import AsyncOpenAI

from contextual_chunker.llm.base import BaseContextLLM


class OpenAIContextLLM(BaseContextLLM):
    """
    OpenAI adapter using the async Responses API.

    Set OPENAI_API_KEY in your environment, or pass api_key directly.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Set it in .env or pass api_key= explicitly."
            )
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=key)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _extract_output_text(resp) -> str:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        t = getattr(c, "text", "") or ""
                        if t.strip():
                            parts.append(t.strip())
        return "\n".join(parts).strip()

    async def generate_simple_async(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        req = {
            "model": self.model_name,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        # Reasoning models reject arbitrary temperatures; only send if not 1.0.
        if temperature is not None and temperature != 1.0:
            req["temperature"] = temperature

        try:
            resp = await self.client.responses.create(**req)
        except TypeError as e:
            self.logger.warning(f"Retrying OpenAI request with minimal args: {e}")
            resp = await self.client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=max_tokens,
            )

        return self._extract_output_text(resp)
