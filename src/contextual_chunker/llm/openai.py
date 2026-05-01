import logging
import os
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from contextual_chunker.llm.base import BaseContextLLM


class OpenAIContextLLM(BaseContextLLM):
    """
    OpenAI adapter using the async Responses API.

    Defaults to public OpenAI with OPENAI_API_KEY. If AZURE_OPENAI_API_KEY and
    AZURE_OPENAI_ENDPOINT are set in the environment, transparently uses Azure
    OpenAI instead — `model_name` should be your Azure deployment name in that case.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if azure_key and azure_endpoint:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
            self.client = AsyncAzureOpenAI(
                api_key=azure_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
            self.logger.info(f"Using Azure OpenAI (endpoint={azure_endpoint})")
            return

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Set it in .env, pass api_key= "
                "explicitly, or set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT "
                "to use Azure OpenAI."
            )
        self.client = AsyncOpenAI(api_key=key)

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
