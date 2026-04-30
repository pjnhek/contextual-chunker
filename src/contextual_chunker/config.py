from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ContextualConfig(BaseModel):
    """LLM-driven contextual enrichment knobs."""

    enabled: bool = True
    llm_provider: Literal["gemini", "openai"] = "gemini"
    llm_model: str = "gemini-2.5-flash"
    batch_size: int = 50
    max_context_tokens: int = 100
    max_llm_tokens: int = 800
    temperature: float = 1.0
    timeout_seconds: int = 30
    token_budget: Literal["expand", "reserve"] = "expand"
    concurrency_limit: int = 10
    max_retries: int = 3


class ChunkerConfig(BaseModel):
    """Top-level config consumed by the CLI."""

    chunk_size: int = 512
    chunk_overlap: int = 128
    contextual: ContextualConfig = Field(default_factory=ContextualConfig)

    input_dir: str = "data/in"
    output_path: str = "data/out/chunks.jsonl"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChunkerConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
