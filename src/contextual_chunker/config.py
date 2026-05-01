from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ContextualConfig(BaseModel):
    """LLM-driven contextual enrichment knobs."""

    enabled: bool = True
    llm_provider: Literal["gemini", "openai"] = "gemini"
    llm_model: str = "gemini-2.5-flash"
    batch_size: int = Field(default=50, ge=1)
    max_context_tokens: int = Field(default=100, ge=1)
    max_llm_tokens: int = Field(default=800, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    timeout_seconds: int = Field(default=30, ge=1)
    token_budget: Literal["expand", "reserve"] = "expand"
    concurrency_limit: int = Field(default=10, ge=1)
    max_retries: int = Field(default=3, ge=1)


class ChunkerConfig(BaseModel):
    """Top-level config consumed by the CLI."""

    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=128, ge=0)
    contextual: ContextualConfig = Field(default_factory=ContextualConfig)

    input_dir: str = "data/in"
    output_path: str = "data/out/chunks.jsonl"

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "ChunkerConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChunkerConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
