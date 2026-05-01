import pytest

from contextual_chunker.config import ChunkerConfig, ContextualConfig


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("max_context_tokens", 0),
        ("max_llm_tokens", 0),
        ("timeout_seconds", 0),
        ("concurrency_limit", 0),
        ("max_retries", 0),
    ],
)
def test_contextual_config_requires_positive_counts(field, value):
    with pytest.raises(ValueError):
        ContextualConfig(**{field: value})


def test_contextual_config_rejects_negative_temperature():
    with pytest.raises(ValueError):
        ContextualConfig(temperature=-0.1)


def test_chunker_config_validates_chunk_sizes():
    with pytest.raises(ValueError):
        ChunkerConfig(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkerConfig(chunk_size=64, chunk_overlap=64)

    with pytest.raises(ValueError):
        ChunkerConfig(chunk_overlap=-1)
