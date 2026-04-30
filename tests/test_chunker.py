import pytest

from contextual_chunker import (
    BaseContextLLM,
    ContextualChunker,
    TokenTextSplitter,
    compute_base_chunk_size,
)
from contextual_chunker.prompts import CONTEXT_SEPARATOR


class MockContextLLM(BaseContextLLM):
    """Returns a predictable context. Optionally fails the first N calls to test retry."""

    def __init__(self, fail_first_n: int = 0):
        self.calls = 0
        self.fail_first_n = fail_first_n

    async def generate_simple_async(self, prompt, max_tokens, temperature):
        self.calls += 1
        if self.calls <= self.fail_first_n:
            raise RuntimeError(f"simulated failure on call {self.calls}")
        # Echo the first 30 chars of the chunk so we can verify enrichment ran per-chunk.
        chunk_marker = prompt.split("CHUNK from the document:")[1][:60].strip()
        return f"MOCK CONTEXT for [{chunk_marker[:30]}]"


def _docs(*texts):
    return [{"document": t} for t in texts]


def test_enriched_chunk_format():
    splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=2)
    chunker = ContextualChunker(
        base_chunker=splitter,
        llm_generator=MockContextLLM(),
        concurrency_limit=4,
        max_context_tokens=100,
    )

    text = "Alpha beta gamma delta. " * 30
    _, enriched, counts = chunker.split_documents(_docs(text), "document")

    assert sum(counts) == len(enriched)
    assert len(enriched) > 1
    for chunk in enriched:
        assert chunk.startswith("MOCK CONTEXT for ")
        assert CONTEXT_SEPARATOR in chunk


def test_get_chunk_contexts_returns_components():
    splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=2)
    chunker = ContextualChunker(
        base_chunker=splitter,
        llm_generator=MockContextLLM(),
        concurrency_limit=2,
    )

    text = "lorem ipsum dolor sit amet " * 20
    _, enriched, originals, contexts, _ = chunker.get_chunk_contexts(_docs(text), "document")

    assert len(enriched) == len(originals) == len(contexts)
    for enr, orig, ctx in zip(enriched, originals, contexts):
        assert ctx is not None
        assert enr == f"{ctx}{CONTEXT_SEPARATOR}{orig}"


def test_retry_then_succeed():
    """One simulated failure followed by success — chunk should still get enriched."""
    splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=5)
    llm = MockContextLLM(fail_first_n=1)
    chunker = ContextualChunker(
        base_chunker=splitter,
        llm_generator=llm,
        concurrency_limit=1,
        max_retries=3,
    )

    _, enriched, _ = chunker.split_documents(_docs("hello world. " * 5), "document")

    assert len(enriched) == 1
    assert enriched[0].startswith("MOCK CONTEXT for ")
    assert llm.calls == 2  # one failure + one success


def test_retry_exhausted_falls_back_to_raw_chunk():
    """All retries fail — chunk is kept unenriched, not dropped."""
    splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=5)
    llm = MockContextLLM(fail_first_n=999)
    chunker = ContextualChunker(
        base_chunker=splitter,
        llm_generator=llm,
        concurrency_limit=1,
        max_retries=2,
    )

    raw_text = "hello world. " * 5
    _, enriched, _ = chunker.split_documents(_docs(raw_text), "document")

    assert len(enriched) == 1
    assert "MOCK CONTEXT" not in enriched[0]
    assert CONTEXT_SEPARATOR not in enriched[0]


def test_token_budget_reserve_shrinks_base_chunks():
    assert compute_base_chunk_size(512, 30, 100, "expand") == 512
    assert compute_base_chunk_size(512, 30, 100, "reserve") == 410  # 512 - 100 - 2


def test_token_budget_reserve_validates_overlap():
    with pytest.raises(ValueError, match="below chunk_overlap"):
        compute_base_chunk_size(120, 100, 100, "reserve")


def test_validate_context_strips_preambles_and_truncates():
    splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=5)
    chunker = ContextualChunker(
        base_chunker=splitter,
        llm_generator=MockContextLLM(),
        max_context_tokens=5,
    )

    cleaned = chunker._validate_context("Summary: this is a long context that exceeds five tokens easily.")
    assert cleaned is not None
    assert not cleaned.startswith("Summary:")
    # truncated to 5 tokens worth
    assert len(chunker._encoding.encode(cleaned)) <= 5
