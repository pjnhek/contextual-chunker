"""
Contextual chunking decorator: enriches chunks with LLM-generated document context.

For each chunk, an LLM generates a concise summary of the parent document, which
is prepended to the chunk before embedding. This disambiguates chunks during
retrieval (Anthropic's contextual retrieval pattern).

Async concurrency (configurable semaphore) processes chunks in parallel.
"""

import asyncio
import logging
import re
import time
from typing import Any, List, Literal, Optional, Tuple

import nest_asyncio
import tiktoken

from contextual_chunker.base import BaseChunker
from contextual_chunker.llm.base import BaseContextLLM
from contextual_chunker.prompts import CONTEXT_SEPARATOR, CONTEXTUAL_CHUNK_PROMPT


def compute_base_chunk_size(
    chunk_size: int,
    chunk_overlap: int,
    max_context_tokens: int,
    token_budget: Literal["expand", "reserve"],
) -> int:
    """
    Decide how big the underlying base chunks should be.

    - "expand": base chunks stay at chunk_size; enriched output is larger
      (~chunk_size + max_context_tokens). Use when your embedder accepts long inputs.
    - "reserve": base chunks shrink so the enriched output fits within chunk_size.
      Use for embedders with hard token limits (e.g., BGE-Large @ 512 tokens).
    """
    if token_budget == "expand":
        return chunk_size

    if token_budget == "reserve":
        # CONTEXT_SEPARATOR is "\n\n---\n\n" (~2 tokens)
        separator_tokens = 2
        base = chunk_size - max_context_tokens - separator_tokens
        if base < chunk_overlap:
            raise ValueError(
                f"chunk_size ({chunk_size}) - max_context_tokens "
                f"({max_context_tokens}) leaves only {base} tokens for content, "
                f"which is below chunk_overlap ({chunk_overlap}). Increase "
                f"chunk_size or decrease max_context_tokens."
            )
        return base

    raise ValueError(f"Unknown token_budget: {token_budget!r}")


class ContextualChunker(BaseChunker):
    """
    Wraps any BaseChunker, then enriches each chunk with an LLM-generated context.

    Output format per chunk:
        "{contextual_summary}\\n\\n---\\n\\n{original_chunk_text}"

    Enrichment runs asynchronously via a semaphore for rate-limit-friendly parallelism.
    """

    def __init__(
        self,
        base_chunker: BaseChunker,
        llm_generator: BaseContextLLM,
        batch_size: int = 5,
        max_context_tokens: int = 100,
        max_llm_tokens: int = 600,
        temperature: float = 1.0,
        timeout_seconds: int = 30,
        concurrency_limit: int = 10,
        max_retries: int = 3,
    ):
        """
        Args:
            base_chunker: Underlying chunker to delegate splitting to.
            llm_generator: Any BaseContextLLM implementation.
            batch_size: How often to log enrichment progress (every N chunks).
            max_context_tokens: Truncation cap for the LLM-generated context.
            max_llm_tokens: Output token cap for the LLM call. Set higher than
                max_context_tokens for models that spend output budget on
                internal thinking (e.g. Gemini Flash).
            temperature: LLM temperature for context generation.
            timeout_seconds: Per-call timeout (informational; rely on SDK timeouts).
            concurrency_limit: Max parallel LLM calls (semaphore size).
            max_retries: Retry attempts per chunk on transient LLM failures.
        """
        self.base_chunker = base_chunker
        self.llm_generator = llm_generator
        self.batch_size = batch_size
        self.max_context_tokens = max_context_tokens
        self.max_llm_tokens = max_llm_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.concurrency_limit = concurrency_limit
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_documents(
        self, documents: Any, document_col: str
    ) -> Tuple[List, List[str], List[int]]:
        expanded_docs, raw_chunks, chunk_counts = self.base_chunker.split_documents(
            documents, document_col
        )
        enriched_chunks, _, _ = self._run_enrichment(expanded_docs, raw_chunks, document_col)
        return expanded_docs, enriched_chunks, chunk_counts

    def get_chunk_contexts(
        self, documents: Any, document_col: str
    ) -> Tuple[List, List[str], List[str], List[Optional[str]], List[int]]:
        """
        Same as split_documents but also returns the original chunks + contexts
        separately. Useful for callers that want to write all three to disk
        (e.g. JSONL with `text`, `original_chunk`, `chunk_context`).

        Returns:
            (expanded_docs, enriched_chunks, original_chunks, contexts, chunk_counts)
        """
        expanded_docs, raw_chunks, chunk_counts = self.base_chunker.split_documents(
            documents, document_col
        )
        enriched_chunks, contexts, _ = self._run_enrichment(
            expanded_docs, raw_chunks, document_col
        )
        return expanded_docs, enriched_chunks, raw_chunks, contexts, chunk_counts

    # ------------------------------------------------------------------
    # Sync ↔ async bridge
    # ------------------------------------------------------------------

    def _run_enrichment(
        self,
        expanded_docs: List,
        raw_chunks: List[str],
        document_col: str,
    ) -> Tuple[List[str], List[Optional[str]], int]:
        """
        Run the async enrichment loop from a sync context.

        Uses nest_asyncio so this works whether called from a plain script
        or from inside an existing event loop (Jupyter, etc.).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._enrich_chunks_async(expanded_docs, raw_chunks, document_col)

        if loop and loop.is_running():
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return asyncio.run(coro)

    async def _enrich_chunks_async(
        self,
        expanded_docs: List,
        raw_chunks: List[str],
        document_col: str,
    ) -> Tuple[List[str], List[Optional[str]], int]:
        total = len(raw_chunks)
        self.logger.info(
            f"Contextual chunking: enriching {total} chunks "
            f"(concurrency={self.concurrency_limit})"
        )

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        completed = 0

        async def _process_one(doc: Any, chunk: str) -> Tuple[str, Optional[str]]:
            nonlocal completed
            full_doc_text = (
                doc.get(document_col) if isinstance(doc, dict) else getattr(doc, document_col)
            )
            context = await self._generate_chunk_context_async(full_doc_text, chunk, semaphore)
            validated = self._validate_context(context)
            enriched = f"{validated}{CONTEXT_SEPARATOR}{chunk}" if validated else chunk

            completed += 1
            if completed % self.batch_size == 0 or completed == total:
                self.logger.info(f"Contextual chunking progress: {completed}/{total}")

            return enriched, validated

        results = await asyncio.gather(
            *[_process_one(d, c) for d, c in zip(expanded_docs, raw_chunks)]
        )

        enriched_chunks = [r[0] for r in results]
        contexts = [r[1] for r in results]
        skipped = sum(1 for c in contexts if c is None)

        self.logger.info(
            f"Contextual chunking complete: {total - skipped}/{total} enriched, "
            f"{skipped} skipped (no useful context)"
        )

        return enriched_chunks, contexts, skipped

    # ------------------------------------------------------------------
    # LLM call with retry
    # ------------------------------------------------------------------

    async def _generate_chunk_context_async(
        self, full_doc: str, chunk: str, semaphore: asyncio.Semaphore
    ) -> Optional[str]:
        prompt = CONTEXTUAL_CHUNK_PROMPT.format(full_document=full_doc, chunk_text=chunk)

        async with semaphore:
            start_time = time.time()
            last_error: Optional[Exception] = None
            context: Optional[str] = None

            for attempt in range(1, self.max_retries + 1):
                try:
                    context = await self.llm_generator.generate_simple_async(
                        prompt=prompt,
                        max_tokens=self.max_llm_tokens,
                        temperature=self.temperature,
                    )
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries:
                        backoff = self._parse_retry_delay(e) or 2 ** (attempt - 1)
                        self.logger.warning(
                            f"Context generation attempt {attempt}/{self.max_retries} "
                            f"failed: {e}. Retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                    else:
                        self.logger.error(
                            f"Context generation failed after {self.max_retries} "
                            f"attempts: {e}. Skipping chunk."
                        )

            if last_error is not None:
                return None

            self.logger.debug(f"Context generated in {time.time() - start_time:.2f}s")

        if not context or not context.strip():
            preview = chunk[:80].replace("\n", " ")
            self.logger.info(f"Empty context, skipping enrichment for chunk: \"{preview}...\"")
            return None

        return context.strip()

    # ------------------------------------------------------------------
    # Validation / truncation
    # ------------------------------------------------------------------

    def _validate_context(self, context: Optional[str]) -> Optional[str]:
        if not context or not context.strip():
            return None

        context = context.strip()

        for preamble in ("Here is the summary:", "Contextual summary:", "Summary:"):
            if context.startswith(preamble):
                context = context[len(preamble):].strip()

        token_count = len(self._encoding.encode(context))
        if token_count > self.max_context_tokens:
            self.logger.warning(
                f"Context exceeds {self.max_context_tokens} tokens "
                f"({token_count}); truncating."
            )
            context = self._truncate_to_tokens(context, self.max_context_tokens)

        return context

    @staticmethod
    def _parse_retry_delay(error: Exception) -> Optional[float]:
        """Pull a retry delay out of common API error strings (429 / RESOURCE_EXHAUSTED)."""
        match = re.search(r"retry in ([\d.]+)s", str(error), re.IGNORECASE)
        if match:
            return float(match.group(1))
        if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
            return 20.0
        return None

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])
