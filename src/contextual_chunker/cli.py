"""
CLI entry point: walk an input directory, chunk + enrich each file,
write embed-ready JSONL to disk.

Usage:
    python -m contextual_chunker --config config/example.yaml
    python -m contextual_chunker --input data/in --output data/out/chunks.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from contextual_chunker.chunker import ContextualChunker, compute_base_chunk_size
from contextual_chunker.config import ChunkerConfig, ContextualConfig
from contextual_chunker.extractors import extract_text
from contextual_chunker.io import slugify, write_jsonl
from contextual_chunker.llm.base import BaseContextLLM
from contextual_chunker.token_chunker import TokenTextSplitter


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


def _build_llm(ctx: ContextualConfig) -> BaseContextLLM:
    if ctx.llm_provider == "gemini":
        from contextual_chunker.llm.gemini import GeminiContextLLM
        return GeminiContextLLM(model_name=ctx.llm_model)
    if ctx.llm_provider == "openai":
        from contextual_chunker.llm.openai import OpenAIContextLLM
        return OpenAIContextLLM(model_name=ctx.llm_model)
    raise ValueError(f"Unknown llm_provider: {ctx.llm_provider!r}")


def _build_chunker(config: ChunkerConfig) -> ContextualChunker:
    ctx = config.contextual
    base_chunk_size = compute_base_chunk_size(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        max_context_tokens=ctx.max_context_tokens,
        token_budget=ctx.token_budget,
    )
    base = TokenTextSplitter(
        chunk_size=base_chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return ContextualChunker(
        base_chunker=base,
        llm_generator=_build_llm(ctx),
        batch_size=ctx.batch_size,
        max_context_tokens=ctx.max_context_tokens,
        max_llm_tokens=ctx.max_llm_tokens,
        temperature=ctx.temperature,
        timeout_seconds=ctx.timeout_seconds,
        concurrency_limit=ctx.concurrency_limit,
        max_retries=ctx.max_retries,
    )


def _gather_input_files(input_dir: Path) -> List[Path]:
    files = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(
            f"No supported files found in {input_dir} "
            f"(supported: {sorted(SUPPORTED_SUFFIXES)})"
        )
    return files


def run(config: ChunkerConfig) -> int:
    """Run extraction + chunking + enrichment for every file under input_dir."""
    input_dir = Path(config.input_dir)
    files = _gather_input_files(input_dir)
    logging.info(f"Found {len(files)} input files in {input_dir}")

    docs = []
    for path in files:
        text, meta = extract_text(path)
        if not text.strip():
            logging.warning(f"Skipping empty document: {path}")
            continue
        doc_id = slugify(path.stem)
        docs.append({"document": text, "doc_id": doc_id, "metadata": meta})

    chunker = _build_chunker(config)
    expanded_docs, enriched, originals, contexts, _ = chunker.get_chunk_contexts(docs, "document")

    # Build per-doc chunk indices so chunk_id is deterministic.
    chunk_index_per_doc: dict[str, int] = {}
    records = []
    for doc, enr, orig, ctx in zip(expanded_docs, enriched, originals, contexts):
        doc_id = doc["doc_id"]
        idx = chunk_index_per_doc.get(doc_id, 0)
        chunk_index_per_doc[doc_id] = idx + 1

        records.append({
            "chunk_id": f"{doc_id}_chunk_{idx}",
            "text": enr,
            "original_chunk": orig,
            "chunk_context": ctx,
            "source_doc_id": doc_id,
            "chunk_index": idx,
            "metadata": doc["metadata"],
        })

    written = write_jsonl(records, config.output_path)
    logging.info(f"Wrote {written} chunk records to {config.output_path}")
    return written


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(prog="contextual-chunker")
    parser.add_argument("--config", help="Path to YAML config (overrides --input/--output).")
    parser.add_argument("--input", help="Input directory (overrides config.input_dir).")
    parser.add_argument("--output", help="Output JSONL path (overrides config.output_path).")
    args = parser.parse_args(argv)

    if args.config:
        config = ChunkerConfig.from_yaml(args.config)
    else:
        config = ChunkerConfig()

    if args.input:
        config.input_dir = args.input
    if args.output:
        config.output_path = args.output

    try:
        run(config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
