# AGENTS.md

Instructions for AI coding agents (Cursor, Claude Code, Codex, etc.) working in this repo. If your tool reads `CLAUDE.md` or `.cursorrules` instead, copy or symlink this file.

## What this repo is

A standalone library implementation of Anthropic's contextual retrieval pattern (https://www.anthropic.com/news/contextual-retrieval). It takes documents in (`.txt`, `.md`, `.pdf`, `.docx`), runs token-based chunking, then calls an LLM per chunk to generate a short document-context summary that's prepended to the chunk before embedding. Output is embed-ready JSONL.

## Setup

Requires `uv` (https://docs.astral.sh/uv/getting-started/installation/) and `make` on `PATH` first. If `uv` is missing, run `curl -LsSf https://astral.sh/uv/install.sh | sh`. If `make` is missing (Windows), the human user can run the commands inside `Makefile` directly.

```bash
make install                          # uv sync
cp .env.example .env                  # fill in GOOGLE_API_KEY or OPENAI_API_KEY
make chunk-config CONFIG=config/example.yaml
```

## Output contract

One JSON object per line in `data/out/chunks.jsonl`:

```json
{
  "chunk_id": "doc1_chunk_0",
  "text": "<context>\n\n---\n\n<original_chunk>",
  "original_chunk": "<original_chunk>",
  "chunk_context": "<llm-generated context>",
  "source_doc_id": "doc1",
  "chunk_index": 0,
  "metadata": {"source_file": "...", "doc_type": "pdf"}
}
```

The `text` field is what downstream embedders should consume. `original_chunk` and `chunk_context` are kept separately for debugging and for retrieval-quality evaluations.

## Where to extend

- **New file format** (e.g. HTML, RTF): add a module under `src/contextual_chunker/extractors/` exporting `extract_<format>(path) -> (text, metadata)`, then register the suffix in `extractors/__init__.py:extract_text`.
- **New LLM provider** (e.g. Anthropic, Bedrock, vLLM, Azure): subclass `BaseContextLLM` from `src/contextual_chunker/llm/base.py` and implement `async def generate_simple_async(prompt, max_tokens, temperature) -> str`. Register in `cli.py:_build_llm` if you want it selectable from YAML, or pass an instance directly to `ContextualChunker(llm_generator=...)`.
- **New chunking strategy** (e.g. semantic, recursive): subclass `BaseChunker`. `ContextualChunker` wraps any `BaseChunker`, so you only have to implement `split_documents`.

## Don't do

- **Don't add Redis, vector stores, or embedding code.** The library outputs JSONL on purpose so it's storage-agnostic. A reference Redis ingestion script lives in `examples/` (not part of core).
- **Don't bake any user's docs / API keys / customer data into the repo.** `data/in/*` and `data/out/*` are gitignored. `.env` is gitignored. Keep it that way.
- **Don't add async/sync duplication.** The chunker's enrichment loop is async-native; sync usage works via the `nest_asyncio` bridge in `_run_enrichment`. Don't add a parallel sync code path.
- **Don't expand the built-in LLM providers without a real reason.** Two adapters (Gemini, OpenAI) plus a clean `BaseContextLLM` abstract class is the design — anything else is the caller's adapter.

## Conventions

- `uv` for dependency management. Add deps via `pyproject.toml`, then `uv sync`. Don't use `pip install` directly.
- Tests live in `tests/`; run with `make test`. Tests must be hermetic — no real API keys, no network calls. Use a stub `BaseContextLLM` (see `tests/test_chunker.py:MockContextLLM` and `tests/test_cli.py:_StubLLM`).
- Keep modules small and single-purpose. The whole repo should be readable in 10 minutes.
