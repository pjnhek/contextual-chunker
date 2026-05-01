# contextual-chunker

A standalone library implementation of Anthropic's [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) pattern.

For each chunk in a document, an LLM generates a short context summary explaining what the document is about and where the chunk fits. The summary is prepended to the chunk before embedding, which substantially improves retrieval quality versus chunking alone (Anthropic reports ~35% reduction in retrieval failures on their internal benchmarks).

```
[CONTEXT] This is the user manual for the Acme Pro 9000 cordless drill, covering
battery care and warranty terms.

---

[ORIGINAL CHUNK] ...the lithium-ion battery should be charged for at least 4 hours
before first use and stored at 40-60% charge if unused for more than 30 days...
```

## What this is (and isn't)

- **In scope**: read documents (`.txt` / `.md` / `.pdf` / `.docx`) → token-chunk → enrich each chunk with LLM context → write embed-ready JSONL.
- **Out of scope**: embedding, Redis ingestion, retrieval, evaluation. Those are downstream of this library.

## Quickstart

```bash
git clone <this repo>
cd contextual-chunker
make install                                # uv sync
cp .env.example .env                        # fill in GOOGLE_API_KEY or OPENAI_API_KEY
make chunk-config CONFIG=config/example.yaml
```

Output lands in `data/out/chunks.jsonl`. One JSON object per line:

```json
{
  "chunk_id": "acme_pro_9000_manual_chunk_0",
  "text": "<context>\n\n---\n\n<original_chunk>",
  "original_chunk": "<original_chunk>",
  "chunk_context": "<llm-generated context>",
  "source_doc_id": "acme_pro_9000_manual",
  "chunk_index": 0,
  "metadata": {"source_file": "data/in/acme_pro_9000_manual.pdf", "doc_type": "pdf"}
}
```

The `text` field is what your embedder should consume.

## Bring-your-own LLM

Built-in providers: **Gemini** (`google-genai`) and **OpenAI**. To plug in any other provider (Anthropic, Bedrock, vLLM, etc.), implement `BaseContextLLM`:

```python
from contextual_chunker.llm.base import BaseContextLLM

class MyLLM(BaseContextLLM):
    async def generate_simple_async(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # your async LLM call returning a string
        ...
```

Then pass an instance directly:

```python
from contextual_chunker import ContextualChunker, TokenTextSplitter

chunker = ContextualChunker(
    base_chunker=TokenTextSplitter(chunk_size=512, chunk_overlap=128),
    llm_generator=MyLLM(),
    concurrency_limit=10,
)
```

## Configuration

See `config/example.yaml`. Key knobs:

| Field | Purpose |
|---|---|
| `chunk_size` / `chunk_overlap` | Token-based chunk size and overlap. |
| `contextual.token_budget` | `expand` keeps base chunks at `chunk_size` (enriched output is larger). `reserve` shrinks base chunks so enriched output fits within `chunk_size` — use this for embedders with hard token limits like BGE-512. |
| `contextual.max_context_tokens` | Hard cap on the LLM-generated context. |
| `contextual.concurrency_limit` | Max parallel LLM calls. |
| `contextual.llm_provider` | `gemini` or `openai`. |

## Downstream

The output JSONL is embedder-agnostic. See `examples/ingest_to_redis.py` for a reference script that embeds + writes to Redis (not part of the core library).
