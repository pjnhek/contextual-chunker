"""
Reference script — NOT part of the core library.

Reads chunks.jsonl produced by the chunker, embeds the `text` field with
your embedder of choice, and HSETs each record into Redis as a vector
document. This file exists so teammates have a starting point; copy it
into your own project and adapt it.

Dependencies (install in your own environment, NOT in this repo):
    pip install redis sentence-transformers numpy

Usage:
    python examples/ingest_to_redis.py \
        --input data/out/chunks.jsonl \
        --redis-host localhost \
        --redis-port 6379 \
        --index-name my_index \
        --embedding-model BAAI/bge-large-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-password", default=None)
    parser.add_argument("--index-name", required=True)
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-large-en-v1.5",
        help="Any sentence-transformers model id",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        import numpy as np
        import redis
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(
            "This example requires extra deps. Install in your project:\n"
            "    pip install redis sentence-transformers numpy",
            file=sys.stderr,
        )
        raise

    records = list(iter_jsonl(Path(args.input)))
    print(f"Loaded {len(records)} chunk records")

    model = SentenceTransformer(args.embedding_model)
    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} chunks with {args.embedding_model}...")
    vectors = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        password=args.redis_password,
        decode_responses=False,
    )

    pipe = client.pipeline()
    for record, vec in zip(records, vectors):
        key = f"{args.index_name}:{record['chunk_id']}"
        payload = {
            "document": record["text"],
            "original_chunk": record.get("original_chunk", ""),
            "chunk_context": record.get("chunk_context") or "",
            "source_doc_id": record["source_doc_id"],
            "chunk_index": record["chunk_index"],
            "vector": vec.tobytes(),
        }
        for k, v in record.get("metadata", {}).items():
            payload[f"meta_{k}"] = str(v)
        pipe.hset(key, mapping=payload)

    pipe.execute()
    print(f"Wrote {len(records)} keys to Redis at {args.redis_host}:{args.redis_port}")
    print(
        "Note: this script does NOT create a RediSearch index. "
        "Create one (FT.CREATE ...) matching your retrieval setup separately."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
