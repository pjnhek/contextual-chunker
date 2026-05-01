"""End-to-end test: CLI walks a tmp dir, chunks, and writes JSONL."""
import json
from pathlib import Path

from contextual_chunker import BaseContextLLM
from contextual_chunker.config import ChunkerConfig
from contextual_chunker import cli


class _StubLLM(BaseContextLLM):
    async def generate_simple_async(self, prompt, max_tokens, temperature):
        return "STUB CONTEXT"


def test_cli_run_writes_jsonl(tmp_path: Path, monkeypatch):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "doc_a.txt").write_text("alpha beta gamma " * 200, encoding="utf-8")
    (in_dir / "doc_b.md").write_text("# Title\n\n" + "lorem ipsum " * 200, encoding="utf-8")

    out_path = tmp_path / "out" / "chunks.jsonl"

    config = ChunkerConfig(
        chunk_size=64,
        chunk_overlap=8,
        input_dir=str(in_dir),
        output_path=str(out_path),
    )
    config.contextual.concurrency_limit = 2
    config.contextual.batch_size = 1
    config.contextual.max_context_tokens = 50

    # Don't call real LLMs.
    monkeypatch.setattr(cli, "_build_llm", lambda ctx: _StubLLM())

    written = cli.run(config)
    assert written > 0
    assert out_path.exists()

    records = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert len(records) == written

    seen_doc_ids = {r["source_doc_id"] for r in records}
    assert seen_doc_ids == {"doc_a", "doc_b"}

    for r in records:
        assert r["text"].startswith("STUB CONTEXT")
        assert r["chunk_id"] == f"{r['source_doc_id']}_chunk_{r['chunk_index']}"
        assert r["original_chunk"]
        assert r["chunk_context"] == "STUB CONTEXT"
        assert r["metadata"]["doc_type"] in {"txt", "md"}

    # chunk_index should restart from 0 per source doc
    indices_by_doc = {}
    for r in records:
        indices_by_doc.setdefault(r["source_doc_id"], []).append(r["chunk_index"])
    for doc_id, indices in indices_by_doc.items():
        assert indices == list(range(len(indices)))


def test_cli_uses_relative_paths_for_doc_ids(tmp_path: Path, monkeypatch):
    in_dir = tmp_path / "in"
    (in_dir / "alpha").mkdir(parents=True)
    (in_dir / "beta").mkdir()
    (in_dir / "alpha" / "report.txt").write_text("alpha beta gamma " * 50, encoding="utf-8")
    (in_dir / "beta" / "report.txt").write_text("lorem ipsum dolor " * 50, encoding="utf-8")

    out_path = tmp_path / "out" / "chunks.jsonl"
    config = ChunkerConfig(
        chunk_size=64,
        chunk_overlap=8,
        input_dir=str(in_dir),
        output_path=str(out_path),
    )
    config.contextual.concurrency_limit = 2
    config.contextual.batch_size = 1
    config.contextual.max_context_tokens = 50

    monkeypatch.setattr(cli, "_build_llm", lambda ctx: _StubLLM())

    written = cli.run(config)
    records = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]

    assert written == len(records)
    assert {r["source_doc_id"] for r in records} == {"alpha__report", "beta__report"}
    assert len({r["chunk_id"] for r in records}) == len(records)
