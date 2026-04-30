import json
from pathlib import Path
from typing import Iterable, Mapping


def write_jsonl(records: Iterable[Mapping], output_path: str | Path) -> int:
    """Write an iterable of records as JSONL. Returns the count written."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    return count


def slugify(name: str) -> str:
    """Turn a filename stem into a safe doc_id."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe.strip("_") or "doc"
