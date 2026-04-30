from pathlib import Path
from typing import Dict, Tuple


def extract_text_file(path: Path) -> Tuple[str, Dict]:
    """Read .txt or .md as UTF-8."""
    text = Path(path).read_text(encoding="utf-8")
    return text, {
        "source_file": str(path),
        "doc_type": path.suffix.lstrip(".").lower(),
        "extractor": "text",
    }
