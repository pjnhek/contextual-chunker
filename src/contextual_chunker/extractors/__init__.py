from pathlib import Path
from typing import Dict, Tuple

from contextual_chunker.extractors.docx import extract_docx
from contextual_chunker.extractors.pdf import extract_pdf
from contextual_chunker.extractors.text import extract_text_file


def extract_text(path: str | Path) -> Tuple[str, Dict]:
    """
    Read any supported document file and return (text, metadata).

    Supported formats: .txt, .md, .pdf, .docx

    Returns a tuple of:
        text: The extracted body text.
        metadata: {"source_file": str, "doc_type": str, "extractor": str}
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".txt", ".md"):
        return extract_text_file(p)
    if suffix == ".pdf":
        return extract_pdf(p)
    if suffix == ".docx":
        return extract_docx(p)

    raise ValueError(
        f"Unsupported file type: {suffix!r}. "
        f"Supported: .txt, .md, .pdf, .docx"
    )


__all__ = ["extract_text", "extract_text_file", "extract_pdf", "extract_docx"]
