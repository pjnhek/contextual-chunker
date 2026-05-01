from pathlib import Path
from typing import Dict, Tuple

import fitz  # pymupdf


def extract_pdf(path: Path) -> Tuple[str, Dict]:
    """Extract plain text from a PDF using PyMuPDF (fitz)."""
    pages = []
    with fitz.open(path) as doc:
        page_count = doc.page_count
        for page in doc:
            pages.append(page.get_text("text"))

    # Don't strip — trailing whitespace is meaningful for chunk-boundary parity
    # with downstream tokenizers, and any leading whitespace from the first page
    # is rarely worth losing structural fidelity over.
    text = "\n".join(pages)
    return text, {
        "source_file": str(path),
        "doc_type": "pdf",
        "extractor": "pymupdf",
        "page_count": page_count,
    }
