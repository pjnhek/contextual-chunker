from dataclasses import dataclass
from typing import AbstractSet, Any, Collection, List, Literal, Optional, Union

import tiktoken

from contextual_chunker.base import BaseChunker


@dataclass(frozen=True)
class _Tokenizer:
    chunk_overlap: int
    tokens_per_chunk: int
    decode: Any
    encode: Any


def _split_text_on_tokens(text: str, tokenizer: _Tokenizer) -> List[str]:
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return splits


class TokenTextSplitter(BaseChunker):
    """Split text into chunks based on token count using tiktoken."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 30,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str], None] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size})."
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._allowed_special = allowed_special if allowed_special is not None else set()
        self._disallowed_special = disallowed_special

        if model_name is not None:
            self._tokenizer = tiktoken.encoding_for_model(model_name)
        else:
            self._tokenizer = tiktoken.get_encoding(encoding_name)

    def split_text(self, text: str) -> List[str]:
        def _encode(t: str) -> List[int]:
            return self._tokenizer.encode(
                t,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenizer = _Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )
        return _split_text_on_tokens(text, tokenizer)

    def split_documents(self, documents, document_col: str):
        expanded_docs: List = []
        chunks: List[str] = []
        chunks_per_doc: List[int] = []

        for doc in documents:
            text = doc.get(document_col) if isinstance(doc, dict) else getattr(doc, document_col)
            count = 0
            for chunk in self.split_text(text):
                count += 1
                expanded_docs.append(doc)
                chunks.append(chunk)
            chunks_per_doc.append(count)

        return expanded_docs, chunks, chunks_per_doc
