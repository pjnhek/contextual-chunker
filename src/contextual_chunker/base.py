from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class BaseChunker(ABC):
    """Base interface for all chunkers."""

    @abstractmethod
    def split_documents(
        self, documents: Any, document_col: str
    ) -> Tuple[List, List[str], List[int]]:
        """
        Split documents into chunks.

        Args:
            documents: Iterable of dicts or objects.
            document_col: Attribute / key holding the text to chunk.

        Returns:
            (expanded_documents, chunks, chunks_per_document)
            - expanded_documents: Original document repeated once per chunk.
            - chunks: List of chunk strings.
            - chunks_per_document: Number of chunks each input document produced.
        """
        ...
