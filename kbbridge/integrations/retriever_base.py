"""
Retriever interface for QA Hub.

Clean, backend-agnostic abstraction for retrieval operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ChunkHit:
    """Represents a single chunk (text segment) from retrieval."""

    def __init__(
        self,
        content: str,
        document_name: str,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.document_name = document_name
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self):
        return f"ChunkHit(document='{self.document_name}', score={self.score:.3f})"


class FileHit:
    """Represents a file with aggregated chunk scores."""

    def __init__(self, file_name: str, score: float, chunks: List[ChunkHit] = None):
        self.file_name = file_name
        self.score = score
        self.chunks = chunks or []

    def __repr__(self):
        return f"FileHit(file='{self.file_name}', score={self.score:.3f}, chunks={len(self.chunks)})"


class Retriever(ABC):
    """Abstract base class for retrieval backends (e.g., Dify, OpenSearch, Weaviate)."""

    @abstractmethod
    def call(self, *, query: str, method: str, top_k: int, **kw) -> Dict[str, any]:
        raise NotImplementedError

    @abstractmethod
    def normalize_chunks(self, resp: Dict[str, any]) -> List["ChunkHit"]:
        raise NotImplementedError

    @abstractmethod
    def group_files(
        self, chunks: List["ChunkHit"], agg: str = "max"
    ) -> List["FileHit"]:
        raise NotImplementedError

    @abstractmethod
    def build_metadata_filter(self, *, document_name: str = "") -> Optional[dict]:
        raise NotImplementedError

    @abstractmethod
    def list_files(self, *, resource_id: str, timeout: int = 30) -> List[str]:
        raise NotImplementedError("list_files is not implemented for this backend")
