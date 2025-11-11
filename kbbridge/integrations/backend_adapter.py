from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from kbbridge.integrations.credentials import RetrievalCredentials


class BackendAdapter(ABC):
    """Generic backend adapter interface bound to a resource identifier."""

    def __init__(
        self, credentials: RetrievalCredentials, resource_id: Optional[str] = None
    ):
        """
        Initialize backend adapter.

        Args:
            credentials: Backend credentials
            resource_id: Optional resource identifier. If provided, adapter is resource-bound.
        """
        self.credentials = credentials
        self.resource_id = resource_id

    @property
    def _backend_id(self) -> str:
        """Backend-specific identifier (defaults to resource_id)."""
        if self.resource_id is None:
            raise ValueError("resource_id is required for this operation")
        return self.resource_id

    @abstractmethod
    def search(
        self,
        query: str,
        method: str = "hybrid_search",
        top_k: int = 20,
        does_rerank: bool = False,
        document_name: str = "",
        score_threshold: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
        resource_id: Optional[str] = None,
        **options,
    ) -> Dict[str, Any]:
        """
        Search without needing resource identifier (if resource-bound).

        Args:
            resource_id: Optional resource identifier (required if adapter not resource-bound)
        """

    @abstractmethod
    def list_files(
        self, timeout: int = 30, resource_id: Optional[str] = None
    ) -> List[str]:
        """
        List files without needing resource identifier (if resource-bound).

        Args:
            resource_id: Optional resource identifier (required if adapter not resource-bound)
        """

    @abstractmethod
    def build_metadata_filter(
        self, document_name: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Build backend-specific metadata filter from generic parameters."""


class BackendAdapterFactory:
    """Factory to create resource-bound backend adapters."""

    @staticmethod
    def create(
        resource_id: str,
        credentials: RetrievalCredentials,
        backend_type: Optional[str] = None,
    ) -> BackendAdapter:
        """
        Create a resource-bound adapter.

        Args:
            resource_id: Generic resource identifier
            credentials: Backend credentials
            backend_type: Backend type (defaults to credentials.backend_type)

        Raises:
            ValueError: If backend type is not supported
        """
        backend_type = backend_type or credentials.backend_type

        if backend_type == "dify":
            from kbbridge.integrations.dify.dify_backend_adapter import (
                DifyBackendAdapter,
            )

            return DifyBackendAdapter(credentials, resource_id)
        elif backend_type == "opensearch":
            raise NotImplementedError("OpenSearch backend adapter not yet implemented")
        elif backend_type == "n8n":
            raise NotImplementedError("n8n backend adapter not yet implemented")
        else:
            raise ValueError(
                f"Unsupported backend type: {backend_type}. "
                f"Supported backends: dify (opensearch, n8n coming soon)"
            )
