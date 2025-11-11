"""
Example Implementation: Resource-Bound Adapter Pattern

This shows how we could refactor to move dataset_id into the adapter layer.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from kbbridge.integrations.credentials import RetrievalCredentials


class BackendAdapter(ABC):
    """
    Generic backend adapter interface.

    Adapters are bound to a resource identifier at initialization,
    so service methods don't need to pass identifiers.
    """

    def __init__(self, resource_id: str, credentials: RetrievalCredentials):
        """
        Initialize adapter with resource identifier.

        Args:
            resource_id: Generic resource identifier (e.g., "my-kb")
            credentials: Backend credentials
        """
        self.resource_id = resource_id
        self.credentials = credentials
        self._backend_id = self._translate_resource_id(resource_id)

    @abstractmethod
    def _translate_resource_id(self, resource_id: str) -> str:
        """
        Translate generic resource_id to backend-specific identifier.

        Args:
            resource_id: Generic resource identifier

        Returns:
            Backend-specific identifier (e.g., dataset_id for Dify, index_name for OpenSearch)
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        method: str = "hybrid_search",
        top_k: int = 20,
        **options
    ) -> Dict[str, Any]:
        """Search without needing resource identifier."""
        pass

    @abstractmethod
    def list_files(self, timeout: int = 30) -> List[str]:
        """List files without needing resource identifier."""
        pass


class DifyBackendAdapter(BackendAdapter):
    """Dify-specific adapter implementation."""

    def _translate_resource_id(self, resource_id: str) -> str:
        """For Dify, resource_id maps directly to dataset_id."""
        return resource_id  # No translation needed

    def search(self, query: str, method: str = "hybrid_search", top_k: int = 20, **options) -> Dict[str, Any]:
        """Search using self._backend_id (which is dataset_id for Dify)."""
        from kbbridge.integrations.dify import DifyAdapter, DifyCredentials

        dify_creds = DifyCredentials(
            endpoint=self.credentials.endpoint,
            api_key=self.credentials.api_key
        )
        adapter = DifyAdapter(credentials=dify_creds)
        return adapter.search(
            dataset_id=self._backend_id,  # Use translated identifier
            query=query,
            method=method,
            top_k=top_k,
            **options
        )

    def list_files(self, timeout: int = 30) -> List[str]:
        """List files using self._backend_id."""
        from kbbridge.integrations.dify import DifyAdapter, DifyCredentials

        dify_creds = DifyCredentials(
            endpoint=self.credentials.endpoint,
            api_key=self.credentials.api_key
        )
        adapter = DifyAdapter(credentials=dify_creds)
        return adapter.list_files(dataset_id=self._backend_id, timeout=timeout)


class OpenSearchBackendAdapter(BackendAdapter):
    """OpenSearch-specific adapter implementation."""

    def _translate_resource_id(self, resource_id: str) -> str:
        """For OpenSearch, resource_id maps to index_name."""
        return resource_id  # Could add prefix/suffix logic here if needed

    def search(self, query: str, method: str = "hybrid_search", top_k: int = 20, **options) -> Dict[str, Any]:
        """Search using self._backend_id (which is index_name for OpenSearch)."""
        # Implementation would use OpenSearch client with self._backend_id as index_name
        raise NotImplementedError("OpenSearch adapter not yet implemented")

    def list_files(self, timeout: int = 30) -> List[str]:
        """List files using self._backend_id."""
        # Implementation would use OpenSearch client
        raise NotImplementedError("OpenSearch adapter not yet implemented")


class BackendAdapterFactory:
    """Factory to create resource-bound adapters."""

    @staticmethod
    def create(
        resource_id: str,
        credentials: RetrievalCredentials,
        backend_type: Optional[str] = None
    ) -> BackendAdapter:
        """
        Create a resource-bound adapter.

        Args:
            resource_id: Generic resource identifier
            credentials: Backend credentials
            backend_type: Backend type (if None, uses credentials.backend_type)

        Returns:
            BackendAdapter instance bound to the resource
        """
        backend_type = backend_type or credentials.backend_type

        if backend_type == "dify":
            return DifyBackendAdapter(resource_id, credentials)
        elif backend_type == "opensearch":
            return OpenSearchBackendAdapter(resource_id, credentials)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")


# Example: How services would use this
def retriever_service_example(
    resource_id: str,  # Generic identifier, not dataset_id!
    query: str,
    method: str = "hybrid_search",
    top_k: int = 20,
    credentials: Optional[RetrievalCredentials] = None,
    **options
) -> Dict[str, Any]:
    """
    Example of how retriever_service would look with resource-bound adapters.

    Note: No dataset_id parameter needed - it's encapsulated in the adapter!
    """
    if not credentials:
        credentials = RetrievalCredentials.from_env()

    # Create resource-bound adapter
    adapter = BackendAdapterFactory.create(
        resource_id=resource_id,
        credentials=credentials
    )

    # Call adapter method - no resource identifier needed!
    return adapter.search(
        query=query,
        method=method,
        top_k=top_k,
        **options
    )
