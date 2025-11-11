from typing import Any, Dict, List, Optional, Union

from kbbridge.integrations.backend_adapter import BackendAdapter
from kbbridge.integrations.credentials import RetrievalCredentials
from kbbridge.integrations.dify.dify_credentials import DifyCredentials
from kbbridge.integrations.dify.dify_retriever import DifyRetriever


class DifyBackendAdapter(BackendAdapter):
    """
    Dify-specific backend adapter implementation.

    Supports both resource-bound and non-resource-bound modes:
    - Resource-bound: DifyBackendAdapter(credentials, resource_id="dataset-123")
    - Non-resource-bound: DifyBackendAdapter(credentials) - methods take resource_id parameter

    Also supports backward compatibility with DifyCredentials:
    - DifyBackendAdapter(dify_credentials) - converts to RetrievalCredentials internally
    """

    def __init__(
        self,
        credentials: Union[RetrievalCredentials, DifyCredentials, None] = None,
        resource_id: Optional[str] = None,
    ):
        # Handle backward compatibility: accept DifyCredentials or None
        if credentials is None:
            dify_creds = DifyCredentials.from_env()
            credentials = RetrievalCredentials(
                endpoint=dify_creds.endpoint,
                api_key=dify_creds.api_key,
                backend_type="dify",
            )
        elif isinstance(credentials, DifyCredentials):
            # Convert DifyCredentials to RetrievalCredentials
            credentials = RetrievalCredentials(
                endpoint=credentials.endpoint,
                api_key=credentials.api_key,
                backend_type="dify",
            )

        super().__init__(credentials, resource_id)

        dify_creds = DifyCredentials(
            endpoint=credentials.endpoint, api_key=credentials.api_key
        )
        self._dify_creds = dify_creds

        # Create retriever if resource-bound
        if self.resource_id:
            self._dify_retriever = DifyRetriever(
                endpoint=dify_creds.endpoint,
                api_key=dify_creds.api_key,
                dataset_id=self._backend_id,
            )
        else:
            self._dify_retriever = None

    def _get_retriever(self, resource_id: Optional[str] = None) -> DifyRetriever:
        """Get retriever instance, creating if needed."""
        if self._dify_retriever:
            return self._dify_retriever

        if resource_id is None:
            raise ValueError(
                "resource_id is required (not provided in __init__ or method call)"
            )

        return DifyRetriever(
            endpoint=self._dify_creds.endpoint,
            api_key=self._dify_creds.api_key,
            dataset_id=resource_id,
        )

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
        dataset_id: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """
        Search using Dify backend.

        Args:
            resource_id: Optional resource identifier (required if adapter not resource-bound)
            dataset_id: Backward compatibility alias for resource_id
        """
        # Handle backward compatibility: dataset_id -> resource_id
        if dataset_id is not None and resource_id is None:
            resource_id = dataset_id

        retriever = self._get_retriever(resource_id)
        metadata_filter = retriever.build_metadata_filter(document_name=document_name)

        raw_result = retriever.call(
            query=query,
            method=method,
            top_k=top_k,
            does_rerank=does_rerank,
            metadata_filter=metadata_filter,
            score_threshold=score_threshold,
            weights=weights,
            **options
        )

        return raw_result

    def list_files(
        self,
        timeout: int = 30,
        resource_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> List[str]:
        """
        List files using Dify backend.

        Args:
            resource_id: Optional resource identifier (required if adapter not resource-bound)
            dataset_id: Backward compatibility alias for resource_id
        """
        # Handle backward compatibility: dataset_id -> resource_id
        if dataset_id is not None and resource_id is None:
            resource_id = dataset_id

        retriever = self._get_retriever(resource_id)
        dataset_id = resource_id if resource_id else self._backend_id
        return retriever.list_files(resource_id=dataset_id, timeout=timeout)

    def build_metadata_filter(
        self, document_name: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Build Dify-specific metadata filter from generic parameters."""
        if not self._dify_retriever:
            # Create a temporary retriever just for building filter
            temp_retriever = DifyRetriever(
                endpoint=self._dify_creds.endpoint,
                api_key=self._dify_creds.api_key,
                dataset_id="temp",  # Not used for filter building
            )
            return temp_retriever.build_metadata_filter(document_name=document_name)
        return self._dify_retriever.build_metadata_filter(document_name=document_name)

    def create_retriever(self, dataset_id: str, timeout: int = 30) -> DifyRetriever:
        """Create a retriever for a dataset (backward compatibility)."""
        return DifyRetriever(
            endpoint=self._dify_creds.endpoint,
            api_key=self._dify_creds.api_key,
            dataset_id=dataset_id,
            timeout=timeout,
        )

    def get_credentials_summary(self) -> Dict[str, str]:
        """Get masked summary of credentials for logging."""
        return self._dify_creds.get_masked_summary()

    @classmethod
    def from_env(cls, resource_id: Optional[str] = None) -> "DifyBackendAdapter":
        """Create adapter from environment variables."""
        credentials = RetrievalCredentials.from_env()
        return cls(credentials, resource_id)

    @classmethod
    def from_params(
        cls,
        dify_endpoint: Optional[str] = None,
        dify_api_key: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> "DifyBackendAdapter":
        """Create adapter from parameters."""
        credentials = RetrievalCredentials(
            endpoint=dify_endpoint or "",
            api_key=dify_api_key or "",
            backend_type="dify",
        )
        return cls(credentials, resource_id)


def create_dify_adapter(
    dify_endpoint: Optional[str] = None,
    dify_api_key: Optional[str] = None,
    resource_id: Optional[str] = None,
) -> DifyBackendAdapter:
    """
    Convenience function to create a Dify adapter (backward compatibility).

    Args:
        dify_endpoint: Dify API endpoint URL (defaults to env var)
        dify_api_key: Dify API key (defaults to env var)
        resource_id: Optional resource identifier for resource-bound mode

    Returns:
        DifyBackendAdapter instance
    """
    return DifyBackendAdapter.from_params(
        dify_endpoint=dify_endpoint,
        dify_api_key=dify_api_key,
        resource_id=resource_id,
    )
