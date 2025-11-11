"""
Dify Adapter - High-Level API

Provides convenient high-level operations for Dify integration.
"""

from typing import Any, Dict, List, Optional

from .dify_credentials import DifyCredentials
from .dify_retriever import DifyRetriever


class DifyAdapter:
    """
    High-level Dify operations wrapper.

    Provides convenient methods for common Dify operations with
    automatic credential management and error handling.

    Example:
        # From environment
        adapter = DifyAdapter()

        # From explicit credentials
        adapter = DifyAdapter(
            credentials=DifyCredentials(
                endpoint="https://dify.com",
                api_key="your-api-key"
            )
        )

        # Search operation
        result = adapter.search(
            dataset_id="dataset-123",
            query="What is the policy?",
            top_k=20
        )
    """

    def __init__(self, credentials: Optional[DifyCredentials] = None):
        """
        Initialize Dify adapter.

        Args:
            credentials: DifyCredentials instance (defaults to environment)

        Raises:
            ValueError: If credentials are invalid
        """
        self.credentials = credentials or DifyCredentials.from_env()

        # Validate credentials on initialization
        valid, error = self.credentials.validate()
        if not valid:
            raise ValueError(f"Invalid Dify credentials: {error}")

    @classmethod
    def from_env(cls) -> "DifyAdapter":
        """
        Create adapter from environment variables.

        Returns:
            DifyAdapter instance

        Raises:
            ValueError: If environment credentials are invalid
        """
        return cls(credentials=DifyCredentials.from_env())

    @classmethod
    def from_params(
        cls, dify_endpoint: Optional[str] = None, dify_api_key: Optional[str] = None
    ) -> "DifyAdapter":
        """
        Create adapter from parameters.

        Args:
            dify_endpoint: Dify API endpoint URL
            dify_api_key: Dify API key

        Returns:
            DifyAdapter instance

        Raises:
            ValueError: If credentials are invalid
        """
        credentials = DifyCredentials(endpoint=dify_endpoint, api_key=dify_api_key)
        return cls(credentials=credentials)

    def create_retriever(self, dataset_id: str, timeout: int = 30) -> DifyRetriever:
        """
        Create a retriever for a dataset.

        Args:
            dataset_id: Dataset ID
            timeout: Request timeout in seconds

        Returns:
            DifyRetriever instance
        """
        return DifyRetriever(
            endpoint=self.credentials.endpoint,
            api_key=self.credentials.api_key,
            dataset_id=dataset_id,
            timeout=timeout,
        )

    def search(
        self,
        dataset_id: str,
        query: str,
        method: str = "hybrid_search",
        top_k: int = 20,
        does_rerank: bool = True,
        document_name: str = "",
        **options,
    ) -> Dict[str, Any]:
        """
        High-level search operation.

        Args:
            dataset_id: Dataset ID
            query: Search query
            method: Search method (semantic_search, hybrid_search, keyword_search)
            top_k: Number of results
            does_rerank: Whether to enable reranking
            document_name: Filter by document name
            **options: Additional search options

        Returns:
            Dictionary containing:
                - chunks: List of ChunkHit objects
                - files: List of FileHit objects
                - raw: Raw Dify API response
        """
        retriever = self.create_retriever(dataset_id)

        # Build metadata filter if needed
        metadata_filter = None
        if document_name:
            metadata_filter = retriever.build_metadata_filter(
                document_name=document_name
            )

        # Call Dify API
        raw_result = retriever.call(
            query=query,
            method=method,
            top_k=top_k,
            does_rerank=does_rerank,
            metadata_filter=metadata_filter,
            **options,
        )

        # Normalize results
        chunks = retriever.normalize_chunks(raw_result)
        files = retriever.group_files(chunks)

        return {
            "chunks": chunks,
            "files": files,
            "raw": raw_result,
        }

    def list_files(self, dataset_id: str, timeout: int = 30) -> List[str]:
        """
        List files in a dataset.

        Args:
            dataset_id: Dataset ID
            timeout: Request timeout in seconds

        Returns:
            List of file names
        """
        retriever = self.create_retriever(dataset_id, timeout=timeout)
        return retriever.list_files(dataset_id=dataset_id, timeout=timeout)

    def get_credentials_summary(self) -> Dict[str, str]:
        """
        Get masked summary of credentials for logging.

        Returns:
            Dictionary with masked credential status
        """
        return self.credentials.get_masked_summary()


def create_dify_adapter(
    dify_endpoint: Optional[str] = None,
    dify_api_key: Optional[str] = None,
) -> DifyAdapter:
    """
    Convenience function to create a Dify adapter.

    Args:
        dify_endpoint: Dify API endpoint URL (defaults to env var)
        dify_api_key: Dify API key (defaults to env var)

    Returns:
        DifyAdapter instance

    Raises:
        ValueError: If credentials are invalid
    """
    if dify_endpoint or dify_api_key:
        return DifyAdapter.from_params(
            dify_endpoint=dify_endpoint, dify_api_key=dify_api_key
        )
    else:
        return DifyAdapter.from_env()
