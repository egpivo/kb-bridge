"""
Retriever Router

Routes retrieval requests to the appropriate backend based on environment configuration.
Supports multiple backends: Dify, OpenSearch, n8n, etc.
"""

import logging
import os
from typing import Any, Dict, Optional, Type

from .retriever_base import Retriever

logger = logging.getLogger(__name__)


class RetrieverRouter:
    """
    Routes retrieval requests to the appropriate backend based on configuration.

    Environment Variables:
        RETRIEVER_BACKEND: Backend type ("dify", "opensearch", "n8n", etc.)
        RETRIEVAL_ENDPOINT: Generic retrieval endpoint (preferred, works with any backend)
        RETRIEVAL_API_KEY: Generic API key (preferred, works with any backend)
        DIFY_ENDPOINT: Dify API endpoint (fallback for backward compatibility)
        DIFY_API_KEY: Dify API key (fallback for backward compatibility)
        OPENSEARCH_ENDPOINT: OpenSearch endpoint
        OPENSEARCH_AUTH: OpenSearch authentication
        N8N_WEBHOOK_URL: n8n webhook URL
        N8N_API_KEY: n8n API key
    """

    # Registry of available retrievers
    _retrievers: Dict[str, Type[Retriever]] = {}

    @classmethod
    def register_retriever(cls, backend_type: str, retriever_class: Type[Retriever]):
        """Register a retriever class for a backend type."""
        cls._retrievers[backend_type.lower()] = retriever_class

    @classmethod
    def get_available_backends(cls) -> list:
        """Get list of available backend types."""
        return list(cls._retrievers.keys())

    @classmethod
    def create_retriever(
        cls, dataset_id: str, backend_type: Optional[str] = None, **kwargs
    ) -> Retriever:
        """
        Create a retriever instance for the specified backend.

        Args:
            dataset_id: Dataset ID
            backend_type: Backend type (if None, uses RETRIEVER_BACKEND env var)
            **kwargs: Additional configuration parameters

        Returns:
            Retriever instance

        Raises:
            ValueError: If backend type is not supported or configuration is missing
        """
        # Determine backend type
        if backend_type is None:
            backend_type = os.getenv("RETRIEVER_BACKEND", "dify").lower()
        else:
            backend_type = backend_type.lower()

        if backend_type not in cls._retrievers:
            available = ", ".join(cls.get_available_backends())
            raise ValueError(
                f"Unsupported backend type: {backend_type}. "
                f"Available backends: {available}"
            )

        # Get retriever class
        retriever_class = cls._retrievers[backend_type]

        # Build configuration based on backend type
        config = cls._build_config(backend_type, dataset_id, **kwargs)

        # Create and return retriever instance
        return retriever_class(**config)

    @classmethod
    def _build_config(
        cls, backend_type: str, dataset_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Build configuration for the specified backend type."""
        config = {"dataset_id": dataset_id}

        if backend_type == "dify":
            # Use generic RETRIEVAL_* variables first (preferred), fallback to DIFY_* for backward compatibility
            endpoint = (
                kwargs.get("endpoint")
                or os.getenv("RETRIEVAL_ENDPOINT")
                or os.getenv("DIFY_ENDPOINT")
            )
            api_key = (
                kwargs.get("api_key")
                or os.getenv("RETRIEVAL_API_KEY")
                or os.getenv("DIFY_API_KEY")
            )

            config.update(
                {
                    "endpoint": endpoint,
                    "api_key": api_key,
                    "timeout": kwargs.get("timeout", 30),
                }
            )

            # Validate required credentials
            if not config["endpoint"]:
                raise ValueError(
                    "RETRIEVAL_ENDPOINT or DIFY_ENDPOINT environment variable is required for Dify backend"
                )
            if not config["api_key"]:
                raise ValueError(
                    "RETRIEVAL_API_KEY or DIFY_API_KEY environment variable is required for Dify backend"
                )

        elif backend_type == "opensearch":
            config.update(
                {
                    "endpoint": kwargs.get("endpoint")
                    or os.getenv("OPENSEARCH_ENDPOINT"),
                    "auth": kwargs.get("auth") or os.getenv("OPENSEARCH_AUTH"),
                    "index_name": kwargs.get("index_name")
                    or os.getenv("OPENSEARCH_INDEX", dataset_id),
                    "timeout": kwargs.get("timeout", 30),
                }
            )

            # Validate required OpenSearch credentials
            if not config["endpoint"]:
                raise ValueError(
                    "OPENSEARCH_ENDPOINT environment variable is required for OpenSearch backend"
                )

        elif backend_type == "n8n":
            config.update(
                {
                    "webhook_url": kwargs.get("webhook_url")
                    or os.getenv("N8N_WEBHOOK_URL"),
                    "api_key": kwargs.get("api_key") or os.getenv("N8N_API_KEY"),
                    "timeout": kwargs.get("timeout", 30),
                }
            )

            # Validate required n8n credentials
            if not config["webhook_url"]:
                raise ValueError(
                    "N8N_WEBHOOK_URL environment variable is required for n8n backend"
                )

        else:
            # For custom backends, pass through all kwargs
            config.update(kwargs)

        return config


# Auto-register available retrievers
def _register_default_retrievers():
    """
    Register default retriever implementations.

    TODO: Add support for additional retrievers:
    - OpenSearchRetriever (from .opensearch.opensearch_retriever)
    - N8NRetriever (from .n8n.n8n_retriever)
    """
    try:
        from .dify.dify_retriever import DifyRetriever

        RetrieverRouter.register_retriever("dify", DifyRetriever)
    except ImportError as e:
        logger.debug(f"Dify retriever not available: {e}")


# Initialize default retrievers
_register_default_retrievers()


# Convenience function for backward compatibility
def make_retriever(kind: str, **kwargs) -> Retriever:
    """
    Factory function to create a retriever instance.

    Args:
        kind: Retriever type ("dify", "opensearch", "n8n", etc.)
        **kwargs: Retriever-specific configuration

    Returns:
        Retriever instance
    """
    dataset_id = kwargs.pop("dataset_id", "default")
    return RetrieverRouter.create_retriever(dataset_id, kind, **kwargs)


# Convenience function for environment-based creation
def create_retriever_from_env(dataset_id: str, **kwargs) -> Retriever:
    """
    Create a retriever instance based on environment configuration.

    Args:
        dataset_id: Dataset ID
        **kwargs: Additional configuration parameters

    Returns:
        Retriever instance
    """
    return RetrieverRouter.create_retriever(dataset_id, **kwargs)
