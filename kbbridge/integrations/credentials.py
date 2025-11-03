"""
Generic Retrieval Credentials

Backend-agnostic credential management that works with any retrieval backend
(Dify, OpenSearch, Pinecone, Weaviate, etc.).
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RetrievalCredentials:
    """
    Generic credentials for retrieval backends.

    This class provides a backend-agnostic way to handle credentials
    for various retrieval systems without coupling to a specific provider.

    Attributes:
        endpoint: API endpoint or connection URL
        api_key: API key or authentication token
        backend_type: Type of backend ("dify", "opensearch", "pinecone", etc.)
        additional_config: Optional dict for backend-specific configuration
    """

    endpoint: str
    api_key: str
    backend_type: str = "dify"  # Default for backward compatibility
    additional_config: Optional[dict] = None

    @classmethod
    def from_env(cls, backend_type: Optional[str] = None) -> "RetrievalCredentials":
        """
        Create credentials from environment variables.

        Args:
            backend_type: Backend type (if None, uses RETRIEVAL_BACKEND env var)

        Environment Variables:
            RETRIEVAL_BACKEND: Backend type ("dify", "opensearch", etc.)
            RETRIEVAL_ENDPOINT: Generic endpoint for the retrieval backend
            RETRIEVAL_API_KEY: Generic API key

            Backend-specific (fallbacks):
            - OPENSEARCH_ENDPOINT, OPENSEARCH_AUTH
            - N8N_WEBHOOK_URL, N8N_API_KEY
            - etc.
        """
        backend_type = backend_type or os.getenv("RETRIEVAL_BACKEND", "dify").lower()

        # Try generic env vars first
        endpoint = os.getenv("RETRIEVAL_ENDPOINT")
        api_key = os.getenv("RETRIEVAL_API_KEY")

        # Fallback to backend-specific env vars
        if not endpoint or not api_key:
            if backend_type == "dify":
                # For Dify, RETRIEVAL_* is primary
                endpoint = endpoint or os.getenv("DIFY_ENDPOINT", "")
                api_key = api_key or os.getenv("DIFY_API_KEY", "")
            elif backend_type == "opensearch":
                endpoint = endpoint or os.getenv("OPENSEARCH_ENDPOINT", "")
                api_key = api_key or os.getenv("OPENSEARCH_AUTH", "")
            elif backend_type == "n8n":
                endpoint = endpoint or os.getenv("N8N_WEBHOOK_URL", "")
                api_key = api_key or os.getenv("N8N_API_KEY", "")

        return cls(
            endpoint=endpoint or "",
            api_key=api_key or "",
            backend_type=backend_type,
        )

    def validate(self) -> Tuple[bool, str]:
        """
        Validate credentials are present and properly formatted.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.endpoint:
            return False, f"Missing {self.backend_type} endpoint"

        if not self.api_key:
            return False, f"Missing {self.backend_type} API key"

        # Basic URL validation
        if not self.endpoint.startswith(("http://", "https://")):
            return False, f"Invalid endpoint URL: {self.endpoint}"

        # Backend-specific validation
        if self.backend_type == "dify":
            # Dify-specific validation (delegate to DifyCredentials)
            from .dify import DifyCredentials

            dify_creds = DifyCredentials(endpoint=self.endpoint, api_key=self.api_key)
            return dify_creds.validate()

        # Generic validation for other backends
        return True, ""

    def get_masked_summary(self) -> dict:
        """Get a masked summary of credentials for logging."""

        def mask_value(value: str) -> str:
            if not value:
                return "NOT SET"
            if len(value) <= 8:
                return "***"
            return f"{value[:4]}***{value[-4:]}"

        return {
            "backend_type": self.backend_type,
            "endpoint": mask_value(self.endpoint),
            "api_key": mask_value(self.api_key),
        }
