"""
Dify Credential Management

Handles validation and management of Dify API credentials.
"""

import os
from typing import Dict, Optional, Tuple


class DifyCredentials:
    """
    Manages Dify API credentials with validation.

    Handles common credential issues and provides helpful error messages.
    """

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Dify credentials.

        Args:
            endpoint: Dify API endpoint URL (defaults to RETRIEVAL_ENDPOINT or DIFY_ENDPOINT env var)
            api_key: Dify API key (defaults to RETRIEVAL_API_KEY or DIFY_API_KEY env var)
        """
        # Use generic RETRIEVAL_* variables first (preferred), fallback to DIFY_* for backward compatibility
        self.endpoint = (
            endpoint or os.getenv("RETRIEVAL_ENDPOINT") or os.getenv("DIFY_ENDPOINT")
        )
        self.api_key = (
            api_key or os.getenv("RETRIEVAL_API_KEY") or os.getenv("DIFY_API_KEY")
        )

    @classmethod
    def from_env(cls) -> "DifyCredentials":
        """
        Create credentials from environment variables.

        Returns:
            DifyCredentials instance
        """
        return cls()

    @classmethod
    def from_dict(cls, creds: Dict[str, Optional[str]]) -> "DifyCredentials":
        """
        Create credentials from dictionary.

        Args:
            creds: Dictionary with 'dify_endpoint' and 'dify_api_key' keys

        Returns:
            DifyCredentials instance
        """
        return cls(
            endpoint=creds.get("dify_endpoint"),
            api_key=creds.get("dify_api_key"),
        )

    @classmethod
    def from_params(
        cls,
        dify_endpoint: Optional[str] = None,
        dify_api_key: Optional[str] = None,
    ) -> "DifyCredentials":
        """
        Create credentials from named parameters.

        Args:
            dify_endpoint: Dify API endpoint URL
            dify_api_key: Dify API key

        Returns:
            DifyCredentials instance
        """
        return cls(endpoint=dify_endpoint, api_key=dify_api_key)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate credentials and return helpful error messages.

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if credentials are valid
            - error_message: None if valid, otherwise error description
        """
        # Check if endpoint is provided
        if not self.endpoint:
            return False, (
                "DIFY_ENDPOINT is required. "
                "Please set the DIFY_ENDPOINT environment variable or pass dify_endpoint parameter. "
                "Example: https://your-dify-instance.com"
            )

        # Check if api_key is provided
        if not self.api_key:
            return False, (
                "DIFY_API_KEY is required. "
                "Please set the DIFY_API_KEY environment variable or pass dify_api_key parameter."
            )

        # Check for common mistake: passing env var name instead of value
        if self.endpoint.startswith("env.") or self.endpoint.endswith("DIFY_ENDPOINT"):
            return False, (
                "Invalid DIFY_ENDPOINT format. "
                "You provided the environment variable name instead of the actual URL. "
                f"Current value: '{self.endpoint}'. "
                "Expected format: https://your-dify-instance.com"
            )

        # Check for http/https protocol
        if not self.endpoint.startswith(("http://", "https://")):
            return False, (
                "DIFY_ENDPOINT must start with http:// or https://. "
                f"Current value: '{self.endpoint}'. "
                "Example: https://your-dify-instance.com"
            )

        # Check if endpoint looks like a placeholder
        if "example" in self.endpoint.lower() or "placeholder" in self.endpoint.lower():
            return False, (
                "DIFY_ENDPOINT appears to be a placeholder. "
                f"Current value: '{self.endpoint}'. "
                "Please provide your actual Dify instance URL."
            )

        return True, None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """
        Convert credentials to dictionary.

        Returns:
            Dictionary with 'endpoint' and 'api_key' keys
        """
        return {
            "endpoint": self.endpoint,
            "api_key": self.api_key,
        }

    def to_service_dict(self) -> Dict[str, Optional[str]]:
        """
        Convert credentials to service-style dictionary.

        Returns:
            Dictionary with 'dify_endpoint' and 'dify_api_key' keys
        """
        return {
            "dify_endpoint": self.endpoint,
            "dify_api_key": self.api_key,
        }

    def is_set(self) -> bool:
        """
        Check if credentials are set (but not validated).

        Returns:
            True if both endpoint and api_key are non-empty
        """
        return bool(self.endpoint and self.api_key)

    def get_masked_summary(self) -> Dict[str, str]:
        """
        Get a summary with masked credentials for logging.

        Returns:
            Dictionary with masked credential status
        """
        return {
            "dify_endpoint": "SET" if self.endpoint else "NOT SET",
            "dify_api_key": "SET" if self.api_key else "NOT SET",
            "dify_endpoint_value": (
                f"{self.endpoint[:30]}..."
                if self.endpoint and len(self.endpoint) > 30
                else self.endpoint or "NOT SET"
            ),
        }

    def __repr__(self) -> str:
        """String representation with masked API key."""
        endpoint_display = self.endpoint if self.endpoint else "NOT SET"
        api_key_display = "***" if self.api_key else "NOT SET"
        return (
            f"DifyCredentials(endpoint={endpoint_display}, api_key={api_key_display})"
        )


def validate_dify_credentials(
    dify_endpoint: Optional[str] = None,
    dify_api_key: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[DifyCredentials]]:
    """
    Convenience function to validate Dify credentials.

    Args:
        dify_endpoint: Dify API endpoint URL
        dify_api_key: Dify API key

    Returns:
        Tuple of (is_valid, error_message, credentials)
        - is_valid: True if credentials are valid
        - error_message: None if valid, otherwise error description
        - credentials: DifyCredentials instance if valid, None otherwise
    """
    credentials = DifyCredentials(endpoint=dify_endpoint, api_key=dify_api_key)
    is_valid, error = credentials.validate()

    if is_valid:
        return True, None, credentials
    else:
        return False, error, None
