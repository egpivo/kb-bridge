import json
import logging
from typing import Any, Dict, Optional

from fastmcp.server.dependencies import get_http_headers

from kbbridge.config.config import Config, Credentials

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """Authentication middleware for extracting and validating credentials."""

    def __init__(self):
        self._session_credentials: Optional[Credentials] = None

    def get_credentials_from_request(self) -> Optional[Credentials]:
        """Extract credentials from the current HTTP request."""
        try:
            headers = get_http_headers(include_all=True)
            if headers:
                return Config.get_credentials_from_headers(headers)
            return None
        except Exception as e:
            logger.error(f"Error extracting credentials from request: {e}")
            return None

    def get_session_credentials(self) -> Optional[Credentials]:
        """Get session credentials."""
        return self._session_credentials

    def set_session_credentials(self, credentials: Credentials):
        """Set session credentials."""
        self._session_credentials = credentials

    def clear_session_credentials(self):
        """Clear session credentials."""
        self._session_credentials = None

    def get_available_credentials(self) -> Optional[Credentials]:
        """Get credentials from request or session."""
        # Try request headers first
        credentials = self.get_credentials_from_request()
        if credentials:
            return credentials

        # Fall back to session
        session_creds = self.get_session_credentials()
        if session_creds:
            return session_creds

        try:
            env_creds = Config.get_default_credentials()
            if env_creds:
                return env_creds
        except Exception as e:
            logger.debug(f"Failed to get default credentials from config: {e}")

        return None

    def validate_credentials(self, credentials: Credentials) -> Dict[str, Any]:
        """Validate credentials and return validation result."""
        try:
            # Basic validation
            if not credentials.retrieval_endpoint or not credentials.retrieval_api_key:
                return {
                    "valid": False,
                    "errors": ["Missing required retrieval backend credentials"],
                }

            # TODO: Add actual credential validation (ping endpoints, etc.)
            return {"valid": True, "errors": []}

        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {str(e)}"]}

    def create_auth_error_response(self, message: str, errors: list = None) -> str:
        """Create standardized authentication error response."""
        return json.dumps(
            {
                "error": "Authentication failed",
                "status": "error",
                "message": message,
                "errors": errors or [],
                "required_headers": [
                    "X-RETRIEVAL-ENDPOINT",
                    "X-RETRIEVAL-API-KEY",
                    "X-LLM-API-URL",
                    "X-LLM-MODEL",
                ],
            }
        )


# Global middleware instance
auth_middleware = AuthMiddleware()


def get_current_credentials() -> Optional[Credentials]:
    """Get the current request credentials."""
    return auth_middleware.get_available_credentials()


def set_current_credentials(credentials: Optional[Credentials]):
    """Set the current request credentials."""
    if credentials:
        auth_middleware.set_session_credentials(credentials)
    else:
        auth_middleware.clear_session_credentials()


# Export for testing
__all__ = [
    "AuthMiddleware",
    "auth_middleware",
    "Credentials",
    "get_current_credentials",
    "set_current_credentials",
]
