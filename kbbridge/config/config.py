import os
from typing import Any, Dict, Optional

from kbbridge.config.env_loader import get_env_bool, get_env_int


class Credentials:
    """Backend-agnostic retrieval and LLM credentials"""

    def __init__(
        self,
        retrieval_endpoint: str,
        retrieval_api_key: str,
        llm_api_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_token: Optional[str] = None,
        rerank_url: Optional[str] = None,
        rerank_model: Optional[str] = None,
    ):
        """Initialize backend-agnostic retrieval credentials"""
        self.retrieval_endpoint = retrieval_endpoint
        self.retrieval_api_key = retrieval_api_key
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.llm_api_token = llm_api_token
        self.rerank_url = rerank_url
        self.rerank_model = rerank_model

    def is_reranking_available(self) -> bool:
        """Check if reranking is available based on credentials."""
        return bool(self.rerank_url and self.rerank_model)


class Config:
    """Configuration management class"""

    def __init__(self):
        """Initialize configuration with defaults"""
        self.config = self.get_default_config()

    @staticmethod
    def get_credentials_from_headers(headers: Dict[str, str]) -> Optional[Credentials]:
        """Extract credentials from HTTP headers (automatically lowercased by server)"""
        try:
            # Extract retrieval credentials from headers
            retrieval_endpoint = headers.get("x-retrieval-endpoint") or headers.get(
                "X-RETRIEVAL-ENDPOINT"
            )
            retrieval_api_key = headers.get("x-retrieval-api-key") or headers.get(
                "X-RETRIEVAL-API-KEY"
            )
            llm_endpoint = headers.get("x-llm-api-url") or headers.get("X-LLM-API-URL")
            llm_model = headers.get("x-llm-model") or headers.get("X-LLM-MODEL")
            rerank_endpoint = headers.get("x-rerank-url") or headers.get("X-RERANK-URL")
            rerank_model = headers.get("x-rerank-model") or headers.get(
                "X-RERANK-MODEL"
            )

            # Check if we have the minimum required credentials
            if not all([retrieval_endpoint, retrieval_api_key]):
                return None

            return Credentials(
                retrieval_endpoint=retrieval_endpoint,
                retrieval_api_key=retrieval_api_key,
                llm_api_url=llm_endpoint,
                llm_model=llm_model,
                llm_api_token=None,  # Not provided in headers
                rerank_url=rerank_endpoint,
                rerank_model=rerank_model,
            )
        except Exception:
            return None

    @staticmethod
    def get_default_credentials() -> Optional[Credentials]:
        """Get credentials from environment variables"""
        try:
            retrieval_endpoint = os.getenv("RETRIEVAL_ENDPOINT")
            retrieval_api_key = os.getenv("RETRIEVAL_API_KEY")
            llm_api_url = os.getenv("LLM_API_URL")
            llm_model = os.getenv("LLM_MODEL")
            llm_api_token = os.getenv("LLM_API_TOKEN")
            rerank_url = os.getenv("RERANK_URL")
            rerank_model = os.getenv("RERANK_MODEL")

            if not all([retrieval_endpoint, retrieval_api_key, llm_api_url, llm_model]):
                return None

            return Credentials(
                retrieval_endpoint=retrieval_endpoint,
                retrieval_api_key=retrieval_api_key,
                llm_api_url=llm_api_url,
                llm_model=llm_model,
                llm_api_token=llm_api_token,
                rerank_url=rerank_url,
                rerank_model=rerank_model,
            )
        except Exception:
            return None

    @staticmethod
    def get_search_config() -> Dict[str, Any]:
        """Get search configuration from environment variables"""
        return {
            "max_workers": get_env_int("MAX_WORKERS", 3),
            "verbose": get_env_bool("VERBOSE", False),
            "use_content_booster": get_env_bool("USE_CONTENT_BOOSTER", True),
            "max_boost_keywords": get_env_int(
                "MAX_BOOST_KEYWORDS", 1
            ),  # Reduced to 1 to limit content booster size
            "timeout": get_env_int("TIMEOUT", 30),
            # Timeout configurations
            "overall_request_timeout": get_env_int(
                "OVERALL_REQUEST_TIMEOUT", 300
            ),  # 5 minutes
            "mcp_client_timeout": get_env_int("MCP_CLIENT_TIMEOUT", 300),  # 5 minutes
            "dify_api_timeout": get_env_int("DIFY_API_TIMEOUT", 60),  # 1 minute
            "llm_timeout": get_env_int("LLM_TIMEOUT_SECONDS", 120),  # 2 minutes
        }

    @staticmethod
    def validate_credentials(credentials: Optional[Credentials]) -> bool:
        """Validate that credentials have required fields"""
        if credentials is None:
            return False
        return bool(credentials.retrieval_endpoint and credentials.retrieval_api_key)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "max_workers": 3,
            "verbose": False,
            "use_content_booster": True,
            "max_boost_keywords": 1,  # Reduced to 1 to limit content booster size
            "timeout": 30,
        }

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set the configuration"""
        self.config = config.copy()

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(key, default)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        self.config.update(updates)

    def reset_config(self) -> None:
        """Reset configuration to defaults"""
        self.config = self.get_default_config()

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        return self.config.copy()

    def import_config(self, config_data: Dict[str, Any]) -> bool:
        """Import configuration from data"""
        try:
            if isinstance(config_data, dict):
                self.config = config_data.copy()
                return True
            return False
        except Exception:
            return False
