import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MCPConfigHelper:
    """Helper class for managing MCP service configuration"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the config helper

        Args:
            config_file: Optional path to a JSON configuration file
        """
        self.config_file = config_file or os.path.join(
            os.path.expanduser("~"), ".mcp_kb_config.json"
        )
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load config file {self.config_file}: {e}")

        return {}

    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            return True
        except IOError as e:
            logger.error(f"Error saving config file {self.config_file}: {e}")
            return False

    def set_credentials(self, **kwargs) -> None:
        """Set credentials in the configuration"""
        for key, value in kwargs.items():
            if value is not None:
                self.config[key] = value

    def get_credentials(self) -> Dict[str, str]:
        """Get all configured credentials"""
        return {
            "RETRIEVAL_ENDPOINT": self.config.get("retrieval_endpoint", ""),
            "RETRIEVAL_API_KEY": self.config.get("retrieval_api_key", ""),
            "LLM_API_URL": self.config.get("llm_api_url", ""),
            "LLM_MODEL": self.config.get("llm_model", ""),
            "LLM_API_TOKEN": self.config.get("llm_api_token", ""),
            "RERANK_URL": self.config.get("rerank_url", ""),
            "RERANK_MODEL": self.config.get("rerank_model", ""),
        }

    def apply_to_environment(self) -> None:
        """Apply configuration to environment variables"""
        credentials = self.get_credentials()
        for key, value in credentials.items():
            if value and not os.getenv(key):
                os.environ[key] = value

    def validate_credentials(self) -> Dict[str, Any]:
        """Validate that required credentials are present"""
        credentials = self.get_credentials()
        required = [
            "RETRIEVAL_ENDPOINT",
            "RETRIEVAL_API_KEY",
            "LLM_API_URL",
            "LLM_MODEL",
        ]

        missing = [cred for cred in required if not credentials.get(cred)]
        optional = [cred for cred in credentials.keys() if cred not in required]

        return {
            "valid": len(missing) == 0,
            "missing_required": missing,
            "optional_configured": [cred for cred in optional if credentials.get(cred)],
            "credentials": credentials,
        }
