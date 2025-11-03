import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env_file(env_file: Optional[str] = None) -> bool:
    """Load environment variables from .env file"""
    if env_file is None:
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"

    env_path = Path(env_file)

    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
        return False

    try:
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load .env file: {e}")
        return False


def get_env_var(key: str, default: str = "", required: bool = False) -> str:
    """Get environment variable with optional validation"""
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def get_env_int(key: str, default: int = 0, required: bool = False) -> int:
    """Get environment variable as integer"""
    value = get_env_var(key, str(default), required)
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be an integer, got: {value}")


def get_env_bool(key: str, default: bool = False, required: bool = False) -> bool:
    """Get environment variable as boolean"""
    value = get_env_var(key, str(default).lower(), required)
    return value.lower() in ("true", "1", "yes", "on")


def validate_env_config() -> dict:
    """Validate that all required environment variables are set"""
    retrieval_endpoint = os.getenv("RETRIEVAL_ENDPOINT") or os.getenv("DIFY_ENDPOINT")
    retrieval_api_key = os.getenv("RETRIEVAL_API_KEY") or os.getenv("DIFY_API_KEY")

    required_checks = [
        ("RETRIEVAL_ENDPOINT or DIFY_ENDPOINT", retrieval_endpoint),
        ("RETRIEVAL_API_KEY or DIFY_API_KEY", retrieval_api_key),
        ("LLM_API_URL", os.getenv("LLM_API_URL")),
        ("LLM_MODEL", os.getenv("LLM_MODEL")),
    ]

    missing_vars = [var_name for var_name, value in required_checks if not value]
    return {"valid": len(missing_vars) == 0, "missing": missing_vars}


def print_env_status():
    """Print current environment configuration status"""
    load_env_file()
    validation = validate_env_config()

    if validation["valid"]:
        logger.info("All required environment variables are set")
    else:
        logger.warning(
            f"Missing required variables: {', '.join(validation['missing'])}"
        )

    timeout_vars = [
        "OVERALL_REQUEST_TIMEOUT",
        "MCP_CLIENT_TIMEOUT",
        "DIFY_API_TIMEOUT",
        "LLM_TIMEOUT_SECONDS",
    ]
    for var in timeout_vars:
        value = os.getenv(var, "not set")
        logger.debug(f"{var}: {value}")


# Auto-load .env file when module is imported
load_env_file()
