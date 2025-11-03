"""
Keyword Generator Service

This service provides keyword generation functionality using LLM via direct API call.
"""

import os
from typing import Any, Dict, Optional

import kbbridge.core.query.keyword_generator as keyword_generator

# Default configuration
DEFAULT_CONFIG = {
    "max_sets": 5,
}


def keyword_generator_service(
    query: str,
    max_sets: int = DEFAULT_CONFIG["max_sets"],
    # Credentials (will be passed from environment or config)
    llm_api_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_token: Optional[str] = None,
    use_dspy: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Generate keyword sets using LLM via direct API call.

    This tool generates multiple keyword sets from a given query using language models
    to expand search capabilities and improve retrieval results.

    Args:
        query: The query to generate keywords from
        max_sets: Maximum number of keyword sets to generate (default: 5)
        llm_api_url: LLM API service URL
        llm_model: LLM model name

    Returns:
        Dict containing the generated keyword sets and metadata
    """
    try:
        # Get credentials from parameters or environment
        credentials_dict = {
            "llm_api_url": llm_api_url or os.getenv("LLM_API_URL"),
            "llm_model": llm_model or os.getenv("LLM_MODEL"),
            "llm_api_token": llm_api_token or os.getenv("LLM_API_TOKEN"),
        }

        # Validate required credentials
        required_creds = ["llm_api_url", "llm_model"]
        missing_creds = [
            cred for cred in required_creds if not credentials_dict.get(cred)
        ]
        if missing_creds:
            return {
                "error": f"Missing required credentials: {', '.join(missing_creds)}",
                "details": "Please provide all required credentials as parameters or environment variables",
            }

        # Validate parameters
        if not query:
            return {"error": "'query' parameter is required"}

        # Generate keywords using the extracted logic
        use_dspy_flag = (
            use_dspy
            if use_dspy is not None
            else os.getenv("KEYWORD_GENERATOR_USE_DSPY", "").lower()
            in {"1", "true", "yes", "on"}
        )

        result = keyword_generator.generate_keywords(
            query,
            max_sets,
            credentials_dict["llm_api_url"],
            credentials_dict["llm_model"],
            llm_api_token=credentials_dict.get("llm_api_token"),
            use_dspy=use_dspy_flag,
        )

        # Handle the result based on success/failure
        if result.get("success", False):
            # Remove the success flag before returning to user
            clean_result = {k: v for k, v in result.items() if k != "success"}
            return clean_result
        else:
            # Handle error cases
            error_type = result.get("error", "unknown_error")
            message = result.get("message", "Unknown error occurred")

            if error_type in ["api_error", "response_format_error"]:
                return {"error": message}
            elif error_type == "parse_error":
                # For parse errors, still return the result with raw_response
                clean_result = {
                    k: v for k, v in result.items() if k not in ["success", "error"]
                }
                return clean_result
            else:
                return {"error": message}

    except Exception as e:
        return {"error": f"Error generating keywords: {str(e)}"}
