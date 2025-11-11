import os
from enum import Enum

from kbbridge.config.env_loader import load_env_file

# Ensure env file is loaded before reading env vars
load_env_file()


class DifyRetrieverDefaults(Enum):
    """Default values specific to Dify retriever.

    Values can be overridden via environment variables:
    - DIFY_RERANKING_PROVIDER_NAME
    - DIFY_RERANKING_MODEL_NAME
    """

    # Dify-specific reranking configuration
    # Load from env vars with fallback to defaults
    RERANKING_PROVIDER_NAME = os.getenv(
        "DIFY_RERANKING_PROVIDER_NAME",
        "langgenius/openai_api_compatible/openai_api_compatible",
    )
    RERANKING_MODEL_NAME = os.getenv(
        "DIFY_RERANKING_MODEL_NAME",
        "jinaai/jina-reranker-v2-base-multilingual",
    )

    # Default search parameters
    SEARCH_METHOD = "hybrid_search"
    DOES_RERANK = True
    TOP_K = 40
    SCORE_THRESHOLD = None
    WEIGHTS = 0.5


class DifySearchMethod(Enum):
    """Valid search methods for Dify API."""

    HYBRID_SEARCH = "hybrid_search"
    SEMANTIC_SEARCH = "semantic_search"
    FULL_TEXT_SEARCH = "full_text_search"
    KEYWORD_SEARCH = "keyword_search"
    VECTOR_SEARCH = "vector_search"
