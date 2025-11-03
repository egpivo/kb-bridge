from enum import Enum


class DifyRetrieverDefaults(Enum):
    """Default values specific to Dify retriever."""

    # Dify-specific reranking configuration
    RERANKING_PROVIDER_NAME = "langgenius/openai_api_compatible/openai_api_compatible"
    RERANKING_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

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
