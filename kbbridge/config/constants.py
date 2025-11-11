from enum import Enum

from kbbridge.config.env_loader import get_env_int


class RetrieverDefaults(Enum):
    """Default values for Knowledge Base Retriever (generic, backend-agnostic)."""

    SEARCH_METHOD = "hybrid_search"
    DOES_RERANK = True
    TOP_K = 40
    METADATA_FILTER = ""
    SCORE_THRESHOLD = None
    WEIGHTS = 0.5


class AssistantDefaults(Enum):
    """Default values for KB Assistant tool"""

    # Search parameters
    TOP_K = 40
    SCORE_THRESHOLD = None

    # File validation
    FILE_CHECK_TIMEOUT = 30

    # API timeouts
    RETRIEVAL_API_TIMEOUT = get_env_int("RETRIEVAL_API_TIMEOUT", 60)

    # Overall request timeouts
    OVERALL_REQUEST_TIMEOUT = get_env_int("OVERALL_REQUEST_TIMEOUT", 300)
    MCP_CLIENT_TIMEOUT = get_env_int("MCP_CLIENT_TIMEOUT", 300)

    # Processing limits
    MAX_KEYWORDS = 8
    TOP_K_PER_KEYWORD = 20
    RERANK_THRESHOLD = 10
    RELEVANCE_SCORE_THRESHOLD = 0.1
    MAX_FILES = 20
    MAX_WORKERS = 10

    # Content booster settings
    USE_CONTENT_BOOSTER = True
    MAX_BOOST_KEYWORDS = 1
    ADAPTIVE_TOP_K_ENABLED = True
    TOTAL_SEGMENT_BUDGET = 80
    MIN_TOP_K_PER_QUERY = 10
    MAX_QUERY_WORKERS = 4
    DATASET_FILTER_WORKERS = 5

    # Display limits
    MAX_TOP_ANSWERS_TO_COMBINE = 3
    MAX_SOURCE_FILES_TO_SHOW = 5
    MAX_DISPLAY_SOURCES = 3
    MAX_FILE_SEARCH_KEYWORDS_TO_LOG = 5

    # Query processing limits
    MAX_TOP_K_PER_FILE_QUERY = 40

    # LLM configuration defaults
    LLM_MAX_TOKENS = 12800
    LLM_TEMPERATURE = 0.0
    LLM_TIMEOUT_SECONDS = get_env_int("LLM_TIMEOUT_SECONDS", 120)
    LLM_AUTHORIZATION_HEADER = "Bearer dummy_token"

    # General settings
    VERBOSE = False

    # Advanced approach
    ADVANCED_APPROACH_SEARCH_METHOD = "hybrid_search"
    DOES_RERANK = True


class RetrieverSearchMethod(Enum):
    """Generic search methods (backend-agnostic)."""

    HYBRID_SEARCH = "hybrid_search"
    SEMANTIC_SEARCH = "semantic_search"
    FULL_TEXT_SEARCH = "full_text_search"
    KEYWORD_SEARCH = "keyword_search"
    VECTOR_SEARCH = "vector_search"


class FileSearcherDefaults(Enum):
    """Default values specifically for File Searcher tool"""

    MAX_KEYWORDS = 8
    TOP_K_PER_KEYWORD = 50
    MAX_WORKERS = 8
    RERANK_THRESHOLD = 100
    RELEVANCE_SCORE_THRESHOLD = 0.0
    VERBOSE_MODE = False
    MAX_WORKERS_LIMIT = 10
    MIN_WORKERS_LIMIT = 1
    SEARCH_METHOD = "keyword_search"

    # File name detection and filtering
    ENABLE_FILE_NAME_FILTERING = True
    FILE_NAME_MATCH_THRESHOLD = 0.5
    PRIORITIZE_SPECIFIC_KEYWORDS = True
    MAX_FILES_WITHOUT_FILTERING = 5


class ContentBoosterDefaults(Enum):
    """Default values specifically for Content Booster tool"""

    # Search parameters
    SEARCH_METHOD = "hybrid_search"
    MAX_KEYWORDS = 15
    TOP_K_PER_KEYWORD = 50
    MAX_WORKERS = 10
    VERBOSE_MODE = False
    MAX_WORKERS_LIMIT = 5
    MIN_WORKERS_LIMIT = 1
    CONTENT_CHUNKS_LIMIT = 500


class LLMDefaults(Enum):
    """Shared LLM configuration defaults"""

    MAX_TOKENS = 12800
    TEMPERATURE = 0
    TIMEOUT_SECONDS = 60
    AUTHORIZATION_HEADER = "Bearer dummy_token"


class ParagraphLocatorDefaults(Enum):
    """Default values specifically for Paragraph Locator"""

    MAX_TOKENS = 12800
    TEMPERATURE = 0.0
    TIMEOUT = 30
    TIMEOUT_SECONDS = 30
    AUTHORIZATION_HEADER = "Bearer dummy_token"


class ContentClusterDefaults(Enum):
    """Default values specifically for Content Cluster"""

    MAX_TOKENS = 12800
    TEMPERATURE = 0.0
    TIMEOUT = 30
    TIMEOUT_SECONDS = 30
    AUTHORIZATION_HEADER = "Bearer dummy_token"


class FileListerDefaults(Enum):
    """Default values specifically for File Lister tool"""

    LIMIT = 100

    # LLM configuration for reflection
    TIMEOUT = 60
    TEMPERATURE = 0.0
