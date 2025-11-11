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
    TOP_K = 40  # Reduced to match adaptive top_k actual usage
    SCORE_THRESHOLD = None  # Disabled by default

    # File validation
    FILE_CHECK_TIMEOUT = 30

    # API timeouts
    DIFY_API_TIMEOUT = get_env_int("DIFY_API_TIMEOUT", 60)  # Timeout for Dify API calls

    # Overall request timeouts
    OVERALL_REQUEST_TIMEOUT = get_env_int(
        "OVERALL_REQUEST_TIMEOUT", 300
    )  # 5 minutes for entire request processing
    MCP_CLIENT_TIMEOUT = get_env_int(
        "MCP_CLIENT_TIMEOUT", 300
    )  # 5 minutes for MCP client timeout

    # Processing limits
    MAX_KEYWORDS = (
        8  # Increased from 5 - more keywords = better coverage for specialized queries
    )
    TOP_K_PER_KEYWORD = 20  # Reduced from 30 - avoid noise
    RERANK_THRESHOLD = 10
    RELEVANCE_SCORE_THRESHOLD = 0.1  # Lowered from 0.2 to 0.1 - minimal filtering to avoid missing relevant files
    MAX_FILES = 20  # Cap files per dataset for quality and to avoid candidate overload
    MAX_WORKERS = 10  # Reduced from 20 to 10 for better stability

    # Content booster settings
    USE_CONTENT_BOOSTER = True  # Enabled for parallel processing with adaptive top_k
    MAX_BOOST_KEYWORDS = 1
    ADAPTIVE_TOP_K_ENABLED = True  # Enable adaptive top_k based on number of queries
    TOTAL_SEGMENT_BUDGET = 80  # Optimized segment budget (2 queries × 40 segments)
    MIN_TOP_K_PER_QUERY = 10  # Minimum segments per query in adaptive calculation
    MAX_QUERY_WORKERS = 4  # Maximum parallel query workers
    DATASET_FILTER_WORKERS = 5  # Maximum workers for parallel dataset filtering

    # Display limits
    MAX_TOP_ANSWERS_TO_COMBINE = 3  # Limit for combining answers
    MAX_SOURCE_FILES_TO_SHOW = 5  # Maximum source files to include
    MAX_DISPLAY_SOURCES = 3  # Maximum sources to show in display_source
    MAX_FILE_SEARCH_KEYWORDS_TO_LOG = 5  # Maximum keywords to log

    # Query processing limits
    MAX_TOP_K_PER_FILE_QUERY = 40  # Maximum top_k per file query in advanced approach

    # LLM configuration defaults
    LLM_MAX_TOKENS = 12800
    LLM_TEMPERATURE = 0.0
    LLM_TIMEOUT_SECONDS = get_env_int("LLM_TIMEOUT_SECONDS", 120)
    LLM_AUTHORIZATION_HEADER = "Bearer dummy_token"

    # General settings
    VERBOSE = False

    # Advanced approach (top_k now calculated adaptively, see ADAPTIVE_TOP_K_ENABLED)
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

    MAX_KEYWORDS = 8  # Increased from 5 to 8 for better keyword diversity and coverage
    TOP_K_PER_KEYWORD = 50  # Increased from 20 to 50 to capture more candidates - file may rank low in vector search
    MAX_WORKERS = 8  # Increased to match MAX_KEYWORDS for parallel processing
    RERANK_THRESHOLD = 100
    RELEVANCE_SCORE_THRESHOLD = 0.0  # Must be float
    VERBOSE_MODE = False
    MAX_WORKERS_LIMIT = 10
    MIN_WORKERS_LIMIT = 1
    SEARCH_METHOD = "keyword_search"

    # File name detection and filtering
    ENABLE_FILE_NAME_FILTERING = True
    FILE_NAME_MATCH_THRESHOLD = 0.5  # Minimum similarity for file name matching
    PRIORITIZE_SPECIFIC_KEYWORDS = True  # Prioritize keywords containing file names
    MAX_FILES_WITHOUT_FILTERING = (
        5  # Apply filtering when more than this many files found
    )


class ContentBoosterDefaults(Enum):
    """Default values specifically for Content Booster tool"""

    # Search parameters
    SEARCH_METHOD = "hybrid_search"
    MAX_KEYWORDS = 15  # Increased from 10 to 15 for comprehensive search
    TOP_K_PER_KEYWORD = 50
    MAX_WORKERS = 10
    VERBOSE_MODE = False
    MAX_WORKERS_LIMIT = 5
    MIN_WORKERS_LIMIT = 1
    CONTENT_CHUNKS_LIMIT = (
        500  # Increased from 300 to 500 for more complete content extraction
    )


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
    TIMEOUT = 60  # Timeout for reflection API calls
    TEMPERATURE = 0.0
