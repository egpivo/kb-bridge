from enum import Enum


class ResponseMessages:
    """Standard response messages used across the system."""

    NO_ANSWER = "N/A"
    NO_ANSWER_WITH_CONTEXT = "N/A - No relevant information found"


class AnswerExtractorDefaults(Enum):
    """Default values for Answer Extractor."""

    MAX_TOKENS = 12800
    TEMPERATURE = 0.0
    TIMEOUT = 120  # Increased from 60 to 120 to handle large context extractions
    TIMEOUT_SECONDS = (
        120  # Increased from 60 to 120 to handle large context extractions
    )
    AUTHORIZATION_HEADER = "Bearer dummy_token"


class StructuredAnswerFormatterDefaults(Enum):
    """Default values for Structured Answer Formatter."""

    MAX_TOKENS = 12800
    TEMPERATURE = 0.0
    TIMEOUT = 120
    TIMEOUT_SECONDS = 120
    AUTHORIZATION_HEADER = "Bearer dummy_token"

    # Payload size limits to prevent 413 errors
    MAX_CANDIDATE_ANSWER_LENGTH = 4000  # Max chars per candidate answer
    MAX_CANDIDATES = 20  # Max number of candidates to send to formatter
    RERANK_THRESHOLD = 100  # Use reranking if candidates exceed this number
