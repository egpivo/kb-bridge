from enum import Enum


class KeywordGeneratorDefaults(Enum):
    """Default values for Keyword Generator."""

    MAX_TOKENS = 300
    TEMPERATURE = 0.0
    TIMEOUT = 30
    TIMEOUT_SECONDS = 30
    AUTHORIZATION_HEADER = "Bearer dummy_token"
    MAX_SETS = 3


class IntentionExtractorDefaults(Enum):
    """Default values for Intention Extractor."""

    MAX_TOKENS = 12800
    TEMPERATURE = 0.0
    TIMEOUT = 120
    TIMEOUT_SECONDS = 120
    AUTHORIZATION_HEADER = "Bearer dummy_token"
