from enum import Enum
from typing import Dict


class ReflectorDefaults(Enum):
    """User-configurable defaults for Reflector component."""

    # Reflection mode
    MODE = "standard"  # Options: "off", "standard", "comprehensive"

    # Enable/disable reflection by default
    ENABLED = True

    # Minimum quality score to pass (0-1)
    QUALITY_THRESHOLD = 0.70

    # Maximum answer refinement attempts
    MAX_ANSWER_ITERATIONS = 2


class ReflectionConstants:
    """Internal algorithm constants for reflection and quality evaluation."""

    # Quality thresholds
    DEFAULT_QUALITY_THRESHOLD: float = 0.70
    MINIMUM_SCORE_THRESHOLD: float = 0.70
    REFINEMENT_THRESHOLD_GAP: float = 0.30

    # Iteration limits
    DEFAULT_MAX_ITERATIONS: int = 2
    MAX_EXAMPLES_TO_USE: int = 4

    # LLM configuration
    DEFAULT_TEMPERATURE: float = 0.0

    # Score weights for overall quality calculation
    DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
        "completeness": 0.30,
        "accuracy": 0.30,
        "relevance": 0.20,
        "clarity": 0.10,
        "confidence": 0.10,
    }

    # Fallback score when evaluation fails
    FALLBACK_SCORE: float = 0.70

    @classmethod
    def validate_threshold(cls, threshold: float) -> bool:
        """Validate that threshold is in valid range [0.0, 1.0]."""
        return 0.0 <= threshold <= 1.0

    @classmethod
    def validate_weights(cls, weights: Dict[str, float]) -> bool:
        """Validate that weights sum to approximately 1.0."""
        total = sum(weights.values())
        return 0.99 <= total <= 1.01
