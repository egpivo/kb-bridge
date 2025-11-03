from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QualityScores:
    """Quality evaluation scores across multiple dimensions."""

    completeness: float  # 0-1: Does it fully address the query?
    accuracy: float  # 0-1: Are sources relevant and correctly cited?
    clarity: float  # 0-1: Is it clear and well-structured?
    relevance: float  # 0-1: Does it stay on topic?
    confidence: float  # 0-1: Quality of supporting sources?

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "clarity": self.clarity,
            "relevance": self.relevance,
            "confidence": self.confidence,
        }

    def calculate_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall quality score using weighted average.

        Args:
            weights: Custom weights for each dimension (default: balanced)

        Returns:
            Overall quality score (0-1)
        """
        if weights is None:
            weights = {
                "completeness": 0.30,
                "accuracy": 0.30,
                "relevance": 0.20,
                "clarity": 0.10,
                "confidence": 0.10,
            }

        return (
            self.completeness * weights["completeness"]
            + self.accuracy * weights["accuracy"]
            + self.relevance * weights["relevance"]
            + self.clarity * weights["clarity"]
            + self.confidence * weights["confidence"]
        )


@dataclass
class ReflectionResult:
    """Result of answer quality reflection."""

    # Quality metrics
    scores: QualityScores
    overall_score: float
    passed: bool  # Whether quality meets threshold

    # Feedback
    feedback: str
    refinement_suggestions: List[str] = field(default_factory=list)
    missing_aspects: List[str] = field(default_factory=list)

    # Metadata
    attempt: int = 1
    threshold: float = 0.70

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "scores": self.scores.to_dict(),
            "overall_score": self.overall_score,
            "passed": self.passed,
            "feedback": self.feedback,
            "refinement_suggestions": self.refinement_suggestions,
            "missing_aspects": self.missing_aspects,
            "attempt": self.attempt,
            "threshold": self.threshold,
        }


@dataclass
class RefinementPlan:
    """Plan for improving answer based on reflection feedback."""

    strategy: str  # 'refine_keywords', 'expand_sources', 're_generate', 'abort'
    refined_query: Optional[str] = None
    additional_keywords: List[str] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "refined_query": self.refined_query,
            "additional_keywords": self.additional_keywords,
            "focus_areas": self.focus_areas,
            "reasoning": self.reasoning,
        }
