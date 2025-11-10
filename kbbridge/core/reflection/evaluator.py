import json
import logging
from typing import Any, Dict, List, Optional

import dspy

from .constants import ReflectionConstants
from .models import QualityScores, ReflectionResult

logger = logging.getLogger(__name__)


class QualityEval(dspy.Signature):
    """DSPy signature for evaluating answer quality."""

    query: str = dspy.InputField()
    answer: str = dspy.InputField()
    sources: str = dspy.InputField()

    completeness: float = dspy.OutputField(desc="0-1")
    accuracy: float = dspy.OutputField(desc="0-1")
    clarity: float = dspy.OutputField(desc="0-1")
    relevance: float = dspy.OutputField(desc="0-1")
    confidence: float = dspy.OutputField(desc="0-1")
    feedback: str = dspy.OutputField()
    suggestions: str = dspy.OutputField(desc="JSON array")
    missing: str = dspy.OutputField(desc="JSON array")


class Evaluator:
    """Evaluates answer quality using DSPy and LLM."""

    def __init__(
        self,
        lm: dspy.LM,
        threshold: float = ReflectionConstants.DEFAULT_QUALITY_THRESHOLD,
        examples: Optional[List[Any]] = None,
    ) -> None:
        self._lm = lm
        self.threshold = threshold
        self.examples = (
            examples[: ReflectionConstants.MAX_EXAMPLES_TO_USE] if examples else []
        )
        self.evaluator = dspy.ChainOfThought(QualityEval)

        if self.examples:
            logger.info(f"Evaluator initialized with {len(self.examples)} examples")

    async def evaluate(
        self, query: str, answer: str, sources: List[Dict[str, Any]], attempt: int = 1
    ) -> ReflectionResult:
        """Evaluate answer quality and return structured result."""
        try:
            sources_text = self._format_sources(sources)

            logger.info(
                f"Evaluating answer (attempt {attempt}): "
                f"query_length={len(query)}, answer_length={len(answer)}"
            )

            with dspy.settings.context(lm=self._lm):
                result = self.evaluator(
                    query=query,
                    answer=answer,
                    sources=sources_text,
                )

            scores = QualityScores(
                completeness=float(result.completeness),
                accuracy=float(result.accuracy),
                clarity=float(result.clarity),
                relevance=float(result.relevance),
                confidence=float(result.confidence),
            )

            overall = scores.calculate_overall()
            passed = overall >= self.threshold

            logger.info(
                f"Evaluation result (attempt {attempt}): "
                f"overall={overall:.3f}, passed={passed}"
            )

            return ReflectionResult(
                scores=scores,
                overall_score=overall,
                passed=passed,
                feedback=result.feedback,
                refinement_suggestions=self._parse_json(result.suggestions),
                missing_aspects=self._parse_json(result.missing),
                attempt=attempt,
                threshold=self.threshold,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)

            return ReflectionResult(
                scores=QualityScores(
                    completeness=ReflectionConstants.FALLBACK_SCORE,
                    accuracy=ReflectionConstants.FALLBACK_SCORE,
                    clarity=ReflectionConstants.FALLBACK_SCORE,
                    relevance=ReflectionConstants.FALLBACK_SCORE,
                    confidence=ReflectionConstants.FALLBACK_SCORE,
                ),
                overall_score=ReflectionConstants.FALLBACK_SCORE,
                passed=True,
                feedback=str(e),
                attempt=attempt,
                threshold=self.threshold,
            )

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format source list into readable text."""
        if not sources:
            return "No sources"

        formatted = []
        for i, source in enumerate(sources[:10], 1):
            title = source.get("title", "Unknown")
            content = source.get("content", "")[:200]
            formatted.append(f"{i}. {title}\n   {content}...")
        return "\n".join(formatted)

    def _parse_json(self, text: str) -> List[str]:
        """Parse JSON array from text."""
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            items = text.strip("[]\"'")
            return [item.strip("\"'") for item in items.split(",") if item]


def get_default_examples() -> List[Any]:
    """Return empty list. Examples should be generated dynamically."""
    return []
