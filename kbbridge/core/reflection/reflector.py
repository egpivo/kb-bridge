import logging
from typing import Any, Dict, List, Optional

from .config import setup
from .constants import ReflectionConstants
from .evaluator import Evaluator, get_default_examples
from .feedback import FeedbackGenerator
from .models import ReflectionResult

logger = logging.getLogger(__name__)


class Reflector:
    """Manages answer quality reflection and refinement process."""

    def __init__(
        self,
        llm_model: str,
        llm_api_url: str,
        api_key: str,
        threshold: float = ReflectionConstants.DEFAULT_QUALITY_THRESHOLD,
        max_iterations: int = ReflectionConstants.DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.evaluator: Optional[Evaluator] = None
        self.use_dspy = False

        try:
            setup(
                llm_model,
                llm_api_url,
                api_key,
                temperature=ReflectionConstants.DEFAULT_TEMPERATURE,
            )
            examples = get_default_examples()
            self.evaluator = Evaluator(threshold, examples=examples)
            self.use_dspy = True
        except Exception as e:
            logger.warning(f"Reflection disabled: DSPy setup failed: {e}")

        self.feedback = FeedbackGenerator()

    async def reflect(
        self, query: str, answer: str, sources: List[Dict[str, Any]], attempt: int = 1
    ) -> Optional[ReflectionResult]:
        """Reflect on answer quality and return evaluation result."""
        if not self.evaluator:
            logger.info("Reflection skipped: evaluator not initialized")
            return None

        logger.info(f"Reflecting attempt {attempt}")

        reflection = await self.evaluator.evaluate(
            query=query,
            answer=answer,
            sources=sources,
            attempt=attempt,
        )

        logger.info(
            f"Score: {reflection.overall_score:.2f}, Passed: {reflection.passed}"
        )

        return reflection

    def should_refine(self, reflection: ReflectionResult, current_attempt: int) -> bool:
        """Determine if answer should be refined based on reflection result."""
        if reflection.passed:
            return False
        if current_attempt >= self.max_iterations:
            return False
        if reflection.overall_score < (
            self.threshold - ReflectionConstants.REFINEMENT_THRESHOLD_GAP
        ):
            return False
        return True

    def create_report(self, reflections: List[ReflectionResult]) -> Dict[str, Any]:
        """Create summary report from reflection history."""
        if not reflections:
            return {}

        final = reflections[-1]
        report = {
            "total_attempts": len(reflections),
            "final_score": final.overall_score,
            "passed": final.passed,
            "threshold": final.threshold,
            "scores": final.scores.to_dict(),
            "dspy_enabled": self.use_dspy,
        }

        if len(reflections) > 1:
            report["improvement"] = final.overall_score - reflections[0].overall_score

        report["history"] = [
            {
                "attempt": i,
                "score": r.overall_score,
                "passed": r.passed,
                "feedback": r.feedback[:200],
            }
            for i, r in enumerate(reflections, 1)
        ]

        return report
