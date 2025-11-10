import logging
from typing import Any, Dict, List, Optional

from fastmcp import Context

from .constants import ReflectorDefaults
from .reflector import Reflector

logger = logging.getLogger(__name__)


class ReflectionIntegration:
    """Adapter for integrating Reflector into services with optional MCP Context support."""

    def __init__(
        self,
        llm_api_url: str,
        llm_model: str,
        llm_api_token: str,
        enable_reflection: bool = True,
        quality_threshold: float = ReflectorDefaults.QUALITY_THRESHOLD.value,
        max_iterations: int = ReflectorDefaults.MAX_ANSWER_ITERATIONS.value,
    ):
        """Initialize reflection integration."""
        self.enable_reflection = enable_reflection
        self.reflector = None

        if enable_reflection:
            try:
                if not llm_api_token:
                    logger.warning("Reflection disabled: API token is empty")
                    self.enable_reflection = False
                    return

                self.reflector = Reflector(
                    llm_model=llm_model,
                    llm_api_url=llm_api_url,
                    api_key=llm_api_token,
                    threshold=quality_threshold,
                    max_iterations=max_iterations,
                )

                if not self.reflector.evaluator:
                    logger.warning("Reflection disabled: Evaluator not initialized")
                    self.enable_reflection = False
                    return

                logger.info(
                    f"Reflector initialized: threshold={quality_threshold}, "
                    f"max_iterations={max_iterations}, "
                    f"dspy_enabled={self.reflector.use_dspy}"
                )
            except Exception as e:
                logger.warning(f"Reflector initialization failed: {e}", exc_info=True)
                self.enable_reflection = False

    async def reflect_on_answer(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        ctx: Optional[Context] = None,
    ) -> tuple[str, Optional[Dict]]:
        """Reflect on answer quality and optionally refine."""
        if not self.enable_reflection or not self.reflector:
            return answer, None

        try:
            if ctx:
                await ctx.info("Starting answer quality reflection...")

            # Perform initial reflection
            reflection = await self.reflector.reflect(
                query=query,
                answer=answer,
                sources=sources,
                attempt=1,
            )

            if reflection is None:
                logger.warning(
                    "Initial reflection returned None, using original answer"
                )
                if ctx:
                    await ctx.warning(
                        "Reflection evaluation failed, using original answer"
                    )
                return answer, {
                    "enabled": True,
                    "error": "Reflection evaluation returned None",
                    "quality_score": None,
                    "passed": False,
                }

            reflections_history = [reflection]

            if ctx:
                await ctx.info(
                    f"Quality score: {reflection.overall_score:.2f} "
                    f"(threshold: {reflection.threshold})"
                )

            current_answer = answer
            attempt = 1

            while not reflection.passed and attempt < self.reflector.max_iterations:
                attempt += 1

                if not self.reflector.should_refine(reflection, attempt):
                    if ctx:
                        await ctx.warning(f"Refinement not viable at attempt {attempt}")
                    break

                if ctx:
                    await ctx.info(f"Attempting refinement (attempt {attempt})...")

                # Note: Simplified version without full refinement plan generation
                try:
                    if ctx:
                        await ctx.info(f"Re-evaluating answer (attempt {attempt})...")
                        await ctx.warning(
                            "Full refinement with re-search not yet implemented. "
                            "Re-evaluating existing answer."
                        )

                    # Re-evaluate with refinement context added
                    reflection = await self.reflector.reflect(
                        query=query,
                        answer=current_answer,
                        sources=sources,
                        attempt=attempt,
                    )

                    if reflection is None:
                        logger.warning(f"Reflection attempt {attempt} returned None")
                        if ctx:
                            await ctx.warning(
                                f"Reflection evaluation failed at attempt {attempt}"
                            )
                        break

                    reflections_history.append(reflection)

                    if ctx:
                        await ctx.info(
                            f"Updated quality score: {reflection.overall_score:.2f}"
                        )

                except Exception as e:
                    logger.error(
                        f"Refinement attempt {attempt} failed: {e}", exc_info=True
                    )
                    if ctx:
                        await ctx.error(f"Refinement error: {e}")
                    break

            if not reflections_history:
                logger.warning("No reflection history available")
                return answer, {
                    "enabled": True,
                    "error": "No reflection history available",
                    "quality_score": None,
                    "passed": False,
                }

            final_reflection = reflections_history[-1]

            if final_reflection is None:
                logger.warning("Final reflection is None")
                return answer, {
                    "enabled": True,
                    "error": "Final reflection is None",
                    "quality_score": None,
                    "passed": False,
                }

            metadata = {
                "enabled": True,
                "quality_score": final_reflection.overall_score,
                "passed": final_reflection.passed,
                "total_attempts": len(reflections_history),
                "scores": final_reflection.scores.to_dict(),
                "feedback": final_reflection.feedback,
            }

            if len(reflections_history) > 1:
                initial_reflection = reflections_history[0]
                if initial_reflection is not None:
                    initial_score = initial_reflection.overall_score
                    improvement = final_reflection.overall_score - initial_score
                    metadata["improvement"] = improvement
                    metadata["initial_score"] = initial_score

            if not final_reflection.passed and ctx:
                await ctx.warning(
                    f"Answer quality below threshold after {len(reflections_history)} attempts. "
                    f"Score: {final_reflection.overall_score:.2f}"
                )
            elif ctx:
                await ctx.info(
                    f"Answer quality meets threshold: {final_reflection.overall_score:.2f}"
                )

            return current_answer, metadata

        except Exception as e:
            logger.error(f"Reflection failed: {e}", exc_info=True)
            if ctx:
                await ctx.error(f"Reflection error: {e}")
            return answer, {
                "enabled": True,
                "error": str(e),
                "quality_score": None,
                "passed": False,  # Mark as not passed on reflection errors
            }


def parse_reflection_params(
    enable_reflection: Optional[bool] = None,
    reflection_threshold: Optional[float] = None,
    max_reflection_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parse and validate reflection parameters.

    Args:
        enable_reflection: Enable/disable reflection
        reflection_threshold: Quality threshold (0-1)
        max_reflection_iterations: Max refinement attempts

    Returns:
        Validated parameters dictionary
    """
    params = {}

    # Enable/disable
    if enable_reflection is not None:
        params["enable_reflection"] = bool(enable_reflection)
    else:
        params["enable_reflection"] = ReflectorDefaults.ENABLED.value

    # Quality threshold
    if reflection_threshold is not None:
        threshold = float(reflection_threshold)
        if not 0.0 <= threshold <= 1.0:
            logger.warning(
                f"Invalid reflection threshold {threshold}, "
                f"using default {ReflectorDefaults.QUALITY_THRESHOLD.value}"
            )
            params["quality_threshold"] = ReflectorDefaults.QUALITY_THRESHOLD.value
        else:
            params["quality_threshold"] = threshold
    else:
        params["quality_threshold"] = ReflectorDefaults.QUALITY_THRESHOLD.value

    # Max iterations
    if max_reflection_iterations is not None:
        max_iter = int(max_reflection_iterations)
        if max_iter < 1:
            logger.warning(
                f"Invalid max_reflection_iterations {max_iter}, "
                f"using default {ReflectorDefaults.MAX_ANSWER_ITERATIONS.value}"
            )
            params["max_iterations"] = ReflectorDefaults.MAX_ANSWER_ITERATIONS.value
        else:
            params["max_iterations"] = min(max_iter, 5)
    else:
        params["max_iterations"] = ReflectorDefaults.MAX_ANSWER_ITERATIONS.value

    return params
