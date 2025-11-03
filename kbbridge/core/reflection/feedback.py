from .constants import ReflectionConstants
from .models import RefinementPlan, ReflectionResult


class FeedbackGenerator:
    """Generates user-facing feedback from reflection results."""

    def generate_user_feedback(self, reflection: ReflectionResult) -> str:
        """Generate user-friendly feedback message based on reflection scores."""
        if reflection.passed:
            return "Answer quality is acceptable."

        issues = []

        if reflection.scores.completeness < ReflectionConstants.MINIMUM_SCORE_THRESHOLD:
            issues.append("Completeness could be improved")
        if reflection.scores.accuracy < ReflectionConstants.MINIMUM_SCORE_THRESHOLD:
            issues.append("Accuracy needs verification")
        if reflection.scores.clarity < ReflectionConstants.MINIMUM_SCORE_THRESHOLD:
            issues.append("Clarity could be enhanced")

        if issues:
            return "Answer needs improvement: " + ", ".join(issues)

        return reflection.feedback

    def format_refinement_context(
        self, reflection: ReflectionResult, plan: RefinementPlan
    ) -> str:
        """Format refinement context for retry attempt."""
        context = f"Quality Score: {reflection.overall_score:.2f}\n"
        context += f"Feedback: {reflection.feedback}\n"
        context += f"Strategy: {plan.strategy}\n"

        if reflection.refinement_suggestions:
            context += "Suggestions:\n"
            for suggestion in reflection.refinement_suggestions:
                context += f"- {suggestion}\n"

        return context
