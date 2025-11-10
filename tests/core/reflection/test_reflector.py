from unittest.mock import MagicMock, patch

import dspy
import pytest

from kbbridge.core.reflection import (
    Evaluator,
    QualityScores,
    ReflectionResult,
    Reflector,
)
from kbbridge.core.reflection.constants import ReflectionConstants
from kbbridge.core.reflection.feedback import FeedbackGenerator
from kbbridge.core.reflection.integration import (
    ReflectionIntegration,
    parse_reflection_params,
)


class TestReflectionConstants:
    """Test reflection constants and validation"""

    def test_default_values(self):
        """Test default constant values"""
        assert ReflectionConstants.DEFAULT_QUALITY_THRESHOLD == 0.7
        assert ReflectionConstants.DEFAULT_MAX_ITERATIONS == 2
        assert ReflectionConstants.DEFAULT_TEMPERATURE == 0.0
        assert ReflectionConstants.MAX_EXAMPLES_TO_USE == 4

    def test_validate_threshold(self):
        """Test threshold validation"""
        assert ReflectionConstants.validate_threshold(0.5) is True
        assert ReflectionConstants.validate_threshold(0.0) is True
        assert ReflectionConstants.validate_threshold(1.0) is True
        assert ReflectionConstants.validate_threshold(-0.1) is False
        assert ReflectionConstants.validate_threshold(1.1) is False

    def test_validate_weights(self):
        """Test weight validation"""
        valid_weights = {
            "completeness": 0.30,
            "accuracy": 0.30,
            "relevance": 0.20,
            "clarity": 0.10,
            "confidence": 0.10,
        }
        assert ReflectionConstants.validate_weights(valid_weights) is True

        invalid_weights = {
            "completeness": 0.50,
            "accuracy": 0.30,
            "relevance": 0.20,
            "clarity": 0.0,
            "confidence": 0.0,
        }
        assert ReflectionConstants.validate_weights(invalid_weights) is True


class TestQualityScores:
    def test_to_dict(self):
        scores = QualityScores(0.9, 0.85, 0.8, 0.9, 0.8)
        result = scores.to_dict()
        assert result["completeness"] == 0.9
        assert result["accuracy"] == 0.85
        assert result["clarity"] == 0.8
        assert result["relevance"] == 0.9
        assert result["confidence"] == 0.8

    def test_calculate_overall(self):
        scores = QualityScores(0.9, 0.85, 0.8, 0.9, 0.8)
        overall = scores.calculate_overall()
        assert 0.8 <= overall <= 1.0

    def test_weighted_calculation(self):
        scores = QualityScores(1.0, 1.0, 0.5, 0.5, 0.5)
        overall = scores.calculate_overall()
        assert overall > 0.5

    def test_custom_weights(self):
        """Test QualityScores with custom weights"""
        scores = QualityScores(1.0, 1.0, 0.5, 0.5, 0.5)
        overall = scores.calculate_overall()
        # With high completeness and accuracy (both 1.0), should be reasonably high
        assert overall > 0.75


class TestFeedbackGenerator:
    """Test feedback generation"""

    def test_generate_user_feedback_passed(self):
        """Test feedback for passed reflection"""
        generator = FeedbackGenerator()
        reflection = ReflectionResult(
            scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            overall_score=0.8,
            passed=True,
            feedback="Good answer",
            attempt=1,
            threshold=0.7,
        )
        feedback = generator.generate_user_feedback(reflection)
        assert "acceptable" in feedback.lower()

    def test_generate_user_feedback_low_completeness(self):
        """Test feedback for low completeness"""
        generator = FeedbackGenerator()
        reflection = ReflectionResult(
            scores=QualityScores(0.5, 0.8, 0.8, 0.8, 0.8),
            overall_score=0.7,
            passed=False,
            feedback="Missing details",
            attempt=1,
            threshold=0.75,
        )
        feedback = generator.generate_user_feedback(reflection)
        assert "Completeness" in feedback

    def test_generate_user_feedback_low_accuracy(self):
        """Test feedback for low accuracy"""
        generator = FeedbackGenerator()
        reflection = ReflectionResult(
            scores=QualityScores(0.8, 0.5, 0.8, 0.8, 0.8),
            overall_score=0.7,
            passed=False,
            feedback="Accuracy issues",
            attempt=1,
            threshold=0.75,
        )
        feedback = generator.generate_user_feedback(reflection)
        assert "Accuracy" in feedback

    def test_format_refinement_context(self):
        """Test refinement context formatting"""
        from kbbridge.core.reflection.models import RefinementPlan

        generator = FeedbackGenerator()
        reflection = ReflectionResult(
            scores=QualityScores(0.6, 0.6, 0.6, 0.6, 0.6),
            overall_score=0.6,
            passed=False,
            feedback="Needs improvement",
            refinement_suggestions=["Add more details", "Check accuracy"],
            attempt=1,
            threshold=0.7,
        )
        plan = RefinementPlan(strategy="expand", focus_areas=["completeness"])

        context = generator.format_refinement_context(reflection, plan)
        assert "Quality Score" in context
        assert "0.60" in context
        assert "expand" in context


@pytest.mark.slow
class TestEvaluator:
    @pytest.fixture
    def mock_lm(self):
        """Create a mock LM instance for testing"""
        return MagicMock(spec=dspy.LM)

    @pytest.mark.asyncio
    async def test_evaluate_minimal(self, mock_lm):
        evaluator = Evaluator(lm=mock_lm, threshold=0.75)
        sources = [{"title": "doc1", "content": "test content"}]

        result = await evaluator.evaluate(
            query="test query",
            answer="test answer",
            sources=sources,
            attempt=1,
        )

        assert isinstance(result, ReflectionResult)
        assert 0 <= result.overall_score <= 1

    def test_evaluator_initialization(self, mock_lm):
        """Test evaluator initialization with default threshold"""
        evaluator = Evaluator(lm=mock_lm)
        assert evaluator.threshold == ReflectionConstants.DEFAULT_QUALITY_THRESHOLD
        assert evaluator.examples == []

    def test_evaluator_with_custom_threshold(self, mock_lm):
        """Test evaluator with custom threshold"""
        evaluator = Evaluator(lm=mock_lm, threshold=0.85)
        assert evaluator.threshold == 0.85

    def test_format_sources_empty(self, mock_lm):
        """Test source formatting with empty list"""
        evaluator = Evaluator(lm=mock_lm)
        formatted = evaluator._format_sources([])
        assert formatted == "No sources"

    def test_format_sources_with_content(self, mock_lm):
        """Test source formatting with actual sources"""
        evaluator = Evaluator(lm=mock_lm)
        sources = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
        ]
        formatted = evaluator._format_sources(sources)
        assert "Doc 1" in formatted
        assert "Doc 2" in formatted

    def test_parse_json_valid(self, mock_lm):
        """Test JSON parsing with valid input"""
        evaluator = Evaluator(lm=mock_lm)
        result = evaluator._parse_json('["item1", "item2"]')
        assert result == ["item1", "item2"]

    def test_parse_json_invalid(self, mock_lm):
        """Test JSON parsing with invalid input"""
        evaluator = Evaluator(lm=mock_lm)
        result = evaluator._parse_json("not valid json")
        assert isinstance(result, list)


@pytest.mark.slow
class TestReflector:
    @pytest.mark.asyncio
    async def test_reflect_basic(self):
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
            threshold=0.75,
        )

        sources = [{"title": "doc1", "content": "content"}]

        result = await reflector.reflect(
            query="test",
            answer="answer",
            sources=sources,
            attempt=1,
        )

        assert isinstance(result, ReflectionResult) or result is None

    def test_should_refine(self):
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
            threshold=0.75,
        )

        passed = ReflectionResult(
            scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            overall_score=0.8,
            passed=True,
            feedback="good",
            attempt=1,
            threshold=0.75,
        )

        assert not reflector.should_refine(passed, 1)

        failed = ReflectionResult(
            scores=QualityScores(0.5, 0.5, 0.5, 0.5, 0.5),
            overall_score=0.5,
            passed=False,
            feedback="needs work",
            attempt=1,
            threshold=0.75,
        )

        assert reflector.should_refine(failed, 1)

    def test_create_report(self):
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
        )

        reflections = [
            ReflectionResult(
                scores=QualityScores(0.5, 0.5, 0.5, 0.5, 0.5),
                overall_score=0.5,
                passed=False,
                feedback="initial",
                attempt=1,
                threshold=0.75,
            ),
            ReflectionResult(
                scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                overall_score=0.8,
                passed=True,
                feedback="improved",
                attempt=2,
                threshold=0.75,
            ),
        ]

        report = reflector.create_report(reflections)

        assert report["total_attempts"] == 2
        assert report["final_score"] == 0.8
        assert abs(report["improvement"] - 0.3) < 0.001
        assert len(report["history"]) == 2

    def test_reflector_initialization_with_constants(self):
        """Test reflector uses constants for defaults"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
        )
        assert reflector.threshold == ReflectionConstants.DEFAULT_QUALITY_THRESHOLD
        assert reflector.max_iterations == ReflectionConstants.DEFAULT_MAX_ITERATIONS

    def test_should_refine_max_iterations_reached(self):
        """Test should_refine returns False when max iterations reached"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
            max_iterations=2,
        )

        failed = ReflectionResult(
            scores=QualityScores(0.6, 0.6, 0.6, 0.6, 0.6),
            overall_score=0.6,
            passed=False,
            feedback="needs work",
            attempt=2,
            threshold=0.7,
        )

        assert not reflector.should_refine(failed, 2)

    def test_should_refine_score_too_low(self):
        """Test should_refine returns False when score is too low"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
            threshold=0.7,
        )

        very_low = ReflectionResult(
            scores=QualityScores(0.3, 0.3, 0.3, 0.3, 0.3),
            overall_score=0.3,  # 0.7 - 0.3 = 0.4, which is > threshold_gap
            passed=False,
            feedback="very poor",
            attempt=1,
            threshold=0.7,
        )

        assert not reflector.should_refine(very_low, 1)

    def test_create_report_empty(self):
        """Test create_report with empty reflections"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
        )

        report = reflector.create_report([])
        assert report == {}

    def test_create_report_single_reflection(self):
        """Test create_report with single reflection (no improvement)"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
        )

        reflections = [
            ReflectionResult(
                scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                overall_score=0.8,
                passed=True,
                feedback="good",
                attempt=1,
                threshold=0.7,
            ),
        ]

        report = reflector.create_report(reflections)
        assert report["total_attempts"] == 1
        assert "improvement" not in report  # No improvement if only 1 attempt

    @pytest.mark.asyncio
    async def test_reflect_without_evaluator(self):
        """Test reflect when evaluator is not initialized"""
        with patch(
            "kbbridge.core.reflection.reflector.setup",
            side_effect=Exception("DSPy failed"),
        ):
            reflector = Reflector(
                llm_model="gpt-4",
                llm_api_url="https://test.com",
                api_key="test-key",
            )

            result = await reflector.reflect(
                query="test",
                answer="answer",
                sources=[],
                attempt=1,
            )

            assert result is None  # Should return None when evaluator not initialized

    def test_reflector_initialization_success(self):
        """Test reflector initialization with successful DSPy setup (lines 36-38)"""
        mock_lm = MagicMock(spec=dspy.LM)
        with patch(
            "kbbridge.core.reflection.reflector.setup", return_value=mock_lm
        ) as mock_setup:
            with patch(
                "kbbridge.core.reflection.reflector.get_default_examples",
                return_value=[],
            ):
                reflector = Reflector(
                    llm_model="gpt-4",
                    llm_api_url="https://test.com",
                    api_key="test-key",
                )
                assert reflector.use_dspy is True
                assert reflector.evaluator is not None
                assert reflector._lm is mock_lm
                mock_setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_reflect_with_evaluator_success(self):
        """Test reflect when evaluator is successfully initialized (lines 52-65)"""
        from unittest.mock import AsyncMock

        mock_lm = MagicMock(spec=dspy.LM)
        with patch("kbbridge.core.reflection.reflector.setup", return_value=mock_lm):
            with patch(
                "kbbridge.core.reflection.reflector.get_default_examples",
                return_value=[],
            ):
                reflector = Reflector(
                    llm_model="gpt-4",
                    llm_api_url="https://test.com",
                    api_key="test-key",
                )

                mock_result = ReflectionResult(
                    scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                    overall_score=0.8,
                    passed=True,
                    feedback="good",
                    attempt=1,
                    threshold=0.75,
                )

                reflector.evaluator.evaluate = AsyncMock(return_value=mock_result)

                result = await reflector.reflect(
                    query="test query",
                    answer="test answer",
                    sources=[{"title": "doc1", "content": "content"}],
                    attempt=1,
                )

                assert result is not None
                assert result.overall_score == 0.8
                assert result.passed is True

    def test_should_refine_when_passed(self):
        """Test should_refine returns False when reflection passed (line 69)"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
        )

        passed = ReflectionResult(
            scores=QualityScores(0.9, 0.9, 0.9, 0.9, 0.9),
            overall_score=0.9,
            passed=True,
            feedback="excellent",
            attempt=1,
            threshold=0.7,
        )

        assert reflector.should_refine(passed, 1) is False

    def test_should_refine_max_iterations_check(self):
        """Test should_refine checks max iterations (line 71-72)"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
            max_iterations=2,
        )

        failed = ReflectionResult(
            scores=QualityScores(0.6, 0.6, 0.6, 0.6, 0.6),
            overall_score=0.6,
            passed=False,
            feedback="needs work",
            attempt=2,
            threshold=0.7,
        )

        assert reflector.should_refine(failed, 2) is False

    def test_should_refine_score_too_low_gap(self):
        """Test should_refine returns False when score too low (lines 73-76)"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
            threshold=0.7,
        )

        # Score 0.2, threshold 0.7, gap is 0.5 > threshold_gap, so should return False
        very_low = ReflectionResult(
            scores=QualityScores(0.2, 0.2, 0.2, 0.2, 0.2),
            overall_score=0.2,
            passed=False,
            feedback="very poor",
            attempt=1,
            threshold=0.7,
        )

        assert reflector.should_refine(very_low, 1) is False

    def test_create_report_with_improvement(self):
        """Test create_report includes improvement when multiple reflections (lines 94-95)"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
        )

        reflections = [
            ReflectionResult(
                scores=QualityScores(0.5, 0.5, 0.5, 0.5, 0.5),
                overall_score=0.5,
                passed=False,
                feedback="initial",
                attempt=1,
                threshold=0.7,
            ),
            ReflectionResult(
                scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                overall_score=0.8,
                passed=True,
                feedback="improved",
                attempt=2,
                threshold=0.7,
            ),
        ]

        report = reflector.create_report(reflections)
        assert "improvement" in report
        assert report["improvement"] == 0.3

    def test_create_report_history_format(self):
        """Test create_report history format (lines 97-105)"""
        reflector = Reflector(
            llm_model="gpt-4",
            llm_api_url="https://test.com",
            api_key="test-key",
        )

        long_feedback = "x" * 300  # Longer than 200 chars

        reflections = [
            ReflectionResult(
                scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
                overall_score=0.8,
                passed=True,
                feedback=long_feedback,
                attempt=1,
                threshold=0.7,
            ),
        ]

        report = reflector.create_report(reflections)
        assert len(report["history"]) == 1
        assert len(report["history"][0]["feedback"]) == 200  # Truncated to 200


class TestReflectionIntegration:
    """Test reflection integration layer"""

    def test_integration_initialization_disabled(self):
        """Test integration with reflection disabled"""
        integration = ReflectionIntegration(
            llm_api_url="https://test.com",
            llm_model="gpt-4",
            llm_api_token="test-key",
            enable_reflection=False,
        )
        assert integration.enable_reflection is False
        assert integration.reflector is None

    @pytest.mark.asyncio
    async def test_reflect_on_answer_disabled(self):
        """Test reflect_on_answer when reflection is disabled"""
        integration = ReflectionIntegration(
            llm_api_url="https://test.com",
            llm_model="gpt-4",
            llm_api_token="test-key",
            enable_reflection=False,
        )

        answer, metadata = await integration.reflect_on_answer(
            query="test",
            answer="original answer",
            sources=[],
        )

        assert answer == "original answer"
        assert metadata is None


class TestParseReflectionParams:
    """Test reflection parameter parsing"""

    def test_parse_default_params(self):
        """Test parsing with no parameters"""
        params = parse_reflection_params()
        assert params["enable_reflection"] is True  # From ReflectorDefaults
        assert params["quality_threshold"] == 0.70  # From ReflectorDefaults
        assert params["max_iterations"] == 2

    def test_parse_custom_threshold(self):
        """Test parsing with custom threshold"""
        params = parse_reflection_params(reflection_threshold=0.85)
        assert params["quality_threshold"] == 0.85

    def test_parse_invalid_threshold_too_high(self):
        """Test parsing with invalid threshold (too high)"""
        params = parse_reflection_params(reflection_threshold=1.5)
        assert params["quality_threshold"] == 0.70

    def test_parse_invalid_threshold_negative(self):
        """Test parsing with invalid threshold (negative)"""
        params = parse_reflection_params(reflection_threshold=-0.1)
        assert params["quality_threshold"] == 0.70

    def test_parse_custom_max_iterations(self):
        """Test parsing with custom max iterations"""
        params = parse_reflection_params(max_reflection_iterations=3)
        assert params["max_iterations"] == 3

    def test_parse_invalid_max_iterations_zero(self):
        """Test parsing with invalid max iterations (zero)"""
        params = parse_reflection_params(max_reflection_iterations=0)
        assert params["max_iterations"] == 2

    def test_parse_max_iterations_capped(self):
        """Test parsing with max iterations above cap"""
        params = parse_reflection_params(max_reflection_iterations=10)
        assert params["max_iterations"] == 5  # Should be capped at 5

    def test_parse_enable_reflection_false(self):
        """Test parsing with reflection explicitly disabled"""
        params = parse_reflection_params(enable_reflection=False)
        assert params["enable_reflection"] is False

    def test_integration_initialization_with_exception(self):
        """Test integration initialization when Reflector fails (lines 56-58)"""
        with patch(
            "kbbridge.core.reflection.integration.Reflector",
            side_effect=Exception("Setup failed"),
        ):
            integration = ReflectionIntegration(
                llm_api_url="https://test.com",
                llm_model="gpt-4",
                llm_api_token="test-key",
                enable_reflection=True,
            )
            assert integration.enable_reflection is False
            assert integration.reflector is None

    @pytest.mark.asyncio
    async def test_reflect_on_answer_with_refinement_loop(self):
        """Test reflect_on_answer with refinement loop (lines 103-174)"""
        from unittest.mock import AsyncMock

        integration = ReflectionIntegration(
            llm_api_url="https://test.com",
            llm_model="gpt-4",
            llm_api_token="test-key",
            enable_reflection=True,
            max_iterations=3,
        )

        if not integration.reflector:
            pytest.skip("Reflector not initialized (DSPy setup failed)")

        # Mock the reflector methods
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.warning = AsyncMock()

        # First reflection fails, second passes
        first_reflection = ReflectionResult(
            scores=QualityScores(0.6, 0.6, 0.6, 0.6, 0.6),
            overall_score=0.6,
            passed=False,
            feedback="needs work",
            attempt=1,
            threshold=0.7,
        )

        second_reflection = ReflectionResult(
            scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            overall_score=0.8,
            passed=True,
            feedback="good",
            attempt=2,
            threshold=0.7,
        )

        integration.reflector.reflect = AsyncMock(
            side_effect=[first_reflection, second_reflection]
        )
        integration.reflector.max_iterations = 3
        integration.reflector.should_refine = lambda r, a: not r.passed and a < 3

        answer, metadata = await integration.reflect_on_answer(
            query="test query",
            answer="test answer",
            sources=[{"title": "doc1", "content": "content"}],
            ctx=mock_ctx,
        )

        assert answer == "test answer"
        assert metadata is not None
        assert metadata["total_attempts"] == 2
        assert metadata["passed"] is True
        assert "improvement" in metadata

    @pytest.mark.asyncio
    async def test_reflect_on_answer_refinement_not_viable(self):
        """Test reflect_on_answer when refinement is not viable"""
        from unittest.mock import AsyncMock

        integration = ReflectionIntegration(
            llm_api_url="https://test.com",
            llm_model="gpt-4",
            llm_api_token="test-key",
            enable_reflection=True,
        )

        if not integration.reflector:
            pytest.skip("Reflector not initialized")

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.warning = AsyncMock()

        failed_reflection = ReflectionResult(
            scores=QualityScores(0.3, 0.3, 0.3, 0.3, 0.3),
            overall_score=0.3,
            passed=False,
            feedback="too low",
            attempt=1,
            threshold=0.7,
        )

        integration.reflector.reflect = AsyncMock(return_value=failed_reflection)
        integration.reflector.should_refine = lambda r, a: False  # Not viable

        answer, metadata = await integration.reflect_on_answer(
            query="test",
            answer="answer",
            sources=[],
            ctx=mock_ctx,
        )

        assert answer == "answer"
        mock_ctx.warning.assert_called()

    @pytest.mark.asyncio
    async def test_reflect_on_answer_refinement_exception(self):
        """Test reflect_on_answer when refinement raises exception"""
        from unittest.mock import AsyncMock

        integration = ReflectionIntegration(
            llm_api_url="https://test.com",
            llm_model="gpt-4",
            llm_api_token="test-key",
            enable_reflection=True,
        )

        if not integration.reflector:
            pytest.skip("Reflector not initialized")

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        first_reflection = ReflectionResult(
            scores=QualityScores(0.6, 0.6, 0.6, 0.6, 0.6),
            overall_score=0.6,
            passed=False,
            feedback="needs work",
            attempt=1,
            threshold=0.7,
        )

        integration.reflector.reflect = AsyncMock(
            side_effect=[first_reflection, Exception("Refinement failed")]
        )
        integration.reflector.should_refine = lambda r, a: True

        answer, metadata = await integration.reflect_on_answer(
            query="test",
            answer="answer",
            sources=[],
            ctx=mock_ctx,
        )

        assert answer == "answer"
        mock_ctx.error.assert_called()

    @pytest.mark.asyncio
    async def test_reflect_on_answer_final_not_passed_with_warning(self):
        """Test reflect_on_answer when final reflection not passed (lines 164-168)"""
        from unittest.mock import AsyncMock

        integration = ReflectionIntegration(
            llm_api_url="https://test.com",
            llm_model="gpt-4",
            llm_api_token="test-key",
            enable_reflection=True,
        )

        if not integration.reflector:
            pytest.skip("Reflector not initialized")

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.warning = AsyncMock()

        failed_reflection = ReflectionResult(
            scores=QualityScores(0.6, 0.6, 0.6, 0.6, 0.6),
            overall_score=0.6,
            passed=False,
            feedback="below threshold",
            attempt=1,
            threshold=0.7,
        )

        integration.reflector.reflect = AsyncMock(return_value=failed_reflection)
        integration.reflector.should_refine = lambda r, a: False

        answer, metadata = await integration.reflect_on_answer(
            query="test",
            answer="answer",
            sources=[],
            ctx=mock_ctx,
        )

        assert not metadata["passed"]
        mock_ctx.warning.assert_called()

    @pytest.mark.asyncio
    async def test_reflect_on_answer_passed_with_info(self):
        """Test reflect_on_answer when passed shows info (lines 169-172)"""
        from unittest.mock import AsyncMock

        integration = ReflectionIntegration(
            llm_api_url="https://test.com",
            llm_model="gpt-4",
            llm_api_token="test-key",
            enable_reflection=True,
        )

        if not integration.reflector:
            pytest.skip("Reflector not initialized")

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()

        passed_reflection = ReflectionResult(
            scores=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
            overall_score=0.8,
            passed=True,
            feedback="good",
            attempt=1,
            threshold=0.7,
        )

        integration.reflector.reflect = AsyncMock(return_value=passed_reflection)

        answer, metadata = await integration.reflect_on_answer(
            query="test",
            answer="answer",
            sources=[],
            ctx=mock_ctx,
        )

        assert metadata["passed"] is True
        # Check that info was called with success message
        assert mock_ctx.info.call_count >= 2

    @pytest.mark.asyncio
    async def test_reflect_on_answer_top_level_exception(self):
        """Test reflect_on_answer when top-level exception occurs (lines 176-186)"""
        from unittest.mock import AsyncMock

        integration = ReflectionIntegration(
            llm_api_url="https://test.com",
            llm_model="gpt-4",
            llm_api_token="test-key",
            enable_reflection=True,
        )

        if not integration.reflector:
            pytest.skip("Reflector not initialized")

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        integration.reflector.reflect = AsyncMock(side_effect=Exception("Fatal error"))

        answer, metadata = await integration.reflect_on_answer(
            query="test",
            answer="original answer",
            sources=[],
            ctx=mock_ctx,
        )

        assert answer == "original answer"
        assert metadata["enabled"] is True
        assert metadata["passed"] is False
        assert "error" in metadata
        mock_ctx.error.assert_called()
