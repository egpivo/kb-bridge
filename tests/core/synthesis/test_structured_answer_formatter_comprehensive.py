"""
Comprehensive tests for Structured Answer Formatter module to improve code coverage.

This module provides extensive test coverage for the structured answer formatting capabilities.
"""

from unittest.mock import Mock, patch

from kbbridge.core.synthesis.answer_formatter import StructuredAnswerFormatter
from kbbridge.core.synthesis.constants import AnswerExtractorDefaults


class TestStructuredAnswerFormatterInit:
    """Test StructuredAnswerFormatter initialization"""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters"""
        formatter = StructuredAnswerFormatter(
            llm_api_url="https://api.test.com",
            llm_model="gpt-4",
            llm_api_token="test-token",
            llm_temperature=0.5,
            llm_timeout=30,
            max_tokens=256,
        )

        assert formatter.llm_api_url == "https://api.test.com"
        assert formatter.llm_model == "gpt-4"
        assert formatter.llm_api_token == "test-token"
        assert formatter.llm_temperature == 0.5
        assert formatter.llm_timeout == 30
        assert formatter.max_tokens == 256

    def test_init_with_minimal_parameters(self):
        """Test initialization with minimal parameters (using defaults)"""
        formatter = StructuredAnswerFormatter(
            llm_api_url="https://api.test.com", llm_model="gpt-4"
        )

        assert formatter.llm_api_url == "https://api.test.com"
        assert formatter.llm_model == "gpt-4"
        assert formatter.llm_api_token is None
        assert formatter.llm_temperature == AnswerExtractorDefaults.TEMPERATURE.value
        assert formatter.llm_timeout == AnswerExtractorDefaults.TIMEOUT_SECONDS.value
        assert formatter.max_tokens == AnswerExtractorDefaults.MAX_TOKENS.value

    def test_init_with_none_values(self):
        """Test initialization with explicit None values (should use defaults)"""
        formatter = StructuredAnswerFormatter(
            llm_api_url="https://api.test.com",
            llm_model="gpt-4",
            llm_api_token=None,
            llm_temperature=None,
            llm_timeout=None,
            max_tokens=None,
        )

        assert formatter.llm_api_token is None
        assert formatter.llm_temperature == AnswerExtractorDefaults.TEMPERATURE.value
        assert formatter.llm_timeout == AnswerExtractorDefaults.TIMEOUT_SECONDS.value
        assert formatter.max_tokens == AnswerExtractorDefaults.MAX_TOKENS.value

    def test_init_with_zero_values(self):
        """Test initialization with zero values (should not use defaults)"""
        formatter = StructuredAnswerFormatter(
            llm_api_url="https://api.test.com",
            llm_model="gpt-4",
            llm_temperature=0.0,
            llm_timeout=0,
            max_tokens=0,
        )

        assert formatter.llm_temperature == 0.0
        assert formatter.llm_timeout == 0
        assert formatter.max_tokens == 0


# TestCallLLMAPI removed - DSPy handles API calls automatically
# TestParseStructuredResponse removed - DSPy handles parsing automatically


class TestFormatStructuredAnswer:
    """Test format_structured_answer method"""

    def test_format_structured_answer_success(self):
        """Test successful structured answer formatting"""
        formatter = StructuredAnswerFormatter(
            "https://api.test.com", "gpt-4", "test-token"
        )

        # Mock DSPy predictor
        with patch.object(formatter, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.answer = "Test answer"
            mock_result.sources = [Mock(source="test.pdf", relevance="high")]
            mock_result.total_sources = 1
            mock_result.confidence = "high"
            mock_predictor.return_value = mock_result

            candidates = [
                {
                    "success": True,
                    "answer": "Test answer",
                    "source": "naive",
                    "dataset_id": "test-dataset",
                }
            ]

            result = formatter.format_structured_answer("Test query", candidates)

            assert result["success"] is True
            assert result["answer"] == "Test answer"
            assert result["total_sources"] == 1
            assert result["confidence"] == "high"
            assert "structured_answer" in result

    def test_format_structured_answer_no_valid_candidates(self):
        """Test formatting with no valid candidates"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")

        candidates = [
            {"success": False, "answer": "Failed answer"},
            {"success": True, "answer": ""},  # Empty answer
        ]

        result = formatter.format_structured_answer("Test query", candidates)

        assert result["success"] is False
        assert result["error"] == "No valid candidates found"
        assert result["candidates_count"] == 0

    def test_format_structured_answer_empty_candidates(self):
        """Test formatting with empty candidates list"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")

        result = formatter.format_structured_answer("Test query", [])

        assert result["success"] is False
        assert result["error"] == "No valid candidates found"

    def test_format_structured_answer_api_failure(self):
        """Test formatting with API failure"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")

        with patch.object(formatter, "predictor") as mock_predictor:
            # Simulate DSPy API error
            mock_predictor.side_effect = Exception("API timeout")

            candidates = [{"success": True, "answer": "Test answer"}]
            result = formatter.format_structured_answer("Test query", candidates)

            assert result["success"] is False
            assert "Structured answer formatting failed" in result["error"]

    def test_format_structured_answer_parse_failure(self):
        """Test formatting with parsing failure (DSPy handles this internally)"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")

        with patch.object(formatter, "predictor") as mock_predictor:
            # DSPy returns minimal result (simulate incomplete parsing)
            mock_result = Mock()
            mock_result.answer = ""  # Empty answer
            mock_result.sources = []  # Empty sources
            mock_result.total_sources = 0
            mock_result.confidence = "low"
            mock_predictor.return_value = mock_result

            candidates = [{"success": True, "answer": "Test answer"}]
            result = formatter.format_structured_answer("Test query", candidates)

            # DSPy will handle parsing internally, so we get a result
            # but with minimal/fallback values
            assert result["success"] is True
            assert result["answer"] == ""
            assert result["total_sources"] == 0

    def test_format_structured_answer_exception(self):
        """Test formatting with general exception"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")

        with patch.object(
            formatter, "predictor", side_effect=Exception("Unexpected error")
        ):
            candidates = [{"success": True, "answer": "Test answer"}]
            result = formatter.format_structured_answer("Test query", candidates)

            assert result["success"] is False
            assert "Structured answer formatting failed" in result["error"]

    def test_format_structured_answer_filters_candidates(self):
        """Test that invalid candidates are filtered out"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")

        with patch.object(formatter, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.answer = "Test answer"
            mock_result.sources = [Mock(source="test.pdf", relevance="high")]
            mock_result.total_sources = 1
            mock_result.confidence = "high"
            mock_predictor.return_value = mock_result

            candidates = [
                {"success": False, "answer": "Failed answer"},  # Will be filtered
                {"success": True, "answer": ""},  # Will be filtered (empty answer)
                {"success": True, "answer": "Valid answer"},  # Will be kept
                {"answer": "No success field"},  # Will be filtered
            ]

            result = formatter.format_structured_answer("Test query", candidates)

            # Should succeed because there's at least one valid candidate
            assert result["success"] is True

            # Check that the predictor was called
            mock_predictor.assert_called_once()


class TestBuildMethods:
    """Test the various _build_* response methods"""

    def test_build_base_response(self):
        """Test _build_base_response method"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")
        candidates = [{"test": "data"}, {"test": "data2"}]

        result = formatter._build_base_response("Test query", candidates)

        assert result["query"] == "Test query"
        assert result["candidates_count"] == 2
        assert result["model_used"] == "gpt-4"
        assert result["tool_type"] == "structured_answer_formatter"

    def test_build_success_response(self):
        """Test _build_success_response method"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")
        structured_data = {
            "answer": "Test answer",
            "sources": [{"source": "test.pdf", "relevance": "high"}],
            "total_sources": 1,
            "confidence": "high",
        }
        candidates = [{"test": "data"}]

        result = formatter._build_success_response(
            structured_data, "Test query", candidates
        )

        assert result["success"] is True
        assert result["structured_answer"] == structured_data
        assert result["answer"] == "Test answer"
        assert result["total_sources"] == 1
        assert result["confidence"] == "high"
        assert result["query"] == "Test query"

    def test_build_success_response_missing_fields(self):
        """Test _build_success_response with missing optional fields"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")
        structured_data = {
            "sources": [{"source": "test.pdf", "relevance": "high"}]
            # Missing answer, total_sources, confidence
        }
        candidates = [{"test": "data"}]

        result = formatter._build_success_response(
            structured_data, "Test query", candidates
        )

        assert result["success"] is True
        assert result["answer"] == ""  # Default value
        assert result["total_sources"] == 0  # Default value
        assert result["confidence"] == "medium"  # Default value

    # test_build_error_response removed - _build_error_response method no longer exists
    # test_build_parse_error_response removed - _build_parse_error_response method no longer exists

    def test_build_no_results_response(self):
        """Test _build_no_results_response method"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")

        result = formatter._build_no_results_response("Test query")

        assert result["success"] is False
        assert result["error"] == "No valid candidates found"
        assert result["query"] == "Test query"
        assert result["candidates_count"] == 0
        assert result["model_used"] == "gpt-4"
        assert result["tool_type"] == "structured_answer_formatter"

    def test_build_exception_response(self):
        """Test _build_exception_response method"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")
        exception = Exception("Test exception")
        candidates = [{"test": "data"}]

        result = formatter._build_exception_response(
            exception, "Test query", candidates
        )

        assert result["success"] is False
        assert "Structured answer formatting failed" in result["error"]
        assert "Test exception" in result["error"]
        assert result["details"] == "An unexpected error occurred during processing"
        assert result["query"] == "Test query"


class TestIntegration:
    """Integration tests for StructuredAnswerFormatter"""

    def test_full_formatting_flow(self):
        """Test complete formatting flow"""

        formatter = StructuredAnswerFormatter(
            llm_api_url="https://api.test.com",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        # Mock DSPy predictor
        with patch.object(formatter, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.answer = "The vacation policy provides 15 days for new employees and 20 days after one year."
            mock_result.sources = [
                Mock(source="dataset1/hr_policies.pdf", relevance="high"),
                Mock(source="dataset1/employee_handbook.pdf", relevance="high"),
            ]
            mock_result.total_sources = 2
            mock_result.confidence = "high"
            mock_predictor.return_value = mock_result

            candidates = [
                {
                    "success": True,
                    "answer": "New employees get 15 vacation days",
                    "source": "advanced",
                    "dataset_id": "dataset1",
                    "file_name": "hr_policies.pdf",
                },
                {
                    "success": True,
                    "answer": "After one year, employees get 20 vacation days",
                    "source": "advanced",
                    "dataset_id": "dataset1",
                    "file_name": "employee_handbook.pdf",
                },
            ]

            result = formatter.format_structured_answer(
                "What is the vacation policy?", candidates
            )

            assert result["success"] is True
            assert "vacation policy" in result["answer"].lower()
            assert result["total_sources"] == 2
            assert result["confidence"] == "high"
            assert len(result["structured_answer"]["sources"]) == 2

    def test_edge_case_candidates(self):
        """Test with various edge case candidates"""
        formatter = StructuredAnswerFormatter("https://api.test.com", "gpt-4")

        # Test with candidates having different combinations of success/answer
        candidates = [
            {"success": True, "answer": "Valid answer 1"},
            {"success": True, "answer": None},  # None answer
            {"success": False, "answer": "Failed answer"},
            {"success": True},  # Missing answer
            {"answer": "No success field"},
            {"success": True, "answer": "   "},  # Whitespace answer
            {"success": True, "answer": "Valid answer 2"},
        ]

        # Should filter to only valid candidates with non-empty answers
        with patch.object(formatter, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.answer = "Combined answer"
            mock_result.sources = [Mock(source="test", relevance="high")]
            mock_result.total_sources = 1
            mock_result.confidence = "medium"
            mock_predictor.return_value = mock_result

            result = formatter.format_structured_answer("Test query", candidates)
            assert result["success"] is True
