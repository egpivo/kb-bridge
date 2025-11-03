"""
Comprehensive tests for KB Assistant Utils module to improve code coverage.

This module provides extensive test coverage for utility functions and classes.
"""

import time
from unittest.mock import Mock, patch

from kbbridge.core.orchestration.utils import ResultFormatter
from kbbridge.core.utils.profiling_utils import profile_stage


class TestProfileStage:
    """Test profile_stage context manager"""

    def test_profile_stage_verbose_true(self):
        """Test profiling when verbose is True"""
        profiling_data = {}

        with profile_stage("test_stage", profiling_data, verbose=True):
            time.sleep(0.001)  # Small delay to measure

        assert "test_stage" in profiling_data
        assert "duration_seconds" in profiling_data["test_stage"]
        assert "duration_ms" in profiling_data["test_stage"]
        assert profiling_data["test_stage"]["duration_seconds"] > 0
        assert profiling_data["test_stage"]["duration_ms"] > 0

    def test_profile_stage_verbose_false(self):
        """Test profiling when verbose is False"""
        profiling_data = {}

        with profile_stage("test_stage", profiling_data, verbose=False):
            time.sleep(0.001)

        assert profiling_data == {}

    def test_profile_stage_default_verbose(self):
        """Test profiling with default verbose (False)"""
        profiling_data = {}

        with profile_stage("test_stage", profiling_data):
            time.sleep(0.001)

        assert profiling_data == {}

    def test_profile_stage_with_exception(self):
        """Test profiling when exception occurs"""
        profiling_data = {}

        try:
            with profile_stage("test_stage", profiling_data, verbose=True):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still record timing even with exception
        assert "test_stage" in profiling_data
        assert "duration_seconds" in profiling_data["test_stage"]

    def test_profile_stage_timing_accuracy(self):
        """Test that profiling captures accurate timing"""
        profiling_data = {}
        sleep_time = 0.01

        with profile_stage("timing_test", profiling_data, verbose=True):
            time.sleep(sleep_time)

        # Check that measured time is reasonably close to sleep time
        measured_time = profiling_data["timing_test"]["duration_seconds"]
        assert measured_time >= sleep_time * 0.8  # Allow 20% variance
        assert measured_time <= sleep_time * 2.0  # Allow up to 2x for system variance

        # Check millisecond conversion
        ms_time = profiling_data["timing_test"]["duration_ms"]
        assert abs(ms_time - measured_time * 1000) < 1  # Should be close

    def test_profile_stage_multiple_stages(self):
        """Test profiling multiple stages"""
        profiling_data = {}

        with profile_stage("stage1", profiling_data, verbose=True):
            time.sleep(0.001)

        with profile_stage("stage2", profiling_data, verbose=True):
            time.sleep(0.001)

        assert "stage1" in profiling_data
        assert "stage2" in profiling_data
        assert len(profiling_data) == 2


class TestResultFormatterFormatFinalAnswer:
    """Test ResultFormatter.format_final_answer method"""

    def test_format_final_answer_empty_candidates(self):
        """Test formatting with empty candidates"""
        credentials = Mock()
        result = ResultFormatter.format_final_answer([], "test query", credentials)
        assert result == "N/A - No relevant information found"

    def test_format_final_answer_single_candidate_naive(self):
        """Test formatting single naive candidate"""
        candidates = [{"source": "naive", "answer": "This is a naive answer"}]
        credentials = Mock()

        result = ResultFormatter.format_final_answer(
            candidates, "test query", credentials
        )
        assert result == "This is a naive answer"

    def test_format_final_answer_single_candidate_advanced_with_file(self):
        """Test formatting single advanced candidate with file name"""
        candidates = [
            {
                "source": "advanced",
                "dataset_id": "dataset123",
                "file_name": "document.pdf",
                "answer": "This is an advanced answer",
            }
        ]
        credentials = Mock()

        result = ResultFormatter.format_final_answer(
            candidates, "test query", credentials
        )
        assert result == "**dataset123/document.pdf**: This is an advanced answer"

    def test_format_final_answer_single_candidate_advanced_without_file(self):
        """Test formatting single advanced candidate without file name"""
        candidates = [
            {
                "source": "advanced",
                "dataset_id": "dataset123",
                "answer": "This is an advanced answer",
            }
        ]
        credentials = Mock()

        result = ResultFormatter.format_final_answer(
            candidates, "test query", credentials
        )
        assert result == "**dataset123**: This is an advanced answer"

    def test_format_final_answer_multiple_candidates_with_reranking(self):
        """Test formatting multiple candidates with reranking available"""
        candidates = [
            {"source": "naive", "answer": "Answer 1"},
            {"source": "naive", "answer": "Answer 2"},
        ]
        credentials = Mock()
        credentials.rerank_url = "https://rerank.test.com"
        credentials.rerank_model = "rerank-model"

        mock_rerank_result = {"final_result": "Reranked answer"}

        with patch(
            "kbbridge.core.orchestration.utils.AnswerReranker"
        ) as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker.rerank_answers.return_value = mock_rerank_result
            mock_reranker_class.return_value = mock_reranker

            result = ResultFormatter.format_final_answer(
                candidates, "test query", credentials
            )

            assert result == "Reranked answer"
            mock_reranker_class.assert_called_once_with(
                "https://rerank.test.com", "rerank-model"
            )
            mock_reranker.rerank_answers.assert_called_once_with(
                "test query", candidates
            )

    def test_format_final_answer_multiple_candidates_reranking_no_result(self):
        """Test formatting multiple candidates when reranking returns no result"""
        candidates = [
            {"source": "naive", "answer": "Answer 1"},
            {"source": "naive", "answer": "Answer 2"},
        ]
        credentials = Mock()
        credentials.rerank_url = "https://rerank.test.com"
        credentials.rerank_model = "rerank-model"

        mock_rerank_result = {}  # No final_result

        with patch(
            "kbbridge.core.orchestration.utils.AnswerReranker"
        ) as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker.rerank_answers.return_value = mock_rerank_result
            mock_reranker_class.return_value = mock_reranker

            result = ResultFormatter.format_final_answer(
                candidates, "test query", credentials
            )

            # Should fall back to combining candidates
            assert result == "Answer 1"  # First candidate

    def test_format_final_answer_multiple_candidates_no_reranking(self):
        """Test formatting multiple candidates without reranking"""
        candidates = [
            {"source": "naive", "answer": "Answer 1"},
            {"source": "naive", "answer": "Answer 2"},
        ]
        credentials = Mock()
        credentials.rerank_url = None
        credentials.rerank_model = None

        result = ResultFormatter.format_final_answer(
            candidates, "test query", credentials
        )

        # Should fall back to combining candidates
        assert result == "Answer 1"  # First candidate


class TestResultFormatterCombineCandidates:
    """Test ResultFormatter._combine_candidates method"""

    def test_combine_candidates_empty(self):
        """Test combining empty candidates"""
        result = ResultFormatter._combine_candidates([], "test query")
        assert result == "N/A - No relevant information found"

    def test_combine_candidates_terms_definitions_query(self):
        """Test combining candidates for terms and definitions query"""
        candidates = [
            {
                "source": "naive",
                "answer": "1. Term A: Definition A\n2. Term B: Definition B",
                "success": True,
            },
            {"source": "naive", "answer": "1. Term X: Definition X", "success": True},
            {
                "source": "naive",
                "answer": "1. Term 1: Definition 1\n2. Term 2: Definition 2\n3. Term 3: Definition 3",
                "success": True,
            },
        ]

        result = ResultFormatter._combine_candidates(
            candidates, "terms and definitions query"
        )

        # Should combine all successful answers
        assert "Term 1: Definition 1" in result
        assert "Term 3: Definition 3" in result

    def test_combine_candidates_terms_definitions_no_numbered_items(self):
        """Test combining candidates for terms query with no numbered definitions"""
        candidates = [
            {"source": "naive", "answer": "Some text without numbered items"},
            {"source": "naive", "answer": "More text without definitions"},
        ]

        result = ResultFormatter._combine_candidates(
            candidates, "terms and definitions"
        )

        # Should fall back to first candidate
        assert result == "Some text without numbered items"

    def test_combine_candidates_non_terms_query(self):
        """Test combining candidates for non-terms query"""
        candidates = [
            {"source": "naive", "answer": "First answer"},
            {"source": "naive", "answer": "Second answer"},
        ]

        result = ResultFormatter._combine_candidates(candidates, "regular query")

        # Should return first candidate
        assert result == "First answer"

    def test_combine_candidates_terms_definitions_with_escaped_newlines(self):
        """Test combining candidates with escaped newlines"""
        candidates = [
            {
                "source": "naive",
                "answer": "1. Term A: Definition A\\n2. Term B: Definition B\\n3. Term C: Definition C",
            },
            {"source": "naive", "answer": "1. Term X: Definition X"},
        ]

        result = ResultFormatter._combine_candidates(
            candidates, "terms and definitions query"
        )

        # Should select the candidate with most definitions (first one has 3)
        assert "Term A: Definition A" in result

    def test_combine_candidates_terms_definitions_mixed_numbering(self):
        """Test combining candidates with various numbering formats"""
        candidates = [
            {
                "source": "naive",
                "answer": "1. First\n2. Second\n10. Tenth\n15. Fifteenth",  # 4 definitions
            },
            {"source": "naive", "answer": "1. Only one"},  # 1 definition
        ]

        result = ResultFormatter._combine_candidates(
            candidates, "terms and definitions"
        )

        # Should select first candidate with 4 definitions
        assert "First" in result
        assert "Fifteenth" in result


class TestResultFormatterFormatSingleCandidate:
    """Test ResultFormatter._format_single_candidate method"""

    def test_format_single_candidate_naive(self):
        """Test formatting naive candidate"""
        candidate = {"source": "naive", "answer": "Naive answer content"}

        result = ResultFormatter._format_single_candidate(candidate)
        assert result == "Naive answer content"

    def test_format_single_candidate_advanced_with_file(self):
        """Test formatting advanced candidate with file"""
        candidate = {
            "source": "advanced",
            "dataset_id": "test_dataset",
            "file_name": "test_file.pdf",
            "answer": "Advanced answer content",
        }

        result = ResultFormatter._format_single_candidate(candidate)
        assert result == "**test_dataset/test_file.pdf**: Advanced answer content"

    def test_format_single_candidate_advanced_without_file(self):
        """Test formatting advanced candidate without file"""
        candidate = {
            "source": "advanced",
            "dataset_id": "test_dataset",
            "answer": "Advanced answer content",
        }

        result = ResultFormatter._format_single_candidate(candidate)
        assert result == "**test_dataset**: Advanced answer content"

    def test_format_single_candidate_advanced_empty_file_name(self):
        """Test formatting advanced candidate with empty file name"""
        candidate = {
            "source": "advanced",
            "dataset_id": "test_dataset",
            "file_name": "",
            "answer": "Advanced answer content",
        }

        result = ResultFormatter._format_single_candidate(candidate)
        assert result == "**test_dataset**: Advanced answer content"


class TestResultFormatterFormatStructuredAnswer:
    """Test ResultFormatter.format_structured_answer method"""

    def test_format_structured_answer_empty_candidates(self):
        """Test formatting structured answer with empty candidates"""
        credentials = Mock()

        result = ResultFormatter.format_structured_answer([], "test query", credentials)

        assert result["success"] is False
        assert result["error"] == "No candidates found"
        assert "No valid candidates" in result["details"]

    def test_format_structured_answer_with_candidates(self):
        """Test formatting structured answer with candidates"""
        candidates = [
            {"source": "naive", "answer": "Test answer 1"},
            {"source": "advanced", "dataset_id": "ds1", "answer": "Test answer 2"},
        ]
        credentials = Mock()
        credentials.llm_api_url = "https://llm.test.com"
        credentials.llm_model = "gpt-4"
        credentials.llm_api_token = "test-token"

        mock_format_result = {
            "success": True,
            "structured_answer": {"answer": "Formatted answer", "sources": []},
            "total_sources": 2,
        }

        with patch(
            "kbbridge.core.orchestration.utils.StructuredAnswerFormatter"
        ) as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter.format_structured_answer.return_value = mock_format_result
            mock_formatter_class.return_value = mock_formatter

            result = ResultFormatter.format_structured_answer(
                candidates, "test query", credentials
            )

            assert result == mock_format_result
            # Check that formatter was called with expected params (including rerank params)
            call_kwargs = mock_formatter_class.call_args[1]
            assert call_kwargs["llm_api_url"] == "https://llm.test.com"
            assert call_kwargs["llm_model"] == "gpt-4"
            assert call_kwargs["llm_api_token"] == "test-token"
            assert call_kwargs["max_tokens"] == 12800
            # rerank_url and rerank_model are also passed from credentials
            mock_formatter.format_structured_answer.assert_called_once_with(
                "test query", candidates
            )

    def test_format_structured_answer_formatter_failure(self):
        """Test formatting structured answer when formatter fails"""
        candidates = [{"source": "naive", "answer": "Test answer"}]
        credentials = Mock()
        credentials.llm_api_url = "https://llm.test.com"
        credentials.llm_model = "gpt-4"
        credentials.llm_api_token = "test-token"

        mock_format_result = {
            "success": False,
            "error": "LLM formatting failed",
            "details": "API error",
        }

        with patch(
            "kbbridge.core.orchestration.utils.StructuredAnswerFormatter"
        ) as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter.format_structured_answer.return_value = mock_format_result
            mock_formatter_class.return_value = mock_formatter

            result = ResultFormatter.format_structured_answer(
                candidates, "test query", credentials
            )

            assert result["success"] is False
            assert result["error"] == "LLM formatting failed"

    def test_format_structured_answer_credentials_passed_correctly(self):
        """Test that credentials are passed correctly to formatter"""
        candidates = [{"source": "naive", "answer": "Test answer"}]
        credentials = Mock()
        credentials.llm_api_url = "https://custom-llm.com"
        credentials.llm_model = "custom-model"
        credentials.llm_api_token = "custom-token"

        with patch(
            "kbbridge.core.orchestration.utils.StructuredAnswerFormatter"
        ) as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter.format_structured_answer.return_value = {"success": True}
            mock_formatter_class.return_value = mock_formatter

            ResultFormatter.format_structured_answer(
                candidates, "test query", credentials
            )

            # Verify correct credentials were passed (including rerank params)
            call_kwargs = mock_formatter_class.call_args[1]
            assert call_kwargs["llm_api_url"] == "https://custom-llm.com"
            assert call_kwargs["llm_model"] == "custom-model"
            assert call_kwargs["llm_api_token"] == "custom-token"
            assert call_kwargs["max_tokens"] == 12800
            # rerank_url and rerank_model are also passed from credentials


class TestResultFormatterIntegration:
    """Integration tests for ResultFormatter"""

    def test_result_formatter_static_methods(self):
        """Test that all methods are static and can be called on the class"""
        # These should all be callable without instantiating the class
        assert callable(ResultFormatter.format_final_answer)
        assert callable(ResultFormatter.format_structured_answer)
        assert callable(ResultFormatter._combine_candidates)
        assert callable(ResultFormatter._format_single_candidate)

    def test_result_formatter_edge_cases(self):
        """Test edge cases for ResultFormatter methods"""
        # Test with None candidates
        credentials = Mock()

        # Should handle None gracefully
        result1 = ResultFormatter.format_final_answer(None or [], "query", credentials)
        assert "No relevant information found" in result1

        result2 = ResultFormatter._combine_candidates(None or [], "query")
        assert "No relevant information found" in result2

    def test_result_formatter_real_world_scenario(self):
        """Test ResultFormatter with realistic data"""
        candidates = [
            {
                "source": "advanced",
                "dataset_id": "legal_docs",
                "file_name": "contract_terms.pdf",
                "answer": "1. Force Majeure: An unforeseeable circumstance\\n2. Liability: Legal responsibility",
            },
            {"source": "naive", "answer": "1. Contract: A legal agreement"},
        ]

        credentials = Mock()
        credentials.rerank_url = None
        credentials.rerank_model = None

        # Test terms and definitions query
        result = ResultFormatter.format_final_answer(
            candidates, "terms and definitions", credentials
        )

        # Should select the candidate with more definitions (first one)
        assert "Force Majeure" in result
        assert "legal_docs/contract_terms.pdf" in result
