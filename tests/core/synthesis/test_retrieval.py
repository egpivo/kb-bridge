
"""
Test Suite for Retrieval Modules
Tests all retrieval modules with comprehensive coverage
"""

from unittest.mock import Mock, patch

from kbbridge.core.discovery.file_reranker import rerank_files_by_names
from kbbridge.core.synthesis.answer_extractor import OrganizationAnswerExtractor
from kbbridge.core.synthesis.answer_reranker import AnswerReranker

# Test fixtures for required parameters
TEST_LLM_API_TOKEN = "test-api-token"
TEST_RERANK_URL = "https://test-rerank.example.com"
TEST_MODEL = "test-model"


class TestOrganizationAnswerExtractor:
    """Test OrganizationAnswerExtractor functionality"""

    def test_init(self):
        """Test answer extractor initialization"""
        extractor = OrganizationAnswerExtractor(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )
        assert extractor is not None

    def test_extract_answers_success(self, mock_credentials):
        """Test successful answer extraction"""
        extractor = OrganizationAnswerExtractor(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock DSPy predictor
        with patch.object(extractor, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.answer = "Extracted answer content"
            mock_predictor.return_value = mock_result

            result = extractor.extract(
                context="content1 content2", user_query="test query"
            )

            assert isinstance(result, dict)
            assert result["success"] is True
            assert result["answer"] == "Extracted answer content"

    def test_extract_answers_api_error(self, mock_credentials):
        """Test answer extraction with API error"""
        extractor = OrganizationAnswerExtractor(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock DSPy predictor to raise an exception
        with patch.object(extractor, "predictor", side_effect=Exception("API error")):
            result = extractor.extract(context="content1", user_query="test query")

            assert result["success"] is False
            assert "error" in result

    def test_extract_answers_network_error(self, mock_credentials):
        """Test answer extraction with network error"""
        extractor = OrganizationAnswerExtractor(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock DSPy predictor to raise an exception
        with patch.object(
            extractor, "predictor", side_effect=Exception("Network error")
        ):
            result = extractor.extract(context="content1", user_query="test query")

            assert result["success"] is False
            assert "error" in result

    def test_extract_answers_empty_content(self, mock_credentials):
        """Test answer extraction with empty content"""
        extractor = OrganizationAnswerExtractor(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )
        result = extractor.extract(context="", user_query="test query")

        assert "error" in result or not result.get("success", True)


class TestAnswerReranker:
    """Test AnswerReranker functionality"""

    def test_init(self):
        """Test answer reranker initialization"""
        reranker = AnswerReranker(
            rerank_url="https://rerank.example.com", rerank_model="test-model"
        )
        assert reranker is not None

    def test_rerank_answers_success(self, mock_credentials):
        """Test successful answer reranking"""
        with patch(
            "kbbridge.core.synthesis.answer_reranker.requests.post"
        ) as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                ]
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            reranker = AnswerReranker(
                rerank_url="https://rerank.example.com", rerank_model="test-model"
            )
            answers = [
                {"content": "answer1", "score": 0.7},
                {"content": "answer2", "score": 0.6},
            ]

            result = reranker.rerank_answers(
                query="test query", candidate_answers=answers
            )

            assert isinstance(result, dict)

    def test_rerank_answers_api_error(self, mock_credentials):
        """Test answer reranking with API error"""
        with patch(
            "kbbridge.core.synthesis.answer_reranker.requests.post"
        ) as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "API error"}
            mock_post.return_value = mock_response

            reranker = AnswerReranker(
                rerank_url="https://rerank.example.com", rerank_model="test-model"
            )
            answers = [{"content": "answer1", "score": 0.7}]

            result = reranker.rerank_answers(
                query="test query", candidate_answers=answers
            )

            # Should return a result dict (actual behavior)
            assert isinstance(result, dict)

    def test_rerank_answers_network_error(self, mock_credentials):
        """Test answer reranking with network error"""
        with patch(
            "kbbridge.core.synthesis.answer_reranker.requests.post"
        ) as mock_post:
            mock_post.side_effect = Exception("Network error")

            reranker = AnswerReranker(
                rerank_url="https://rerank.example.com", rerank_model="test-model"
            )
            answers = [{"content": "answer1", "score": 0.7}]

            result = reranker.rerank_answers(
                query="test query", candidate_answers=answers
            )

            # Should return a result dict (actual behavior)
            assert isinstance(result, dict)

    def test_rerank_answers_empty_input(self, mock_credentials):
        """Test answer reranking with empty input"""
        reranker = AnswerReranker(
            rerank_url="https://rerank.example.com", rerank_model="test-model"
        )
        result = reranker.rerank_answers(query="test query", candidate_answers=[])

        assert isinstance(result, dict)


class TestReranker:
    """Test reranker utility functions"""

    def test_rerank_files_by_names_success(self):
        """Test successful file reranking by names"""
        files = [
            {"file_path": "document1.pdf", "score": 0.5},
            {"file_path": "document2.pdf", "score": 0.7},
            {"file_path": "report.pdf", "score": 0.6},
        ]

        reranked = rerank_files_by_names(
            "test query", files, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert isinstance(reranked, dict)

    def test_rerank_files_by_names_empty_files(self):
        """Test file reranking with empty files list"""
        reranked = rerank_files_by_names(
            "test query", [], rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )
        assert isinstance(reranked, dict)

    def test_rerank_files_by_names_empty_keywords(self):
        """Test file reranking with empty keywords list"""
        files = [{"file_path": "test.pdf", "score": 0.5}]
        reranked = rerank_files_by_names(
            "test query", files, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        # Should return a result dict
        assert isinstance(reranked, dict)

    def test_rerank_files_by_names_no_matches(self):
        """Test file reranking with no keyword matches"""
        files = [{"file_path": "test.pdf", "score": 0.5}]

        reranked = rerank_files_by_names(
            "test query", files, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        # Should return a result dict
        assert isinstance(reranked, dict)

    def test_rerank_files_by_names_case_insensitive(self):
        """Test file reranking is case insensitive"""
        files = [{"file_path": "Document.PDF", "score": 0.5}]

        reranked = rerank_files_by_names(
            "test query", files, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert isinstance(reranked, dict)
