
"""
Test answer_reranker module functionality
"""

from unittest.mock import Mock, patch

from kbbridge.core.synthesis.answer_reranker import AnswerReranker


class TestAnswerReranker:
    """Test AnswerReranker class"""

    def test_init(self):
        """Test reranker initialization"""
        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        assert reranker.rerank_url == "https://rerank.com"
        assert reranker.rerank_model == "rerank-model"

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_success(self, mock_post):
        """Test successful answer reranking"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
            ]
        }
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {
                "success": True,
                "answer": "First answer",
                "source": "naive",
                "dataset_id": "dataset1",
            },
            {
                "success": True,
                "answer": "Second answer",
                "source": "advanced",
                "dataset_id": "dataset2",
                "file_name": "test.pdf",
            },
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert isinstance(result, dict)
        assert "final_result" in result
        assert "detailed_results" in result
        assert len(result["detailed_results"]) == 2
        assert result["detailed_results"][0]["relevance_score"] == 0.9

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_api_error(self, mock_post):
        """Test reranking with API error"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("API error")
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {"success": True, "answer": "Test answer", "source": "naive"}
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert isinstance(result, dict)
        assert "final_result" in result
        assert "detailed_results" in result
        assert "rerank_error" in result

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_network_error(self, mock_post):
        """Test reranking with network error"""
        mock_post.side_effect = Exception("Network error")

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {"success": True, "answer": "Test answer", "source": "naive"}
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert isinstance(result, dict)
        assert "final_result" in result
        assert "detailed_results" in result
        assert "rerank_error" in result

    def test_rerank_answers_empty_candidates(self):
        """Test reranking with empty candidate answers"""
        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        result = reranker.rerank_answers("test query", [])

        assert isinstance(result, dict)
        assert result["final_result"] == ""
        assert result["detailed_results"] == []

    def test_rerank_answers_no_valid_answers(self):
        """Test reranking with no valid answers"""
        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {"success": False, "answer": "Invalid answer"},
            {"success": True, "answer": "N/A"},
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert isinstance(result, dict)
        assert result["final_result"] == ""
        assert result["detailed_results"] == []

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_naive_source_formatting(self, mock_post):
        """Test formatting for naive source answers"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {
                "success": True,
                "answer": "Simple answer",
                "source": "naive",
                "dataset_id": "dataset1",
            }
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert result["final_result"] == "Simple answer"

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_advanced_source_with_filename(self, mock_post):
        """Test formatting for advanced source with filename"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {
                "success": True,
                "answer": "Advanced answer",
                "source": "advanced",
                "dataset_id": "dataset1",
                "file_name": "document.pdf",
            }
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert "**dataset1/document.pdf**: Advanced answer" in result["final_result"]

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_advanced_source_with_source_path(self, mock_post):
        """Test formatting for advanced source with source path"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {
                "success": True,
                "answer": "Advanced answer",
                "source": "advanced",
                "dataset_id": "dataset1",
                "source_path": "/path/to/file",
            }
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert "**dataset1 (/path/to/file)**: Advanced answer" in result["final_result"]

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_advanced_source_dataset_only(self, mock_post):
        """Test formatting for advanced source with dataset only"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {
                "success": True,
                "answer": "Advanced answer",
                "source": "advanced",
                "dataset_id": "dataset1",
            }
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert "**dataset1**: Advanced answer" in result["final_result"]

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_custom_timeout(self, mock_post):
        """Test reranking with custom timeout"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {"success": True, "answer": "Test answer", "source": "naive"}
        ]

        result = reranker.rerank_answers("test query", candidate_answers, timeout=60)

        # Verify timeout was passed to requests.post
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 60

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_malformed_response(self, mock_post):
        """Test reranking with malformed API response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 5, "relevance_score": 0.7},  # Invalid index
            ]
        }
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {"success": True, "answer": "Test answer", "source": "naive"}
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert isinstance(result, dict)
        assert "final_result" in result
        assert "detailed_results" in result
        # Should only include valid results
        assert len(result["detailed_results"]) == 1

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_sorting(self, mock_post):
        """Test that results are sorted by relevance score"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.7},
            ]
        }
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {"success": True, "answer": "First answer", "source": "naive"},
            {"success": True, "answer": "Second answer", "source": "naive"},
            {"success": True, "answer": "Third answer", "source": "naive"},
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        # Results should be sorted by relevance score (highest first)
        scores = [r["relevance_score"] for r in result["detailed_results"]]
        assert scores == [0.9, 0.7, 0.5]


class TestAnswerRerankerEdgeCases:
    """Test edge cases and error conditions"""

    def test_rerank_answers_none_candidates(self):
        """Test reranking with None candidate answers"""
        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        result = reranker.rerank_answers("test query", None)

        assert isinstance(result, dict)
        assert result["final_result"] == ""
        assert result["detailed_results"] == []

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_json_serialization_error(self, mock_post):
        """Test reranking with JSON serialization error"""
        mock_post.side_effect = TypeError("Object not JSON serializable")

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {"success": True, "answer": "Test answer", "source": "naive"}
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert isinstance(result, dict)
        assert "rerank_error" in result

    @patch("kbbridge.core.synthesis.answer_reranker.requests.post")
    def test_rerank_answers_empty_api_response(self, mock_post):
        """Test reranking with empty API response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {"success": True, "answer": "Test answer", "source": "naive"}
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert isinstance(result, dict)
        assert result["final_result"] == ""
        assert result["detailed_results"] == []

    def test_rerank_answers_missing_fields(self):
        """Test reranking with candidate answers missing fields"""
        reranker = AnswerReranker("https://rerank.com", "rerank-model")

        candidate_answers = [
            {
                "success": True,
                # Missing answer field
                "source": "naive",
            },
            {
                # Missing success field
                "answer": "Test answer",
                "source": "naive",
            },
        ]

        result = reranker.rerank_answers("test query", candidate_answers)

        assert isinstance(result, dict)
        assert result["final_result"] == ""
        assert result["detailed_results"] == []
