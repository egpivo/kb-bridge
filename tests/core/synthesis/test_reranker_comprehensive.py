
"""
Comprehensive tests for reranker module
"""

import json
from unittest.mock import Mock, patch

from kbbridge.core.discovery.file_reranker import (
    combine_rerank_results,
    rerank_documents,
    rerank_files_by_names,
)

# Test fixtures for required parameters
TEST_RERANK_URL = "https://test-rerank.example.com"
TEST_MODEL = "test-model"


class TestCombineRerankResults:
    """Test combine_rerank_results function"""

    def test_combine_rerank_results_success(self):
        """Test combining rerank results successfully"""
        rerank_response = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
                {"index": 2, "relevance_score": 0.3},
            ]
        }
        all_docs = [
            {"document_name": "doc1.pdf", "content": "Content 1"},
            {"document_name": "doc2.pdf", "content": "Content 2"},
            {"document_name": "doc3.pdf", "content": "Content 3"},
        ]
        relevance_score_threshold = 0.5

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert len(result) == 2  # Only scores >= 0.5
        assert result[0]["relevance_score"] == 0.9
        assert result[1]["relevance_score"] == 0.7
        assert result[0]["document"]["document_name"] == "doc1.pdf"
        assert result[1]["document"]["document_name"] == "doc2.pdf"

    def test_combine_rerank_results_empty_response(self):
        """Test combining rerank results with empty response"""
        rerank_response = {"results": []}
        all_docs = [{"document_name": "doc1.pdf"}]
        relevance_score_threshold = 0.5

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert result == []

    def test_combine_rerank_results_no_results_key(self):
        """Test combining rerank results with no results key"""
        rerank_response = {}
        all_docs = [{"document_name": "doc1.pdf"}]
        relevance_score_threshold = 0.5

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert result == []

    def test_combine_rerank_results_invalid_index(self):
        """Test combining rerank results with invalid index"""
        rerank_response = {
            "results": [
                {"index": 5, "relevance_score": 0.9},  # Index out of range
                {"index": -1, "relevance_score": 0.8},  # Negative index
                {"index": 0, "relevance_score": 0.7},
            ]
        }
        all_docs = [{"document_name": "doc1.pdf"}]
        relevance_score_threshold = 0.5

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert len(result) == 1  # Only valid index
        assert result[0]["index"] == 0
        assert result[0]["relevance_score"] == 0.7

    def test_combine_rerank_results_none_index(self):
        """Test combining rerank results with None index"""
        rerank_response = {
            "results": [
                {"index": None, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.7},
            ]
        }
        all_docs = [{"document_name": "doc1.pdf"}]
        relevance_score_threshold = 0.5

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert len(result) == 1  # Only valid index
        assert result[0]["index"] == 0

    def test_combine_rerank_results_low_threshold(self):
        """Test combining rerank results with low threshold"""
        rerank_response = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.3},
                {"index": 2, "relevance_score": 0.1},
            ]
        }
        all_docs = [
            {"document_name": "doc1.pdf"},
            {"document_name": "doc2.pdf"},
            {"document_name": "doc3.pdf"},
        ]
        relevance_score_threshold = 0.0

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert len(result) == 3  # All scores >= 0.0

    def test_combine_rerank_results_high_threshold(self):
        """Test combining rerank results with high threshold"""
        rerank_response = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
                {"index": 2, "relevance_score": 0.3},
            ]
        }
        all_docs = [
            {"document_name": "doc1.pdf"},
            {"document_name": "doc2.pdf"},
            {"document_name": "doc3.pdf"},
        ]
        relevance_score_threshold = 0.8

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert len(result) == 1  # Only score >= 0.8
        assert result[0]["relevance_score"] == 0.9


class TestRerankDocuments:
    """Test rerank_documents function"""

    @patch("kbbridge.core.discovery.file_reranker.requests.post")
    def test_rerank_documents_success(self, mock_post):
        """Test successful document reranking"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
            ]
        }
        mock_post.return_value = mock_response

        query = "test query"
        documents = ["Document 1", "Document 2"]
        all_docs = [
            {"document_name": "doc1.pdf", "content": "Document 1"},
            {"document_name": "doc2.pdf", "content": "Document 2"},
        ]

        result = rerank_documents(
            query, documents, all_docs, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is True
        assert len(result["final_results"]) == 2
        assert len(result["detailed_results"]) == 2
        assert result["total_reranked"] == 2
        assert "doc1.pdf" in result["final_results"]
        assert "doc2.pdf" in result["final_results"]

    @patch("kbbridge.core.discovery.file_reranker.requests.post")
    def test_rerank_documents_api_error(self, mock_post):
        """Test document reranking with API error"""
        # Mock API error
        mock_post.side_effect = Exception("API Error")

        query = "test query"
        documents = ["Document 1"]
        all_docs = [{"document_name": "doc1.pdf"}]

        result = rerank_documents(
            query, documents, all_docs, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is False
        assert "error" in result
        assert result["final_results"] == []
        assert result["detailed_results"] == []
        assert result["total_reranked"] == 0

    @patch("kbbridge.core.discovery.file_reranker.requests.post")
    def test_rerank_documents_http_error(self, mock_post):
        """Test document reranking with HTTP error"""
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("400 Bad Request")
        mock_post.return_value = mock_response

        query = "test query"
        documents = ["Document 1"]
        all_docs = [{"document_name": "doc1.pdf"}]

        result = rerank_documents(
            query, documents, all_docs, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is False
        assert "error" in result

    @patch("kbbridge.core.discovery.file_reranker.requests.post")
    def test_rerank_documents_empty_documents(self, mock_post):
        """Test document reranking with empty documents"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response

        query = "test query"
        documents = []
        all_docs = []

        result = rerank_documents(
            query, documents, all_docs, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is True
        assert result["final_results"] == []
        assert result["detailed_results"] == []
        assert result["total_reranked"] == 0

    @patch("kbbridge.core.discovery.file_reranker.requests.post")
    def test_rerank_documents_custom_parameters(self, mock_post):
        """Test document reranking with custom parameters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_post.return_value = mock_response

        query = "test query"
        documents = ["Document 1"]
        all_docs = [{"document_name": "doc1.pdf"}]

        result = rerank_documents(
            query=query,
            documents=documents,
            all_docs=all_docs,
            relevance_score_threshold=0.8,
            rerank_url="https://custom-rerank.com",
            model="custom-model",
        )

        assert result["success"] is True
        # Verify custom parameters were used
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        # URL is normalized to include /rerank suffix
        assert call_args[0][0] == "https://custom-rerank.com/rerank"
        payload = json.loads(call_args[1]["data"])
        assert payload["model"] == "custom-model"

    @patch("kbbridge.core.discovery.file_reranker.requests.post")
    def test_rerank_documents_malformed_response(self, mock_post):
        """Test document reranking with malformed response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response

        query = "test query"
        documents = ["Document 1"]
        all_docs = [{"document_name": "doc1.pdf"}]

        result = rerank_documents(
            query, documents, all_docs, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is False
        assert "error" in result


class TestRerankFilesByNames:
    """Test rerank_files_by_names function"""

    @patch("kbbridge.core.discovery.file_reranker.rerank_documents")
    def test_rerank_files_by_names_success(self, mock_rerank_documents):
        """Test successful file reranking by names"""
        mock_rerank_documents.return_value = {
            "success": True,
            "final_results": ["file1.pdf", "file2.pdf"],
            "detailed_results": [
                {
                    "index": 0,
                    "relevance_score": 0.9,
                    "document": {"document_name": "file1.pdf"},
                },
                {
                    "index": 1,
                    "relevance_score": 0.7,
                    "document": {"document_name": "file2.pdf"},
                },
            ],
            "total_reranked": 2,
        }

        query = "test query"
        file_names = ["file1.pdf", "file2.pdf"]

        result = rerank_files_by_names(
            query, file_names, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is True
        assert len(result["final_results"]) == 2
        assert "file1.pdf" in result["final_results"]
        assert "file2.pdf" in result["final_results"]

        # Verify rerank_documents was called correctly
        mock_rerank_documents.assert_called_once()
        call_args = mock_rerank_documents.call_args
        assert call_args[0][0] == query  # query
        # documents are enhanced with relevance labels for better reranking
        enhanced_docs = call_args[0][1]
        assert len(enhanced_docs) == 2
        assert "file1.pdf" in enhanced_docs[0]
        assert "file2.pdf" in enhanced_docs[1]
        # all_docs contains original document names
        assert call_args[0][2] == [
            {"document_name": "file1.pdf"},
            {"document_name": "file2.pdf"},
        ]  # all_docs

    @patch("kbbridge.core.discovery.file_reranker.rerank_documents")
    def test_rerank_files_by_names_failure(self, mock_rerank_documents):
        """Test file reranking by names with failure"""
        mock_rerank_documents.return_value = {
            "success": False,
            "error": "Rerank failed",
            "final_results": [],
            "detailed_results": [],
            "total_reranked": 0,
        }

        query = "test query"
        file_names = ["file1.pdf"]

        result = rerank_files_by_names(
            query, file_names, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is False
        assert "error" in result
        assert result["final_results"] == []

    @patch("kbbridge.core.discovery.file_reranker.rerank_documents")
    def test_rerank_files_by_names_empty_files(self, mock_rerank_documents):
        """Test file reranking by names with empty file list"""
        mock_rerank_documents.return_value = {
            "success": True,
            "final_results": [],
            "detailed_results": [],
            "total_reranked": 0,
        }

        query = "test query"
        file_names = []

        result = rerank_files_by_names(
            query, file_names, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is True
        assert result["final_results"] == []
        assert result["total_reranked"] == 0

    @patch("kbbridge.core.discovery.file_reranker.rerank_documents")
    def test_rerank_files_by_names_custom_parameters(self, mock_rerank_documents):
        """Test file reranking by names with custom parameters"""
        mock_rerank_documents.return_value = {
            "success": True,
            "final_results": ["file1.pdf"],
            "detailed_results": [],
            "total_reranked": 1,
        }

        query = "test query"
        file_names = ["file1.pdf"]

        result = rerank_files_by_names(
            query=query,
            file_names=file_names,
            relevance_score_threshold=0.8,
            rerank_url="https://custom-rerank.com",
            model="custom-model",
        )

        assert result["success"] is True

        # Verify custom parameters were passed
        mock_rerank_documents.assert_called_once()
        call_args = mock_rerank_documents.call_args
        # Check positional arguments: query, documents, all_docs, relevance_score_threshold, rerank_url, model
        assert call_args[0][0] == query  # query
        # documents are enhanced with relevance labels
        enhanced_docs = call_args[0][1]
        assert len(enhanced_docs) == 1
        assert "file1.pdf" in enhanced_docs[0]
        assert call_args[0][2] == [{"document_name": "file1.pdf"}]  # all_docs
        assert call_args[0][3] == 0.8  # relevance_score_threshold
        assert call_args[0][4] == "https://custom-rerank.com"  # rerank_url
        assert call_args[0][5] == "custom-model"  # model

    @patch("kbbridge.core.discovery.file_reranker.rerank_documents")
    def test_rerank_files_by_names_single_file(self, mock_rerank_documents):
        """Test file reranking by names with single file"""
        mock_rerank_documents.return_value = {
            "success": True,
            "final_results": ["file1.pdf"],
            "detailed_results": [
                {
                    "index": 0,
                    "relevance_score": 0.9,
                    "document": {"document_name": "file1.pdf"},
                }
            ],
            "total_reranked": 1,
        }

        query = "test query"
        file_names = ["file1.pdf"]

        result = rerank_files_by_names(
            query, file_names, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is True
        assert len(result["final_results"]) == 1
        assert result["final_results"][0] == "file1.pdf"

    @patch("kbbridge.core.discovery.file_reranker.rerank_documents")
    def test_rerank_files_by_names_duplicate_files(self, mock_rerank_documents):
        """Test file reranking by names with duplicate files"""
        mock_rerank_documents.return_value = {
            "success": True,
            "final_results": ["file1.pdf", "file1.pdf"],
            "detailed_results": [
                {
                    "index": 0,
                    "relevance_score": 0.9,
                    "document": {"document_name": "file1.pdf"},
                },
                {
                    "index": 1,
                    "relevance_score": 0.8,
                    "document": {"document_name": "file1.pdf"},
                },
            ],
            "total_reranked": 2,
        }

        query = "test query"
        file_names = ["file1.pdf", "file1.pdf"]

        result = rerank_files_by_names(
            query, file_names, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is True
        assert len(result["final_results"]) == 2
        assert result["final_results"] == ["file1.pdf", "file1.pdf"]


class TestRerankerEdgeCases:
    """Test reranker edge cases"""

    def test_combine_rerank_results_missing_score(self):
        """Test combining rerank results with missing relevance score"""
        rerank_response = {
            "results": [
                {"index": 0},  # Missing relevance_score
                {"index": 1, "relevance_score": 0.7},
            ]
        }
        all_docs = [{"document_name": "doc1.pdf"}, {"document_name": "doc2.pdf"}]
        relevance_score_threshold = 0.5

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert len(result) == 1  # Only one with valid score
        assert result[0]["index"] == 1

    def test_combine_rerank_results_none_score(self):
        """Test combining rerank results with None relevance score"""
        rerank_response = {
            "results": [
                {"index": 0, "relevance_score": None},
                {"index": 1, "relevance_score": 0.7},
            ]
        }
        all_docs = [{"document_name": "doc1.pdf"}, {"document_name": "doc2.pdf"}]
        relevance_score_threshold = 0.5

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert len(result) == 1  # Only one with valid score
        assert result[0]["index"] == 1

    def test_combine_rerank_results_string_score(self):
        """Test combining rerank results with string relevance score"""
        rerank_response = {
            "results": [
                {"index": 0, "relevance_score": "0.9"},  # String instead of float
                {"index": 1, "relevance_score": 0.7},
            ]
        }
        all_docs = [{"document_name": "doc1.pdf"}, {"document_name": "doc2.pdf"}]
        relevance_score_threshold = 0.5

        result = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        assert len(result) == 1  # Only one with valid score
        assert result[0]["index"] == 1

    @patch("kbbridge.core.discovery.file_reranker.requests.post")
    def test_rerank_documents_network_timeout(self, mock_post):
        """Test document reranking with network timeout"""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")

        query = "test query"
        documents = ["Document 1"]
        all_docs = [{"document_name": "doc1.pdf"}]

        result = rerank_documents(
            query, documents, all_docs, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is False
        assert "error" in result
        assert "timeout" in result["error"].lower()

    @patch("kbbridge.core.discovery.file_reranker.requests.post")
    def test_rerank_documents_connection_error(self, mock_post):
        """Test document reranking with connection error"""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        query = "test query"
        documents = ["Document 1"]
        all_docs = [{"document_name": "doc1.pdf"}]

        result = rerank_documents(
            query, documents, all_docs, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is False
        assert "error" in result
        assert "connection" in result["error"].lower()

    @patch("kbbridge.core.discovery.file_reranker.rerank_documents")
    def test_rerank_files_by_names_none_file_names(self, mock_rerank_documents):
        """Test file reranking by names with None file names"""
        mock_rerank_documents.return_value = {
            "success": False,
            "error": "Invalid input",
            "final_results": [],
            "detailed_results": [],
            "total_reranked": 0,
        }

        query = "test query"
        file_names = None

        # This should handle the None case gracefully
        try:
            result = rerank_files_by_names(
                query, file_names, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
            )
            assert result["success"] is False
        except (TypeError, AttributeError):
            # Expected behavior for None input
            pass

    @patch("kbbridge.core.discovery.file_reranker.rerank_documents")
    def test_rerank_files_by_names_mixed_types(self, mock_rerank_documents):
        """Test file reranking by names with mixed types"""
        mock_rerank_documents.return_value = {
            "success": True,
            "final_results": ["file1.pdf", "123"],
            "detailed_results": [],
            "total_reranked": 2,
        }

        query = "test query"
        file_names = ["file1.pdf", 123, None]  # Mixed types

        result = rerank_files_by_names(
            query, file_names, rerank_url=TEST_RERANK_URL, model=TEST_MODEL
        )

        assert result["success"] is True
        # Should handle mixed types by converting to strings
        assert len(result["final_results"]) == 2
