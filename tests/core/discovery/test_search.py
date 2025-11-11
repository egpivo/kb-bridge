"""
Test Suite for Search Modules
Tests all search modules with comprehensive coverage
"""

from unittest.mock import Mock, patch

from kbbridge.integrations.dify.dify_retriever import DifyRetriever
from kbbridge.utils.formatting import format_search_results
from kbbridge.utils.working_components import KnowledgeBaseRetriever


class TestDifyRetrieverFileListing:
    """Test file listing via DifyRetriever"""

    def test_init(self):
        """Test file lister initialization"""
        retriever = DifyRetriever(
            endpoint="https://dify-instance",
            api_key="test-api-key",
            dataset_id="test-dataset",
        )
        assert retriever is not None

    def test_list_files_success(self, mock_credentials):
        """Test successful file listing"""
        with patch(
            "kbbridge.integrations.dify.dify_retriever.requests.get"
        ) as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [
                    {"id": "file1", "name": "document1.pdf", "size": 1024},
                    {"id": "file2", "name": "document2.pdf", "size": 2048},
                ]
            }
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            retriever = DifyRetriever(
                endpoint="https://dify-instance",
                api_key="test-api-key",
                dataset_id="test-dataset",
            )
            files = retriever.list_files(dataset_id="test-dataset")

            assert isinstance(files, list)
            assert len(files) == 2

    def test_list_files_api_error(self, mock_credentials):
        """Test file listing with API error"""
        with patch(
            "kbbridge.integrations.dify.dify_retriever.requests.get"
        ) as mock_get:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "API error"}
            mock_get.return_value = mock_response

            retriever = DifyRetriever(
                endpoint="https://dify-instance",
                api_key="test-api-key",
                dataset_id="test-dataset",
            )
            files = retriever.list_files(dataset_id="test-dataset")

            assert isinstance(files, list)

    def test_list_files_network_error(self, mock_credentials):
        """Test file listing with network error"""
        with patch(
            "kbbridge.integrations.dify.dify_retriever.requests.get"
        ) as mock_get:
            mock_get.side_effect = Exception("Network error")

            retriever = DifyRetriever(
                endpoint="https://dify-instance",
                api_key="test-api-key",
                dataset_id="test-dataset",
            )
            files = retriever.list_files(dataset_id="test-dataset")

            assert isinstance(files, list)

    # FileSearcher tests removed; FileDiscover is now the engine and covered separately.


class TestKnowledgeBaseRetriever:
    """Test KnowledgeBaseRetriever functionality"""

    def test_init(self):
        """Test retriever initialization"""
        retriever = KnowledgeBaseRetriever(
            endpoint="https://dify-instance", api_key="test-api-key"
        )
        assert retriever is not None

    def test_retrieve_success(self, mock_credentials):
        """Test successful retrieval"""
        with patch("kbbridge.utils.working_components.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "records": [
                    {
                        "segment": {
                            "content": "Test content 1",
                            "document": {
                                "doc_metadata": {"document_name": "test1.pdf"}
                            },
                        }
                    }
                ]
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            retriever = KnowledgeBaseRetriever(
                endpoint="https://dify-instance", api_key="test-api-key"
            )
            results = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method="hybrid_search",
                does_rerank=False,
                top_k=10,
                reranking_provider_name="",
                reranking_model_name="",
                score_threshold_enabled=False,
            )

            assert isinstance(results, dict)

    def test_retrieve_api_error(self, mock_credentials):
        """Test retrieval with API error"""
        with patch("kbbridge.utils.working_components.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "API error"}
            mock_post.return_value = mock_response

            retriever = KnowledgeBaseRetriever(
                endpoint="https://dify-instance", api_key="test-api-key"
            )
            results = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method="hybrid_search",
                does_rerank=False,
                top_k=10,
                reranking_provider_name="",
                reranking_model_name="",
                score_threshold_enabled=False,
            )

            assert isinstance(results, dict)

    def test_retrieve_network_error(self, mock_credentials):
        """Test retrieval with network error"""
        with patch("kbbridge.utils.working_components.requests.post") as mock_post:
            mock_post.side_effect = Exception("Network error")

            retriever = KnowledgeBaseRetriever(
                endpoint="https://dify-instance", api_key="test-api-key"
            )
            results = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method="hybrid_search",
                does_rerank=False,
                top_k=10,
                reranking_provider_name="",
                reranking_model_name="",
                score_threshold_enabled=False,
            )

            assert isinstance(results, dict)


class TestFormatSearchResults:
    """Test format_search_results utility function"""

    def test_format_search_results_success(self):
        """Test successful result formatting"""
        results = [
            {
                "segment": {
                    "content": "Test content 1",
                    "document": {"doc_metadata": {"document_name": "test1.pdf"}},
                }
            },
            {
                "segment": {
                    "content": "Test content 2",
                    "document": {"doc_metadata": {"document_name": "test2.pdf"}},
                }
            },
        ]

        formatted = format_search_results(results)

        assert isinstance(formatted, dict)
        assert "result" in formatted

    def test_format_search_results_empty(self):
        """Test formatting empty results"""
        formatted = format_search_results([])
        assert isinstance(formatted, dict)
        assert "result" in formatted

    def test_format_search_results_malformed(self):
        """Test formatting malformed results"""
        results = [{"invalid": "structure"}]
        formatted = format_search_results(results)

        # Should handle malformed data gracefully
        assert isinstance(formatted, dict)
        assert "result" in formatted
