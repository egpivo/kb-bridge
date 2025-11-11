"""
Comprehensive tests for qa_hub.services.retriever_service module
"""

from unittest.mock import Mock, patch

import requests

from kbbridge.config.constants import RetrieverDefaults
from kbbridge.services.retriever_service import (
    DEFAULT_CONFIG,
    KnowledgeBaseRetriever,
    RetrieverSearchMethod,
    format_search_results,
    retriever_service,
)


class TestRetrieverService:
    """Test the retriever_service module functionality"""

    def test_retriever_service_is_callable(self):
        """Test that retriever_service is a callable function"""
        # Services are now plain functions, not MCP tools
        assert callable(retriever_service)

    def test_default_config_structure(self):
        """Test DEFAULT_CONFIG structure (backend-agnostic)"""
        assert isinstance(DEFAULT_CONFIG, dict)
        assert "search_method" in DEFAULT_CONFIG
        assert "does_rerank" in DEFAULT_CONFIG
        assert "top_k" in DEFAULT_CONFIG
        assert "score_threshold" in DEFAULT_CONFIG
        assert "weights" in DEFAULT_CONFIG
        assert "document_name" in DEFAULT_CONFIG
        assert "verbose" in DEFAULT_CONFIG
        # Reranking provider/model are NOT in default config - they're backend-specific
        assert "reranking_provider_name" not in DEFAULT_CONFIG
        assert "reranking_model_name" not in DEFAULT_CONFIG


class TestKnowledgeBaseRetrieverComponents:
    """Test KnowledgeBaseRetriever class components"""

    def test_knowledge_base_retriever_init(self):
        """Test KnowledgeBaseRetriever initialization"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")
        assert retriever.endpoint == "https://test.com"
        assert retriever.api_key == "test-key"

    def test_knowledge_base_retriever_retrieve_success(self):
        """Test successful retrieval"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"data": []}
            mock_post.return_value.status_code = 200

            # Test with all required parameters
            result = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method=RetrieverSearchMethod.SEMANTIC_SEARCH.value,
                does_rerank=True,
                top_k=10,
                reranking_provider_name="test_provider",
                reranking_model_name="test_model",
                score_threshold_enabled=True,
                metadata_filter={"key": "value"},
                score_threshold=0.5,
                weights={"content": 0.8},
            )

            # Verify the request was made
            mock_post.assert_called_once()
            # The result should be the mock response
            assert result == mock_post.return_value.json.return_value

    def test_knowledge_base_retriever_retrieve_api_error(self):
        """Test API error handling"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        with patch("kbbridge.utils.working_components.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"
            mock_response.reason = "Bad Request"
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "400 Client Error: Bad Request"
            )
            mock_post.return_value = mock_response

            result = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method=RetrieverSearchMethod.SEMANTIC_SEARCH.value,
                does_rerank=True,
                top_k=10,
                reranking_provider_name="test_provider",
                reranking_model_name="test_model",
                score_threshold_enabled=True,
            )

            # Should return error dict due to HTTPError
            assert result.get("error") is True
            assert "status_code" in result
            assert result["status_code"] == 400

    def test_knowledge_base_retriever_retrieve_network_error(self):
        """Test network error handling"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Network error")

            result = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method=RetrieverSearchMethod.SEMANTIC_SEARCH.value,
                does_rerank=True,
                top_k=10,
                reranking_provider_name="test_provider",
                reranking_model_name="test_model",
                score_threshold_enabled=True,
            )

            # Should handle the exception and return error info
            assert "error" in result
            assert "Network error" in result["error_message"]

    def test_knowledge_base_retriever_build_metadata_filter(self):
        """Test metadata filter building"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        # Test with document_name only
        filter_result = retriever.build_metadata_filter(document_name="test.pdf")
        assert filter_result is not None
        assert "conditions" in filter_result
        assert filter_result["conditions"][0]["name"] == "document_name"
        assert filter_result["conditions"][0]["value"] == "test.pdf"

    def test_format_search_results_success(self):
        """Test format_search_results with valid data"""
        results = [
            {
                "records": [
                    {
                        "segment": {
                            "content": "Test content",
                            "document": {"doc_metadata": {"document_name": "test.pdf"}},
                        }
                    }
                ]
            }
        ]

        result = format_search_results(results)

        assert isinstance(result, dict)
        assert "result" in result
        assert len(result["result"]) == 1
        assert result["result"][0]["content"] == "Test content"
        assert result["result"][0]["document_name"] == "test.pdf"

    def test_format_search_results_error_handling(self):
        """Test format_search_results error handling"""
        # Test with problematic data that causes exceptions
        problematic_results = [
            {
                "records": [
                    {
                        "segment": {
                            "content": None,  # This will cause an exception
                            "document": {"doc_metadata": {"document_name": "test.pdf"}},
                        }
                    }
                ]
            }
        ]

        result = format_search_results(problematic_results)

        # Should handle exception gracefully
        assert isinstance(result, dict)
        assert "result" in result
        assert isinstance(result["result"], list)

    def test_retriever_search_method_validation(self):
        """Test search method validation"""
        # Test that all RetrieverSearchMethod values are valid
        valid_methods = [method.value for method in RetrieverSearchMethod]

        # Test each method
        for method in valid_methods:
            retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = {"data": []}
                mock_post.return_value.status_code = 200

                result = retriever.retrieve(
                    dataset_id="test-dataset",
                    query="test query",
                    search_method=method,
                    does_rerank=True,
                    top_k=10,
                    reranking_provider_name="test_provider",
                    reranking_model_name="test_model",
                    score_threshold_enabled=True,
                )

                # The result should be the mock response
                assert result == mock_post.return_value.json.return_value

    def test_parameter_validation_edge_cases(self):
        """Test parameter validation edge cases"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"data": []}
            mock_post.return_value.status_code = 200

            # Test with top_k as float with decimal part
            result = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method=RetrieverSearchMethod.SEMANTIC_SEARCH.value,
                does_rerank=True,
                top_k=5.5,  # Float with decimal part
                reranking_provider_name="test_provider",
                reranking_model_name="test_model",
                score_threshold_enabled=True,
            )

            # Should handle the float conversion
            assert result == mock_post.return_value.json.return_value

    def test_weights_handling_for_search_methods(self):
        """Test weights handling for different search methods"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"data": []}
            mock_post.return_value.status_code = 200

            # Test with keyword search
            result = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method=RetrieverSearchMethod.KEYWORD_SEARCH.value,
                does_rerank=True,
                top_k=10,
                reranking_provider_name="test_provider",
                reranking_model_name="test_model",
                score_threshold_enabled=True,
                weights={"content": 0.8},
            )

            # Verify the request was made
            mock_post.assert_called_once()
            # The result should be the mock response
            assert result == mock_post.return_value.json.return_value

    def test_score_threshold_handling(self):
        """Test score threshold handling"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"data": []}
            mock_post.return_value.status_code = 200

            # Test with score_threshold set
            result = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                search_method=RetrieverSearchMethod.SEMANTIC_SEARCH.value,
                does_rerank=True,
                top_k=10,
                reranking_provider_name="test_provider",
                reranking_model_name="test_model",
                score_threshold_enabled=True,
                score_threshold=0.5,
            )

            # Verify the request was made
            mock_post.assert_called_once()
            # The result should be the mock response
            assert result == mock_post.return_value.json.return_value


class TestRetrieverServiceEdgeCases:
    """Test edge cases for retriever service"""

    def test_knowledge_base_retriever_invalid_credentials(self):
        """Test KnowledgeBaseRetriever with invalid credentials"""
        # Test with None values - should not raise exception during init
        retriever = KnowledgeBaseRetriever(None, None)
        assert retriever.endpoint is None
        assert retriever.api_key is None

    def test_format_search_results_none(self):
        """Test format_search_results with None input"""
        result = format_search_results(None)
        assert "result" in result
        assert result["result"] == []

    def test_format_search_results_empty(self):
        """Test format_search_results with empty input"""
        result = format_search_results([])
        assert "result" in result
        assert result["result"] == []

    def test_format_search_results_malformed(self):
        """Test format_search_results with malformed data"""
        malformed_data = [{"invalid": "structure"}]
        result = format_search_results(malformed_data)
        assert "result" in result
        assert isinstance(result["result"], list)

    def test_retriever_search_method_enum_values(self):
        """Test that RetrieverSearchMethod enum has expected values"""
        expected_methods = ["semantic_search", "keyword_search", "hybrid_search"]
        actual_methods = [method.value for method in RetrieverSearchMethod]

        for expected in expected_methods:
            assert expected in actual_methods

    def test_default_config_values(self):
        """Test DEFAULT_CONFIG has expected values (backend-agnostic)"""
        assert DEFAULT_CONFIG["search_method"] == RetrieverDefaults.SEARCH_METHOD.value
        assert DEFAULT_CONFIG["does_rerank"] == RetrieverDefaults.DOES_RERANK.value
        assert DEFAULT_CONFIG["top_k"] == RetrieverDefaults.TOP_K.value
        # Reranking provider/model are backend-specific - NOT in default config
        assert (
            DEFAULT_CONFIG["score_threshold"] == RetrieverDefaults.SCORE_THRESHOLD.value
        )
        assert DEFAULT_CONFIG["weights"] == RetrieverDefaults.WEIGHTS.value


class TestRetrieverServiceFunction:
    """Test the retriever_service function with different credential types"""

    def test_retriever_service_with_document_name_filter(self):
        """Test retriever_service with document_name filter"""
        # Patch BackendAdapterFactory at its source module
        with patch("kbbridge.integrations.RetrievalCredentials") as mock_creds_class:
            with patch(
                "kbbridge.integrations.backend_adapter.BackendAdapterFactory"
            ) as mock_factory:
                # Setup mocks
                mock_creds = Mock()
                mock_creds.validate.return_value = (True, None)
                mock_creds.endpoint = "https://test.com"
                mock_creds.api_key = "test-key"
                mock_creds.backend_type = "dify"
                mock_creds_class.return_value = mock_creds

                # Mock adapter
                mock_adapter = Mock()
                # Return non-empty results to avoid fallback logic
                mock_adapter.search.return_value = {
                    "records": [{"segment": {"content": "test"}}]
                }
                mock_factory.create.return_value = mock_adapter

                # Call with document_name filter
                result = retriever_service(
                    resource_id="test-resource",
                    query="test query",
                    retrieval_endpoint="https://test.com",
                    retrieval_api_key="test-key",
                    document_name="test.pdf",
                )

                # Verify adapter.search was called with document_name
                assert (
                    mock_adapter.search.called
                ), "adapter.search should have been called"
                call_kwargs = mock_adapter.search.call_args.kwargs
                assert call_kwargs["document_name"] == "test.pdf"
                assert "result" in result

    def test_retriever_service_with_document_name(self):
        """Test retriever_service with document_name filter"""
        with patch("kbbridge.integrations.RetrievalCredentials") as mock_creds_class:
            with patch(
                "kbbridge.integrations.backend_adapter.BackendAdapterFactory"
            ) as mock_factory:
                # Setup mocks
                mock_creds = Mock()
                mock_creds.validate.return_value = (True, None)
                mock_creds.endpoint = "https://test.com"
                mock_creds.api_key = "test-key"
                mock_creds.backend_type = "dify"
                mock_creds_class.return_value = mock_creds

                # Mock adapter
                mock_adapter = Mock()
                # Return non-empty results to avoid fallback logic
                mock_adapter.search.return_value = {
                    "records": [{"segment": {"content": "test"}}]
                }
                mock_factory.create.return_value = mock_adapter

                # Call with document_name filter
                result = retriever_service(
                    resource_id="test-resource",
                    query="test query",
                    retrieval_endpoint="https://test.com",
                    retrieval_api_key="test-key",
                    document_name="test.pdf",
                )

                # Verify adapter.search was called with document_name
                assert (
                    mock_adapter.search.called
                ), "adapter.search should have been called"
                call_kwargs = mock_adapter.search.call_args.kwargs
                assert call_kwargs["document_name"] == "test.pdf"
                assert "result" in result


class TestListAvailableBackends:
    """Test list_available_backends function"""

    @patch("kbbridge.integrations.RetrieverRouter")
    @patch.dict(
        "os.environ",
        {
            "RETRIEVER_BACKEND": "dify",
            "RETRIEVAL_ENDPOINT": "https://test.com",
            "RETRIEVAL_API_KEY": "key",
        },
    )
    def test_list_available_backends_success(self, mock_router):
        """Test list_available_backends with environment variables"""
        from kbbridge.services.retriever_service import list_available_backends

        mock_router.get_available_backends.return_value = ["dify", "opensearch", "n8n"]

        result = list_available_backends()

        assert "available_backends" in result
        assert "current_backend" in result
        assert "environment_variables" in result
        assert result["current_backend"] == "dify"
        assert result["environment_variables"]["RETRIEVAL_ENDPOINT"] == "***"
        assert result["environment_variables"]["RETRIEVAL_API_KEY"] == "***"

    @patch("kbbridge.integrations.RetrieverRouter")
    @patch.dict("os.environ", {}, clear=True)
    def test_list_available_backends_no_env(self, mock_router):
        """Test list_available_backends without environment variables"""
        from kbbridge.services.retriever_service import list_available_backends

        mock_router.get_available_backends.return_value = ["dify"]

        result = list_available_backends()

        assert "available_backends" in result
        assert result["current_backend"] == "dify"  # default
        assert result["environment_variables"]["RETRIEVAL_ENDPOINT"] is None
        assert result["environment_variables"]["RETRIEVAL_API_KEY"] is None

    @patch("kbbridge.integrations.RetrieverRouter")
    def test_list_available_backends_exception(self, mock_router):
        """Test list_available_backends exception handling"""
        from kbbridge.services.retriever_service import list_available_backends

        mock_router.get_available_backends.side_effect = Exception("Router error")

        result = list_available_backends()

        assert "error" in result
        assert "Router error" in result["error"]


class TestRetrieverServiceValidation:
    """Test retriever_service validation and error paths"""

    def test_retriever_service_empty_dataset_id(self):
        """Test retriever_service with empty resource_id"""
        result = retriever_service(resource_id="", query="test")

        # Covers line 97
        assert "error" in result
        assert "resource_id is required" in result["error"]

    def test_retriever_service_empty_query(self):
        """Test retriever_service with empty query"""
        result = retriever_service(resource_id="test-id", query="")

        # Covers line 99
        assert "error" in result
        assert "query is required" in result["error"]

    def test_retriever_service_opensearch_credentials(self):
        """Test retriever_service with opensearch credentials"""
        with patch("kbbridge.integrations.RetrievalCredentials") as mock_creds_class:
            with patch(
                "kbbridge.integrations.backend_adapter.BackendAdapterFactory"
            ) as mock_factory:
                mock_creds = Mock()
                mock_creds.validate.return_value = (True, None)
                mock_creds.endpoint = "https://opensearch.com"
                mock_creds.api_key = "opensearch-key"
                mock_creds.backend_type = "opensearch"
                mock_creds_class.return_value = mock_creds

                # Mock factory to raise NotImplementedError for opensearch
                mock_factory.create.side_effect = NotImplementedError(
                    "OpenSearch backend adapter not yet implemented"
                )

                result = retriever_service(
                    resource_id="test-resource",
                    query="test query",
                    opensearch_endpoint="https://opensearch.com",
                    opensearch_auth="opensearch-key",
                )

                assert "error" in result
                assert (
                    "OpenSearch backend adapter not yet implemented" in result["error"]
                )

    def test_retriever_service_n8n_credentials(self):
        """Test retriever_service with n8n credentials"""
        with patch("kbbridge.integrations.RetrievalCredentials") as mock_creds_class:
            with patch(
                "kbbridge.integrations.backend_adapter.BackendAdapterFactory"
            ) as mock_factory:
                mock_creds = Mock()
                mock_creds.validate.return_value = (True, None)
                mock_creds.endpoint = "https://n8n.com"
                mock_creds.api_key = "n8n-key"
                mock_creds.backend_type = "n8n"
                mock_creds_class.return_value = mock_creds

                # Mock factory to raise NotImplementedError for n8n
                mock_factory.create.side_effect = NotImplementedError(
                    "n8n backend adapter not yet implemented"
                )

                result = retriever_service(
                    resource_id="test-resource",
                    query="test query",
                    n8n_webhook_url="https://n8n.com",
                    n8n_api_key="n8n-key",
                )

                assert "error" in result
                assert "n8n backend adapter not yet implemented" in result["error"]

    def test_retriever_service_client_side_filtering_fallback(self):
        """Test retriever_service client-side filtering fallback"""
        with patch("kbbridge.integrations.RetrievalCredentials") as mock_creds_class:
            with patch(
                "kbbridge.integrations.backend_adapter.BackendAdapterFactory"
            ) as mock_factory:
                mock_creds = Mock()
                mock_creds.validate.return_value = (True, None)
                mock_creds.endpoint = "https://test.com"
                mock_creds.api_key = "test-key"
                mock_creds.backend_type = "dify"
                mock_creds_class.return_value = mock_creds

                mock_adapter = Mock()
                # First call returns empty (metadata filter didn't work)
                # Second call returns records with document names
                mock_adapter.search.side_effect = [
                    {"records": []},  # First call with metadata filter - empty
                    {
                        "records": [
                            {
                                "segment": {
                                    "content": "test content",
                                    "document": {"name": "test.pdf"},
                                }
                            },
                            {
                                "segment": {
                                    "content": "other content",
                                    "document": {"name": "other.pdf"},
                                }
                            },
                        ]
                    },  # Second call without filter - has records
                ]
                mock_factory.create.return_value = mock_adapter

                result = retriever_service(
                    resource_id="test-resource",
                    query="test query",
                    retrieval_endpoint="https://test.com",
                    retrieval_api_key="test-key",
                    document_name="test.pdf",
                )

                # Should have called search twice (fallback)
                assert mock_adapter.search.call_count == 2
                assert "result" in result
                # Should have filtered to only test.pdf
                assert len(result["result"]) == 1

    def test_retriever_service_resp_none(self):
        """Test retriever_service when resp is None"""
        with patch("kbbridge.integrations.RetrievalCredentials") as mock_creds_class:
            with patch(
                "kbbridge.integrations.backend_adapter.BackendAdapterFactory"
            ) as mock_factory:
                mock_creds = Mock()
                mock_creds.validate.return_value = (True, None)
                mock_creds.endpoint = "https://test.com"
                mock_creds.api_key = "test-key"
                mock_creds.backend_type = "dify"
                mock_creds_class.return_value = mock_creds

                mock_adapter = Mock()
                mock_adapter.search.return_value = None
                mock_factory.create.return_value = mock_adapter

                result = retriever_service(
                    resource_id="test-resource",
                    query="test query",
                    retrieval_endpoint="https://test.com",
                    retrieval_api_key="test-key",
                )

                assert "result" in result
                assert result["result"] == []
