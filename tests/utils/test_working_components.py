"""
Test working_components module functionality
"""

from unittest.mock import Mock, patch

from kbbridge.core.orchestration.models import Credentials
from kbbridge.utils.working_components import (
    KnowledgeBaseRetriever,
    WorkingComponentFactory,
    WorkingDatasetProcessor,
    WorkingIntentionExtractor,
    WorkingResultFormatter,
    format_search_results,
)


class TestFormatSearchResults:
    """Test format_search_results function"""

    def test_format_search_results_empty(self):
        """Test formatting empty results"""
        result = format_search_results([])

        assert isinstance(result, dict)
        assert "result" in result
        assert result["result"] == []

    def test_format_search_results_valid(self):
        """Test formatting valid results"""
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

    def test_format_search_results_dict_input(self):
        """Test formatting when input is a dict"""
        results = {
            "records": [
                {
                    "segment": {
                        "content": "Test content",
                        "document": {"doc_metadata": {"document_name": "test.pdf"}},
                    }
                }
            ]
        }

        result = format_search_results(results)

        assert isinstance(result, dict)
        assert "result" in result
        assert len(result["result"]) == 1

    def test_format_search_results_malformed(self):
        """Test formatting malformed results"""
        results = [{"invalid": "structure"}]

        result = format_search_results(results)

        assert isinstance(result, dict)
        assert "result" in result

    def test_format_search_results_empty_metadata(self):
        """Test formatting results with empty metadata (covers line 38)"""
        results = [
            {
                "records": [
                    {
                        "segment": {
                            "content": "Test content",
                            "document": {"doc_metadata": None},  # Empty metadata
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
        assert result["result"][0]["document_name"] == ""

    def test_format_search_results_exception_in_processing(self):
        """Test formatting results with exception during processing (covers lines 42-44)"""
        # Create data that will cause an exception during processing
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

        # Should handle exception gracefully and return empty result
        assert isinstance(result, dict)
        assert "result" in result
        assert isinstance(result["result"], list)

    def test_format_search_results_top_level_exception(self):
        """Test formatting results with top-level exception (covers lines 49-51)"""
        # Create data that will cause a top-level exception
        problematic_results = None  # This will cause a top-level exception

        result = format_search_results(problematic_results)

        # Should handle exception gracefully
        assert isinstance(result, dict)
        assert "result" in result
        assert isinstance(result["result"], list)

    def test_format_search_results_with_metadata_filter(self):
        """Test formatting results with metadata filter (covers lines 130, 132, 134)"""
        # This test covers the optional parameter handling in retrieve method
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"data": []}
            mock_post.return_value.status_code = 200

            result = retriever.retrieve(
                dataset_id="test-dataset",
                query="test query",
                metadata_filter={"key": "value"},
                score_threshold=0.5,
                weights={"content": 0.8},
            )

            # Verify the request was made with optional parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            # Check the nested retrieval_model structure
            assert "retrieval_model" in payload
            retrieval_model = payload["retrieval_model"]
            assert "metadata_filtering_conditions" in retrieval_model
            assert "score_threshold" in retrieval_model
            assert "weights" in retrieval_model

    def test_working_intention_extractor_with_auth_token(self):
        """Test WorkingIntentionExtractor with auth token (covers line 201)"""
        extractor = WorkingIntentionExtractor(
            llm_api_url="https://llm.com", llm_model="gpt-4", llm_api_token="test-token"
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"intention": "test", "refined_query": "refined"}'
                        }
                    }
                ]
            }
            mock_post.return_value.status_code = 200

            extractor.extract_intention("test query")

            # Verify Authorization header was set
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-token"

    def test_extract_user_intention_json_parsing(self):
        """Test extract_intention with JSON parsing (covers lines 227-242)"""
        extractor = WorkingIntentionExtractor(
            llm_api_url="https://llm.com", llm_model="gpt-4"
        )

        with patch("requests.post") as mock_post:
            # Test successful JSON parsing
            mock_post.return_value.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"intention": "test intention", "refined_query": "refined query"}'
                        }
                    }
                ]
            }
            mock_post.return_value.status_code = 200

            result = extractor.extract_intention("test query")

            assert result["success"] is True
            assert result["intention"] == "test intention"
            assert result["updated_query"] == "refined query"

    def test_extract_user_intention_json_decode_error(self):
        """Test extract_intention with JSON decode error (covers lines 240-246)"""
        extractor = WorkingIntentionExtractor(
            llm_api_url="https://llm.com", llm_model="gpt-4"
        )

        with patch("requests.post") as mock_post:
            # Test JSON decode error fallback
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "invalid json"}}]
            }
            mock_post.return_value.status_code = 200

            result = extractor.extract_intention("test query")

            assert result["success"] is True
            assert (
                "User wants to find information about: test query"
                in result["intention"]
            )
            assert result["updated_query"] == "test query"

    def test_extract_user_intention_exception(self):
        """Test extract_intention with exception (covers line 248)"""
        extractor = WorkingIntentionExtractor(
            llm_api_url="https://llm.com", llm_model="gpt-4"
        )

        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Network error")

            result = extractor.extract_intention("test query")

            # The method returns success=True even on exception (fallback behavior)
            assert result["success"] is True
            assert (
                "User wants to find information about: test query"
                in result["intention"]
            )
            assert result["updated_query"] == "test query"

    def test_process_datasets_with_results(self):
        """Test process_datasets with results (covers lines 274-300)"""
        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )
        factory = WorkingComponentFactory()
        components = factory.create_components(credentials)

        # Create a mock config
        from kbbridge.core.orchestration.models import ProcessingConfig

        config = ProcessingConfig(dataset_info=[{"id": "test"}], query="test query")

        processor = WorkingDatasetProcessor(components, config, credentials)

        dataset_pairs = [
            {"id": "dataset1", "source_path": "path1"},
            {"id": "dataset2", "source_path": "path2"},
        ]

        with patch.object(processor, "_process_naive_approach") as mock_naive:
            with patch.object(processor, "_process_advanced_approach") as mock_advanced:
                mock_naive.return_value = {"candidates": [{"content": "naive result"}]}
                mock_advanced.return_value = {
                    "candidates": [{"content": "advanced result"}]
                }

                dataset_results, all_candidates = processor.process_datasets(
                    dataset_pairs, "test query"
                )

                assert len(dataset_results) == 2
                assert len(all_candidates) == 4  # 2 datasets * 2 approaches
                assert dataset_results[0]["dataset_id"] == "dataset1"
                assert dataset_results[1]["dataset_id"] == "dataset2"

    def test_process_naive_approach(self):
        """Test _process_naive_approach (covers lines 304-326)"""
        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )
        factory = WorkingComponentFactory()
        components = factory.create_components(credentials)

        from kbbridge.core.orchestration.models import ProcessingConfig

        config = ProcessingConfig(dataset_info=[{"id": "test"}], query="test query")

        processor = WorkingDatasetProcessor(components, config, credentials)

        with patch.object(components["retriever"], "retrieve") as mock_retrieve:
            mock_retrieve.return_value = {
                "result": [{"content": "test content", "document_name": "test.pdf"}]
            }

            result = processor._process_naive_approach("test-dataset", "test query")

            assert "candidates" in result
            assert len(result["candidates"]) == 1
            assert result["candidates"][0]["content"] == "test content"
            assert result["candidates"][0]["score"] == 1.0
            assert result["candidates"][0]["source"] == "dify_naive_search"

    def test_process_advanced_approach(self):
        """Test _process_advanced_approach (covers lines 330-354)"""
        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )
        factory = WorkingComponentFactory()
        components = factory.create_components(credentials)

        from kbbridge.core.orchestration.models import ProcessingConfig

        config = ProcessingConfig(dataset_info=[{"id": "test"}], query="test query")

        processor = WorkingDatasetProcessor(components, config, credentials)

        with patch.object(components["retriever"], "retrieve") as mock_retrieve:
            mock_retrieve.return_value = {
                "result": [{"content": "test content", "document_name": "test.pdf"}]
            }

            result = processor._process_advanced_approach("test-dataset", "test query")

            assert "candidates" in result
            assert len(result["candidates"]) == 1
            assert result["candidates"][0]["content"] == "test content"
            assert result["candidates"][0]["score"] == 0.8
            assert result["candidates"][0]["source"] == "dify_advanced_search"

    def test_format_final_answer_with_naive_results(self):
        """Test format_final_answer with naive results (covers lines 379-383)"""
        candidates = [
            {"content": "Result 1", "source": "dify_naive_search"},
            {"content": "Result 2", "source": "dify_naive_search"},
            {"content": "Result 3", "source": "dify_naive_search"},
            {
                "content": "Result 4",
                "source": "dify_naive_search",
            },  # Should be limited to top 3
        ]

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        result = WorkingResultFormatter.format_final_answer(
            candidates, "test query", credentials
        )

        assert "Result 1" in result
        assert "Result 2" in result
        assert "Result 3" in result
        assert "Result 4" not in result  # Should be limited to top 3

    def test_format_final_answer_with_additional_results(self):
        """Test format_final_answer with additional results (covers lines 386-394)"""
        # Only 1 naive result so advanced results will be shown (len(answer_parts) < 3)
        candidates = [
            {"content": "Naive Result 1", "source": "dify_naive_search"},
            {"content": "Advanced Result 1", "source": "dify_advanced_search"},
            {"content": "Advanced Result 2", "source": "dify_advanced_search"},
            {
                "content": "Advanced Result 3",
                "source": "dify_advanced_search",
            },  # Should be limited to top 2 additional
        ]

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        result = WorkingResultFormatter.format_final_answer(
            candidates, "test query", credentials
        )

        assert "Naive Result 1" in result
        assert "Advanced Result 1" in result
        assert "Advanced Result 2" in result
        assert (
            "Advanced Result 3" not in result
        )  # Should be limited to top 2 additional

    def test_format_final_answer_duplicate_content(self):
        """Test format_final_answer with duplicate content (covers lines 391-393)"""
        candidates = [
            {"content": "Duplicate Content", "source": "dify_naive_search"},
            {
                "content": "Duplicate Content",
                "source": "dify_advanced_search",
            },  # Should be filtered out
            {"content": "Unique Content", "source": "dify_advanced_search"},
        ]

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        result = WorkingResultFormatter.format_final_answer(
            candidates, "test query", credentials
        )

        # Count occurrences of "Duplicate Content"
        duplicate_count = result.count("Duplicate Content")
        assert duplicate_count == 1  # Should only appear once
        assert "Unique Content" in result


class TestKnowledgeBaseRetriever:
    """Test KnowledgeBaseRetriever class"""

    def test_init(self):
        """Test retriever initialization"""
        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        retriever = KnowledgeBaseRetriever(
            credentials.retrieval_endpoint, credentials.retrieval_api_key
        )

        assert retriever.endpoint == "https://test.com"
        assert retriever.api_key == "test-key"

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_success(self, mock_post):
        """Test successful retrieval"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [
                {
                    "segment": {
                        "content": "Test content",
                        "document": {"doc_metadata": {"document_name": "test.pdf"}},
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )
        retriever = KnowledgeBaseRetriever(
            credentials.retrieval_endpoint, credentials.retrieval_api_key
        )

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
            search_method="hybrid_search",
            does_rerank=False,
            top_k=10,
            reranking_provider_name="",
            reranking_model_name="",
            score_threshold_enabled=False,
        )

        assert isinstance(result, dict)
        # Now returns raw Dify response with 'records' field
        assert "records" in result

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_api_error(self, mock_post):
        """Test retrieval with API error"""
        import requests

        mock_post.side_effect = requests.exceptions.HTTPError("400 Client Error")

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )
        retriever = KnowledgeBaseRetriever(
            credentials.retrieval_endpoint, credentials.retrieval_api_key
        )

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
            search_method="hybrid_search",
            does_rerank=False,
            top_k=10,
            reranking_provider_name="",
            reranking_model_name="",
            score_threshold_enabled=False,
        )

        assert isinstance(result, dict)
        assert "error" in result

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_network_error(self, mock_post):
        """Test retrieval with network error"""
        mock_post.side_effect = Exception("Network error")

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )
        retriever = KnowledgeBaseRetriever(
            credentials.retrieval_endpoint, credentials.retrieval_api_key
        )

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
            search_method="hybrid_search",
            does_rerank=False,
            top_k=10,
            reranking_provider_name="",
            reranking_model_name="",
            score_threshold_enabled=False,
        )

        assert isinstance(result, dict)
        assert "error" in result


class TestWorkingComponentFactory:
    """Test WorkingComponentFactory class"""

    def test_create_components(self):
        """Test factory component creation"""
        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        components = WorkingComponentFactory.create_components(credentials)

        assert isinstance(components, dict)
        assert "retriever" in components
        assert "intention_extractor" in components


class TestWorkingIntentionExtractor:
    """Test WorkingIntentionExtractor class"""

    def test_init(self):
        """Test extractor initialization"""
        extractor = WorkingIntentionExtractor("https://llm.com", "gpt-4")
        assert extractor is not None

    def test_extract_intention(self):
        """Test intention extraction"""
        extractor = WorkingIntentionExtractor("https://llm.com", "gpt-4")

        result = extractor.extract_intention("test query")

        assert isinstance(result, dict)
        assert "intention" in result


class TestWorkingDatasetProcessor:
    """Test WorkingDatasetProcessor class"""

    def test_init(self):
        """Test processor initialization"""
        processor = WorkingDatasetProcessor(
            {},
            {},
            Credentials(
                retrieval_endpoint="https://test.com",
                retrieval_api_key="test-key",
                llm_api_url="https://llm.com",
                llm_model="gpt-4",
            ),
        )
        assert processor is not None


class TestWorkingResultFormatter:
    """Test WorkingResultFormatter class"""

    def test_init(self):
        """Test formatter initialization"""
        formatter = WorkingResultFormatter()
        assert formatter is not None

    def test_format_final_answer(self):
        """Test formatting final answer"""
        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        candidates = [{"content": "test content", "score": 0.9}]

        answer = WorkingResultFormatter.format_final_answer(
            candidates, "test query", credentials
        )

        assert isinstance(answer, str)
        assert len(answer) > 0


class TestWorkingComponentsEdgeCases:
    """Test edge cases and error conditions"""

    def test_format_search_results_none(self):
        """Test formatting None results"""
        result = format_search_results(None)

        assert isinstance(result, dict)
        assert "result" in result

    def test_knowledge_base_retriever_invalid_credentials(self):
        """Test retriever with invalid credentials"""
        credentials = Credentials(
            retrieval_endpoint="",
            retrieval_api_key="",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        retriever = KnowledgeBaseRetriever(
            credentials.retrieval_endpoint, credentials.retrieval_api_key
        )

        # Should handle invalid credentials gracefully
        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
            search_method="hybrid_search",
            does_rerank=False,
            top_k=10,
            reranking_provider_name="",
            reranking_model_name="",
            score_threshold_enabled=False,
        )

        assert isinstance(result, dict)

    def test_working_result_formatter_invalid_input(self):
        """Test formatter with invalid input"""
        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        answer = WorkingResultFormatter.format_final_answer(
            [], "test query", credentials
        )

        assert isinstance(answer, str)
        assert "No relevant information found" in answer


class TestBuildMetadataFilter:
    """Test build_metadata_filter method"""

    def test_build_metadata_filter_with_document_name(self):
        """Test building metadata filter with document name"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.build_metadata_filter(document_name="test.pdf")

        assert result is not None
        assert "conditions" in result
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["name"] == "document_name"
        assert result["conditions"][0]["comparison_operator"] == "contains"
        assert result["conditions"][0]["value"] == "test.pdf"
        assert result["logical_operator"] == "and"

    def test_build_metadata_filter_with_document_name(self):
        """Test building metadata filter with document name (covers lines 95-102)"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.build_metadata_filter(document_name="test.pdf")

        assert result is not None
        assert "conditions" in result
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["name"] == "document_name"
        assert result["conditions"][0]["comparison_operator"] == "contains"
        assert result["conditions"][0]["value"] == "test.pdf"
        assert result["logical_operator"] == "and"

    def test_build_metadata_filter_with_document_name_only(self):
        """Test building metadata filter with document name only"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.build_metadata_filter(document_name="test.pdf")

        assert result is not None
        assert "conditions" in result
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["name"] == "document_name"
        assert result["logical_operator"] == "and"

    def test_build_metadata_filter_empty(self):
        """Test building metadata filter with empty values"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.build_metadata_filter(document_name="")

        assert result is None

    def test_build_metadata_filter_whitespace(self):
        """Test building metadata filter with whitespace values"""
        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.build_metadata_filter(document_name="   ")

        assert result is None


class TestTopKValidation:
    """Test top_k validation edge cases"""

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_with_negative_top_k(self, mock_post):
        """Test retrieve with negative top_k (covers lines 152-154)"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": []}
        mock_post.return_value = mock_response

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
            top_k=-5,  # Negative value should default to 10
        )

        # Check that the payload was sent with top_k=10
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["retrieval_model"]["top_k"] == 10

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_with_zero_top_k(self, mock_post):
        """Test retrieve with zero top_k"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": []}
        mock_post.return_value = mock_response

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
            top_k=0,  # Zero value should default to 10
        )

        # Check that the payload was sent with top_k=10
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["retrieval_model"]["top_k"] == 10

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_with_string_top_k(self, mock_post):
        """Test retrieve with string top_k"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"records": []}
        mock_post.return_value = mock_response

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
            top_k="invalid",  # Invalid value should default to 10
        )

        # Check that the payload was sent with top_k=10
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["retrieval_model"]["top_k"] == 10


class TestHTTPErrorHandling:
    """Test HTTP error handling branches"""

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_http_error_with_json_response(self, mock_post):
        """Test HTTPError with JSON error content (covers lines 201-208)"""
        import requests

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.reason = "Bad Request"
        mock_response.json.return_value = {"error": "Invalid dataset ID"}

        http_error = requests.exceptions.HTTPError("400 Client Error")
        http_error.response = mock_response

        mock_post.side_effect = http_error

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
        )

        assert result["error"] is True
        assert result["status_code"] == 400
        assert result["reason"] == "Bad Request"
        assert result["error_content"] == "Invalid dataset ID"

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_http_error_with_invalid_json(self, mock_post):
        """Test HTTPError with invalid JSON (covers lines 206-208)"""
        import requests

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.text = "Server error text"
        mock_response.json.side_effect = ValueError("Invalid JSON")

        http_error = requests.exceptions.HTTPError("500 Server Error")
        http_error.response = mock_response

        mock_post.side_effect = http_error

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
        )

        assert result["error"] is True
        assert result["status_code"] == 500
        assert result["error_content"] == "Server error text"

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_http_error_with_response_in_scope(self, mock_post):
        """Test HTTPError with response in outer scope (covers lines 210-212)"""
        import requests

        # Create a mock that returns a response then raises an error
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.reason = "Forbidden"
        mock_response.json.return_value = {"data": []}

        # First call: return response successfully
        # Second call within raise_for_status: raise HTTPError without e.response
        call_count = [0]

        def side_effect_func(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_response
            return mock_response

        mock_post.side_effect = side_effect_func

        # Make raise_for_status raise an HTTPError without e.response set
        http_error = requests.exceptions.HTTPError("403 Forbidden")
        # Explicitly don't set http_error.response to None or leave it unset
        mock_response.raise_for_status.side_effect = http_error

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
        )

        assert result["error"] is True
        assert result["status_code"] == 403
        assert result["reason"] == "Forbidden"
        assert result["error_content"] == "403 Forbidden"


class TestRequestExceptions:
    """Test various request exception types"""

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_timeout_error(self, mock_post):
        """Test Timeout exception (covers line 230)"""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
        )

        assert result["error"] is True
        assert "timed out" in result["error_message"]
        assert "url" in result
        assert "debug_payload" in result

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_connection_error(self, mock_post):
        """Test ConnectionError exception (covers line 239)"""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
        )

        assert result["error"] is True
        assert "Connection failed" in result["error_message"]
        assert "url" in result
        assert "debug_payload" in result

    @patch("kbbridge.utils.working_components.requests.post")
    def test_retrieve_json_decode_error(self, mock_post):
        """Test JSONDecodeError exception (covers line 257)"""
        import json

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response

        retriever = KnowledgeBaseRetriever("https://test.com", "test-key")

        result = retriever.retrieve(
            dataset_id="test-dataset",
            query="test query",
        )

        assert result["error"] is True
        assert "Invalid JSON response" in result["error_message"]
        assert "url" in result
        assert "debug_payload" in result


class TestFormatSearchResultsExceptions:
    """Test exception handling in format_search_results"""

    def test_format_search_results_segment_get_exception(self):
        """Test exception during segment extraction (covers lines 43-45)"""
        # Create a record that will raise an exception when accessing segment
        class BadRecord:
            def get(self, key):
                raise ValueError("Simulated error")

        results = [{"records": [BadRecord()]}]

        result = format_search_results(results)

        # Should handle exception gracefully and skip the record
        assert isinstance(result, dict)
        assert "result" in result
        assert len(result["result"]) == 0

    def test_format_search_results_top_level_exception(self):
        """Test top-level exception handling (covers lines 50-52)"""
        # Pass something that will cause exception during initial processing
        class BadIterable:
            def __iter__(self):
                raise TypeError("Cannot iterate")

        result = format_search_results(BadIterable())

        # Should handle exception and return error info
        assert isinstance(result, dict)
        assert "result" in result
        assert result["result"] == []
        assert "format_error" in result
