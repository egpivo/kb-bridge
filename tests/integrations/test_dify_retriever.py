"""
Tests for Dify Retriever Payload Structure Fix

These tests verify that the DifyRetriever correctly formats the API payload
with the nested retrieval_model structure expected by Dify API.
"""

from unittest.mock import Mock, patch

import pytest

from kbbridge.integrations.dify.dify_retriever import DifyRetriever


class TestDifyRetrieverPayloadStructure:
    """Test that DifyRetriever generates correct Dify API payload structure"""

    def test_call_generates_nested_retrieval_model(self):
        """Test that call() creates payload with nested retrieval_model"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
            timeout=30,
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"records": []}
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            retriever.call(
                query="test query",
                method="hybrid_search",
                top_k=20,
                does_rerank=True,
                reranking_provider_name="cohere",
                reranking_model_name="rerank-multilingual-v2.0",
                score_threshold_enabled=False,
            )

            # Verify the payload structure
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            # Should have query and retrieval_model at top level
            assert "query" in payload
            assert "retrieval_model" in payload
            assert payload["query"] == "test query"

            # retrieval_model should contain all search parameters
            model = payload["retrieval_model"]
            assert model["search_method"] == "hybrid_search"
            assert model["reranking_enable"] == True
            assert model["top_k"] == 20
            assert model["score_threshold_enabled"] == False

            # reranking_model should be nested
            assert "reranking_model" in model
            assert model["reranking_model"]["reranking_provider_name"] == "cohere"
            assert (
                model["reranking_model"]["reranking_model_name"]
                == "rerank-multilingual-v2.0"
            )

    def test_call_with_metadata_filter(self):
        """Test that metadata_filter becomes metadata_filtering_conditions and enables metadata"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
            timeout=30,
        )

        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock enable_metadata check (metadata is disabled, so it will enable)
            status_response = Mock()
            status_response.json.return_value = {"enabled": False}
            status_response.status_code = 200
            status_response.raise_for_status.return_value = None
            mock_get.return_value = status_response

            # Mock enable POST response
            enable_response = Mock()
            enable_response.status_code = 200
            enable_response.raise_for_status.return_value = None

            # Mock retrieve POST response
            retrieve_response = Mock()
            retrieve_response.json.return_value = {"records": []}
            retrieve_response.status_code = 200
            retrieve_response.raise_for_status.return_value = None

            # Set up mock_post to return different responses for enable vs retrieve
            def post_side_effect(*args, **kwargs):
                url = args[0] if args else kwargs.get("url", "")
                if "metadata/built-in/enable" in url:
                    return enable_response
                return retrieve_response

            mock_post.side_effect = post_side_effect

            retriever.call(
                query="test query",
                method="semantic_search",
                top_k=10,
                metadata_filter={
                    "conditions": [{"name": "document_name", "value": "test.pdf"}]
                },
            )

            # Verify enable_metadata was called (check status first)
            assert mock_get.call_count >= 1

            # Verify metadata_filtering_conditions is in retrieval_model
            # Find the retrieve call (not the enable call)
            retrieve_calls = [
                call
                for call in mock_post.call_args_list
                if "retrieve" in (call[0][0] if call[0] else "")
            ]
            assert len(retrieve_calls) > 0
            payload = retrieve_calls[0][1]["json"]
            model = payload["retrieval_model"]
            assert "metadata_filtering_conditions" in model
            assert model["metadata_filtering_conditions"] == {
                "conditions": [{"name": "document_name", "value": "test.pdf"}]
            }

    def test_call_with_optional_parameters(self):
        """Test that optional parameters are correctly nested"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
            timeout=30,
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"records": []}
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            retriever.call(
                query="test query",
                method="hybrid_search",
                top_k=20,
                score_threshold=0.7,
                weights=0.5,
            )

            # Verify optional parameters are in retrieval_model
            payload = mock_post.call_args[1]["json"]
            model = payload["retrieval_model"]
            assert model["score_threshold"] == 0.7
            assert model["weights"] == 0.5

    def test_call_error_logging(self):
        """Test that API errors are properly logged with response details"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
            timeout=30,
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "code": "index_not_found_exception",
                "message": "no such index [vector_index_test_node]",
                "status": 400,
            }
            mock_response.raise_for_status.side_effect = Exception("400 Client Error")
            mock_post.return_value = mock_response

            with pytest.raises(Exception):
                retriever.call(
                    query="test query",
                    method="semantic_search",
                    top_k=10,
                )

    def test_call_uses_correct_endpoint_format(self):
        """Test that the URL is correctly formatted"""
        retriever = DifyRetriever(
            endpoint="https://test.com/",  # with trailing slash
            api_key="test-key",
            dataset_id="test-dataset-id",
            timeout=30,
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"records": []}
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            retriever.call(query="test", method="semantic_search", top_k=10)

            # Verify URL is correctly formatted (no double slashes)
            call_url = mock_post.call_args[0][0]
            assert call_url == "https://test.com/v1/datasets/test-dataset-id/retrieve"

    def test_call_includes_authorization_header(self):
        """Test that Authorization header is correctly set"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-api-key-123",
            dataset_id="test-dataset",
            timeout=30,
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"records": []}
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            retriever.call(query="test", method="semantic_search", top_k=10)

            # Verify Authorization header
            headers = mock_post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-api-key-123"
            assert headers["Content-Type"] == "application/json"


class TestDifyRetrieverNormalizeChunks:
    """Test normalizing Dify API responses to ChunkHit objects"""

    def test_normalize_chunks_success(self):
        """Test normalizing valid response with records"""

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        response = {
            "records": [
                {
                    "segment": {
                        "content": "test content 1",
                        "document": {
                            "doc_metadata": {
                                "document_name": "doc1.pdf",
                            }
                        },
                    },
                    "score": 0.95,
                },
                {
                    "segment": {
                        "content": "test content 2",
                        "document": {
                            "doc_metadata": {
                                "document_name": "doc2.pdf",
                            }
                        },
                    },
                    "score": 0.85,
                },
            ]
        }

        chunks = retriever.normalize_chunks(response)

        assert len(chunks) == 2
        assert chunks[0].content == "test content 1"
        assert chunks[0].document_name == "doc1.pdf"
        assert chunks[0].score == 0.95
        assert chunks[1].content == "test content 2"

    def test_normalize_chunks_empty_content(self):
        """Test that chunks with empty content are skipped"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        response = {
            "records": [
                {
                    "segment": {
                        "content": "",  # Empty content
                        "document": {"doc_metadata": {"document_name": "doc1.pdf"}},
                    },
                    "score": 0.95,
                },
                {
                    "segment": {
                        "content": "valid content",
                        "document": {"doc_metadata": {"document_name": "doc2.pdf"}},
                    },
                    "score": 0.85,
                },
            ]
        }

        chunks = retriever.normalize_chunks(response)

        # Should skip the empty content chunk
        assert len(chunks) == 1
        assert chunks[0].content == "valid content"

    def test_normalize_chunks_malformed_record(self):
        """Test that malformed records are skipped with warning"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        response = {
            "records": [
                {
                    "segment": {
                        # Missing content
                        "document": {"doc_metadata": {"document_name": "doc1.pdf"}},
                    },
                    "score": 0.95,
                },
                {
                    "segment": {
                        "content": "valid content",
                        "document": {"doc_metadata": {"document_name": "doc2.pdf"}},
                    },
                    "score": 0.85,
                },
            ]
        }

        chunks = retriever.normalize_chunks(response)

        # Should skip the malformed record
        assert len(chunks) == 1
        assert chunks[0].content == "valid content"

    def test_normalize_chunks_exception_handling(self):
        """Test that exceptions during normalization return empty list"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        # Invalid response structure
        response = {"invalid": "structure"}

        chunks = retriever.normalize_chunks(response)

        # Should return empty list on error
        assert len(chunks) == 0


class TestDifyRetrieverGroupFiles:
    """Test grouping chunks by file"""

    def test_group_files_max_aggregation(self):
        """Test grouping files with max score aggregation"""
        from kbbridge.integrations.retriever_base import ChunkHit

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        chunks = [
            ChunkHit("content1", "doc1.pdf", score=0.9),
            ChunkHit("content2", "doc1.pdf", score=0.7),
            ChunkHit("content3", "doc2.pdf", score=0.8),
        ]

        files = retriever.group_files(chunks, agg="max")

        assert len(files) == 2
        # Should be sorted by score descending
        assert files[0].file_name == "doc1.pdf"
        assert files[0].score == 0.9  # max of 0.9 and 0.7
        assert files[1].file_name == "doc2.pdf"
        assert files[1].score == 0.8

    def test_group_files_mean_aggregation(self):
        """Test grouping files with mean score aggregation"""
        from kbbridge.integrations.retriever_base import ChunkHit

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        chunks = [
            ChunkHit("content1", "doc1.pdf", score=0.9),
            ChunkHit("content2", "doc1.pdf", score=0.7),
        ]

        files = retriever.group_files(chunks, agg="mean")

        assert len(files) == 1
        assert files[0].score == 0.8  # (0.9 + 0.7) / 2

    def test_group_files_sum_aggregation(self):
        """Test grouping files with sum score aggregation"""
        from kbbridge.integrations.retriever_base import ChunkHit

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        chunks = [
            ChunkHit("content1", "doc1.pdf", score=0.5),
            ChunkHit("content2", "doc1.pdf", score=0.3),
        ]

        files = retriever.group_files(chunks, agg="sum")

        assert len(files) == 1
        assert files[0].score == 0.8  # 0.5 + 0.3

    def test_group_files_unknown_aggregation(self):
        """Test that unknown aggregation defaults to max"""
        from kbbridge.integrations.retriever_base import ChunkHit

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        chunks = [
            ChunkHit("content1", "doc1.pdf", score=0.9),
            ChunkHit("content2", "doc1.pdf", score=0.7),
        ]

        files = retriever.group_files(chunks, agg="unknown")

        # Should default to max
        assert files[0].score == 0.9


class TestDifyRetrieverBuildMetadataFilter:
    """Test building metadata filters"""

    def test_build_metadata_filter_with_document_name_only(self):
        """Test building filter with document name only"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        filter_dict = retriever.build_metadata_filter(document_name="legal.pdf")

        assert filter_dict is not None
        assert "conditions" in filter_dict
        assert len(filter_dict["conditions"]) == 1
        assert filter_dict["conditions"][0]["name"] == "document_name"
        assert filter_dict["conditions"][0]["value"] == "legal.pdf"

    def test_build_metadata_filter_with_document_name(self):
        """Test building filter with document name"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        filter_dict = retriever.build_metadata_filter(document_name="report.pdf")

        assert filter_dict is not None
        assert len(filter_dict["conditions"]) == 1
        assert filter_dict["conditions"][0]["name"] == "document_name"
        assert filter_dict["conditions"][0]["value"] == "report.pdf"

    def test_build_metadata_filter_with_both(self):
        """Test building filter with both source path and document name"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        filter_dict = retriever.build_metadata_filter(document_name="report.pdf")

        assert filter_dict is not None
        assert len(filter_dict["conditions"]) == 1
        assert filter_dict["conditions"][0]["name"] == "document_name"

    def test_build_metadata_filter_empty(self):
        """Test that empty filters return None"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        filter_dict = retriever.build_metadata_filter(document_name="")

        assert filter_dict is None

    def test_build_metadata_filter_whitespace_only(self):
        """Test that whitespace-only filters return None"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        filter_dict = retriever.build_metadata_filter(document_name="  ")

        assert filter_dict is None


class TestDifyRetrieverListFiles:
    """Test listing files from dataset"""

    def test_list_files_success(self):
        """Test successfully listing files"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [
                    {"name": "doc1.pdf"},
                    {"name": "doc2.pdf"},
                    {"name": None},  # Should be filtered out
                ]
            }
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            files = retriever.list_files(dataset_id="test-dataset")

            assert len(files) == 2
            assert "doc1.pdf" in files
            assert "doc2.pdf" in files

    def test_list_files_success(self):
        """Test successfully listing files"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [
                    {"name": "doc1.pdf"},
                    {"name": "doc2.pdf"},
                ]
            }
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            files = retriever.list_files(dataset_id="test-dataset")

            # Should return all files
            assert len(files) == 2
            assert "doc1.pdf" in files
            assert "doc2.pdf" in files

    def test_list_files_exception_handling(self):
        """Test that list_files returns empty list on exception"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.get", side_effect=Exception("Network error")):
            files = retriever.list_files(dataset_id="test-dataset")

            # Should return empty list on error
            assert files == []


class TestMakeRetriever:
    """Test retriever factory function"""

    def test_make_retriever_dify(self):
        """Test creating Dify retriever"""
        from kbbridge.integrations.dify.dify_retriever import make_retriever

        retriever = make_retriever(
            "dify",
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        assert isinstance(retriever, DifyRetriever)
        assert retriever.endpoint == "https://test.com"
        assert retriever.dataset_id == "test-dataset"

    def test_make_retriever_dify_retriever_alias(self):
        """Test creating Dify retriever with alias"""
        from kbbridge.integrations.dify.dify_retriever import make_retriever

        retriever = make_retriever(
            "dify_retriever",
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        assert isinstance(retriever, DifyRetriever)

    def test_make_retriever_unknown_type(self):
        """Test that unknown retriever type raises ValueError"""
        from kbbridge.integrations.dify.dify_retriever import make_retriever

        with pytest.raises(ValueError, match="Unknown retriever type"):
            make_retriever(
                "unknown",
                endpoint="https://test.com",
                api_key="test-key",
                dataset_id="test-dataset",
            )


class TestDifyRetrieverCallEdgeCases:
    """Test edge cases for call method"""

    def test_call_with_zero_top_k(self):
        """Test that zero top_k defaults to 20"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"records": []}
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            retriever.call(query="test", method="semantic_search", top_k=0)

            payload = mock_post.call_args[1]["json"]
            assert payload["retrieval_model"]["top_k"] == 20

    def test_call_with_negative_top_k(self):
        """Test that negative top_k defaults to 20"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"records": []}
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            retriever.call(query="test", method="semantic_search", top_k=-5)

            payload = mock_post.call_args[1]["json"]
            assert payload["retrieval_model"]["top_k"] == 20

    def test_call_error_with_json_response(self):
        """Test error handling when response has JSON error"""
        import requests

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "Bad request"}
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "400 Bad Request"
            )
            mock_post.return_value = mock_response

            with pytest.raises(requests.exceptions.HTTPError):
                retriever.call(query="test", method="semantic_search", top_k=10)

    def test_call_error_with_text_response(self):
        """Test error handling when response has text error (JSON parsing fails)"""
        import requests

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.json.side_effect = Exception("JSON decode error")
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "500 Internal Server Error"
            )
            mock_post.return_value = mock_response

            with pytest.raises(requests.exceptions.HTTPError):
                retriever.call(query="test", method="semantic_search", top_k=10)


class TestDifyRetrieverMetadata:
    """Test metadata enable and status check methods"""

    def test_enable_metadata_success(self):
        """Test successfully enabling metadata when disabled"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock check_metadata_status to return disabled status
            status_response = Mock()
            status_response.json.return_value = {"enabled": False}
            status_response.status_code = 200
            status_response.raise_for_status.return_value = None
            mock_get.return_value = status_response

            # Mock enable_metadata POST response
            enable_response = Mock()
            enable_response.status_code = 200
            enable_response.raise_for_status.return_value = None
            mock_post.return_value = enable_response

            result = retriever.enable_metadata()

            assert result is True
            mock_get.assert_called_once()  # Should check status first
            mock_post.assert_called_once()  # Should then enable
            call_url = mock_post.call_args[0][0]
            assert (
                call_url
                == "https://test.com/v1/datasets/test-dataset/metadata/built-in/enable"
            )
            headers = mock_post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"

    def test_enable_metadata_already_enabled(self):
        """Test that enable_metadata skips if already enabled"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Mock check_metadata_status to return enabled status
            status_response = Mock()
            status_response.json.return_value = {"enabled": True}
            status_response.status_code = 200
            status_response.raise_for_status.return_value = None
            mock_get.return_value = status_response

            result = retriever.enable_metadata()

            assert result is True
            mock_get.assert_called_once()  # Should check status
            mock_post.assert_not_called()  # Should NOT call enable if already enabled

    def test_enable_metadata_force(self):
        """Test that force=True enables even if already enabled"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.post") as mock_post:
            enable_response = Mock()
            enable_response.status_code = 200
            enable_response.raise_for_status.return_value = None
            mock_post.return_value = enable_response

            result = retriever.enable_metadata(force=True)

            assert result is True
            mock_post.assert_called_once()  # Should call enable even if already enabled

    def test_enable_metadata_http_error(self):
        """Test enable_metadata returns False on HTTP error"""
        import requests

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "Bad request"}
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "400 Bad Request"
            )
            mock_post.return_value = mock_response

            result = retriever.enable_metadata()

            assert result is False

    def test_enable_metadata_http_error_text_response(self):
        """Test enable_metadata handles text error response"""
        import requests

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.json.side_effect = Exception("JSON decode error")
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "500 Internal Server Error"
            )
            mock_post.return_value = mock_response

            result = retriever.enable_metadata()

            assert result is False

    def test_enable_metadata_exception(self):
        """Test enable_metadata returns False on general exception"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.post", side_effect=Exception("Network error")):
            result = retriever.enable_metadata()

            assert result is False

    def test_check_metadata_status_success(self):
        """Test successfully checking metadata status"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        expected_status = {
            "enabled": True,
            "built_in_metadata": ["document_name", "document_id"],
        }

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = expected_status
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = retriever.check_metadata_status()

            assert result == expected_status
            mock_get.assert_called_once()
            call_url = mock_get.call_args[0][0]
            assert (
                call_url
                == "https://test.com/v1/datasets/test-dataset/metadata/built-in"
            )
            headers = mock_get.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"

    def test_check_metadata_status_http_error(self):
        """Test check_metadata_status returns None on HTTP error"""
        import requests

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"error": "Not found"}
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "404 Not Found"
            )
            mock_get.return_value = mock_response

            result = retriever.check_metadata_status()

            assert result is None

    def test_check_metadata_status_http_error_text_response(self):
        """Test check_metadata_status handles text error response"""
        import requests

        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.json.side_effect = Exception("JSON decode error")
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "500 Internal Server Error"
            )
            mock_get.return_value = mock_response

            result = retriever.check_metadata_status()

            assert result is None

    def test_check_metadata_status_exception(self):
        """Test check_metadata_status returns None on general exception"""
        retriever = DifyRetriever(
            endpoint="https://test.com",
            api_key="test-key",
            dataset_id="test-dataset",
        )

        with patch("requests.get", side_effect=Exception("Network error")):
            result = retriever.check_metadata_status()

            assert result is None
