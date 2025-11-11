"""
Additional coverage tests for reranking availability feature

Tests to cover missing lines in coverage report:
- server.py main() reranking check
- pipeline.py FileSearchStrategy with discover_factory
- pipeline.py DirectApproachProcessor with None credentials
- dify_adapter.py metadata filter building
"""

from unittest.mock import Mock, patch

import pytest

from kbbridge.config.config import Credentials as ConfigCredentials
from kbbridge.core.orchestration.models import Credentials as ModelCredentials
from kbbridge.core.orchestration.pipeline import (
    DirectApproachProcessor,
    FileSearchStrategy,
)


class TestServerMainRerankingCheck:
    """Test reranking check in server.py main() function"""

    @pytest.mark.asyncio
    async def test_main_reranking_enabled_log(self):
        """Test main() logs reranking enabled when credentials available"""
        import kbbridge.server as server_module

        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        with patch(
            "kbbridge.server.Config.get_default_credentials", return_value=creds
        ):
            with patch("kbbridge.server.logger") as mock_logger:
                with patch("kbbridge.server.mcp.run_http_async") as mock_run:
                    with patch(
                        "sys.argv", ["server.py", "--host", "0.0.0.0", "--port", "5210"]
                    ):
                        mock_run.side_effect = KeyboardInterrupt()  # Stop immediately

                        try:
                            await server_module.main()
                        except KeyboardInterrupt:
                            pass

                        # Verify reranking enabled log
                        info_calls = [
                            str(call) for call in mock_logger.info.call_args_list
                        ]
                        assert any(
                            "Reranking: ENABLED" in str(call) for call in info_calls
                        )

    @pytest.mark.asyncio
    async def test_main_reranking_disabled_log(self):
        """Test main() logs reranking disabled when credentials missing"""
        import kbbridge.server as server_module

        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url=None,
            rerank_model=None,
        )

        with patch(
            "kbbridge.server.Config.get_default_credentials", return_value=creds
        ):
            with patch("kbbridge.server.logger") as mock_logger:
                with patch("kbbridge.server.mcp.run_http_async") as mock_run:
                    with patch(
                        "sys.argv", ["server.py", "--host", "0.0.0.0", "--port", "5210"]
                    ):
                        mock_run.side_effect = KeyboardInterrupt()  # Stop immediately

                        try:
                            await server_module.main()
                        except KeyboardInterrupt:
                            pass

                        # Verify reranking disabled warning
                        warning_calls = [
                            str(call) for call in mock_logger.warning.call_args_list
                        ]
                        assert any(
                            "Reranking: DISABLED" in str(call) for call in warning_calls
                        )

    @pytest.mark.asyncio
    async def test_main_reranking_disabled_no_credentials(self):
        """Test main() logs reranking disabled when no default credentials"""
        import kbbridge.server as server_module

        with patch("kbbridge.server.Config.get_default_credentials", return_value=None):
            with patch("kbbridge.server.logger") as mock_logger:
                with patch("kbbridge.server.mcp.run_http_async") as mock_run:
                    with patch(
                        "sys.argv", ["server.py", "--host", "0.0.0.0", "--port", "5210"]
                    ):
                        mock_run.side_effect = KeyboardInterrupt()  # Stop immediately

                        try:
                            await server_module.main()
                        except KeyboardInterrupt:
                            pass

                        # Verify reranking disabled warning
                        warning_calls = [
                            str(call) for call in mock_logger.warning.call_args_list
                        ]
                        assert any(
                            "Reranking: DISABLED (no default credentials available)"
                            in str(call)
                            for call in warning_calls
                        )


class TestFileSearchStrategyRerankingCheck:
    """Test FileSearchStrategy reranking check with discover_factory"""

    def test_parallel_search_with_discover_factory_reranking_available(self):
        """Test FileSearchStrategy uses reranking when credentials available"""
        from kbbridge.integrations.retriever_base import FileHit

        mock_discover_factory = Mock()
        # Ensure mock doesn't have search_files so it's treated as discover_factory
        del mock_discover_factory.search_files
        mock_discover = Mock()
        mock_discover.retriever = Mock()
        mock_discover.retriever.build_metadata_filter.return_value = {}
        # Create mock FileHit objects
        mock_file = Mock(spec=FileHit)
        mock_file.file_name = "test.pdf"
        mock_file.score = 0.9
        mock_discover.return_value = [mock_file]
        mock_discover_factory.return_value = mock_discover

        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        strategy = FileSearchStrategy(mock_discover_factory, credentials=creds)

        with patch(
            "kbbridge.core.orchestration.pipeline.time.perf_counter"
        ) as mock_time:
            mock_time.side_effect = [0.0, 1.0]

            result = strategy.parallel_search(
                query="test query", resource_id="test-dataset"
            )

            # Verify discover was called with do_file_rerank=True
            mock_discover.assert_called_once()
            call_kwargs = mock_discover.call_args.kwargs
            assert call_kwargs["do_file_rerank"] is True

    def test_parallel_search_with_discover_factory_reranking_unavailable(self):
        """Test FileSearchStrategy disables reranking when credentials unavailable"""
        from kbbridge.integrations.retriever_base import FileHit

        mock_discover_factory = Mock()
        # Ensure mock doesn't have search_files so it's treated as discover_factory
        del mock_discover_factory.search_files
        mock_discover = Mock()
        mock_discover.retriever = Mock()
        mock_discover.retriever.build_metadata_filter.return_value = {}
        # Create mock FileHit objects
        mock_file = Mock(spec=FileHit)
        mock_file.file_name = "test.pdf"
        mock_file.score = 0.9
        mock_discover.return_value = [mock_file]
        mock_discover_factory.return_value = mock_discover

        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url=None,
            rerank_model=None,
        )

        strategy = FileSearchStrategy(mock_discover_factory, credentials=creds)

        with patch(
            "kbbridge.core.orchestration.pipeline.time.perf_counter"
        ) as mock_time:
            mock_time.side_effect = [0.0, 1.0]

            result = strategy.parallel_search(
                query="test query", resource_id="test-dataset"
            )

            # Verify discover was called with do_file_rerank=False
            mock_discover.assert_called_once()
            call_kwargs = mock_discover.call_args.kwargs
            assert call_kwargs["do_file_rerank"] is False

    def test_parallel_search_with_discover_factory_no_credentials(self):
        """Test FileSearchStrategy disables reranking when credentials are None"""
        from kbbridge.integrations.retriever_base import FileHit

        mock_discover_factory = Mock()
        # Ensure mock doesn't have search_files so it's treated as discover_factory
        del mock_discover_factory.search_files
        mock_discover = Mock()
        mock_discover.retriever = Mock()
        mock_discover.retriever.build_metadata_filter.return_value = {}
        # Create mock FileHit objects
        mock_file = Mock(spec=FileHit)
        mock_file.file_name = "test.pdf"
        mock_file.score = 0.9
        mock_discover.return_value = [mock_file]
        mock_discover_factory.return_value = mock_discover

        strategy = FileSearchStrategy(mock_discover_factory, credentials=None)

        with patch(
            "kbbridge.core.orchestration.pipeline.time.perf_counter"
        ) as mock_time:
            mock_time.side_effect = [0.0, 1.0]

            result = strategy.parallel_search(
                query="test query", resource_id="test-dataset"
            )

            # Verify discover was called with do_file_rerank=False
            mock_discover.assert_called_once()
            call_kwargs = mock_discover.call_args.kwargs
            assert call_kwargs["do_file_rerank"] is False


class TestDirectApproachProcessorNoneCredentials:
    """Test DirectApproachProcessor._retrieve_segments with None credentials"""

    def test_retrieve_segments_with_none_credentials(self):
        """Test _retrieve_segments disables reranking when credentials are None"""
        mock_retriever = Mock()
        mock_answer_extractor = Mock()

        processor = DirectApproachProcessor(
            mock_retriever, mock_answer_extractor, credentials=None
        )

        mock_retriever.retrieve.return_value = {
            "success": True,
            "result": [{"content": "test", "document_name": "test.pdf"}],
        }

        result = processor._retrieve_segments(
            resource_id="test-dataset",
            query="test query",
            metadata_filter={},
            score_threshold=None,
            top_k=10,
        )

        # Verify retriever was called with does_rerank=False
        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["does_rerank"] is False

    def test_retrieve_segments_with_none_credentials_uses_call(self):
        """Test _retrieve_segments uses call() when retriever doesn't have retrieve()"""
        mock_retriever = Mock()
        del mock_retriever.retrieve  # Remove retrieve method
        mock_retriever.call = Mock(return_value={"success": True, "result": []})
        mock_answer_extractor = Mock()

        processor = DirectApproachProcessor(
            mock_retriever, mock_answer_extractor, credentials=None
        )

        result = processor._retrieve_segments(
            resource_id="test-dataset",
            query="test query",
            metadata_filter={},
            score_threshold=None,
            top_k=10,
        )

        # Verify call() was used with does_rerank=False
        mock_retriever.call.assert_called_once()
        call_kwargs = mock_retriever.call.call_args.kwargs
        assert call_kwargs["does_rerank"] is False


class TestDifyBackendAdapterMetadataFilter:
    """Test DifyBackendAdapter metadata filter building"""

    def test_search_with_document_name_builds_filter(self):
        """Test search() builds metadata filter when document_name provided"""
        from kbbridge.integrations.dify import DifyBackendAdapter
        from kbbridge.integrations.dify.dify_credentials import DifyCredentials

        mock_retriever = Mock()
        mock_retriever.build_metadata_filter.return_value = {
            "conditions": [{"name": "document_name", "value": "test.pdf"}]
        }
        mock_retriever.call.return_value = {"records": []}
        mock_retriever.normalize_chunks.return_value = []
        mock_retriever.group_files.return_value = []

        creds = DifyCredentials(endpoint="https://test.com", api_key="test-key")
        adapter = DifyBackendAdapter(credentials=creds)

        with patch.object(adapter, "_get_retriever", return_value=mock_retriever):
            result = adapter.search(
                resource_id="test-dataset",
                query="test query",
                document_name="test.pdf",
            )

            # Verify metadata filter was built
            mock_retriever.build_metadata_filter.assert_called_once_with(
                document_name="test.pdf"
            )
            # Verify call() was made with metadata_filter
            mock_retriever.call.assert_called_once()
            call_kwargs = mock_retriever.call.call_args.kwargs
            assert call_kwargs["metadata_filter"] is not None

    def test_search_without_document_name_no_filter(self):
        """Test search() doesn't build filter when document_name is empty"""
        from kbbridge.integrations.dify import DifyBackendAdapter
        from kbbridge.integrations.dify.dify_credentials import DifyCredentials

        mock_retriever = Mock()
        mock_retriever.build_metadata_filter.return_value = None
        mock_retriever.call.return_value = {"records": []}
        mock_retriever.normalize_chunks.return_value = []
        mock_retriever.group_files.return_value = []

        creds = DifyCredentials(endpoint="https://test.com", api_key="test-key")
        adapter = DifyBackendAdapter(credentials=creds)

        with patch.object(adapter, "_get_retriever", return_value=mock_retriever):
            result = adapter.search(
                resource_id="test-dataset", query="test query", document_name=""
            )

            # Verify metadata filter was built (but returns None for empty document_name)
            mock_retriever.build_metadata_filter.assert_called_once_with(
                document_name=""
            )
            # Verify call() was made with metadata_filter=None
            mock_retriever.call.assert_called_once()
            call_kwargs = mock_retriever.call.call_args.kwargs
            assert call_kwargs["metadata_filter"] is None
