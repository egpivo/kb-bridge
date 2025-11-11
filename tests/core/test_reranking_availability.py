from unittest.mock import AsyncMock, Mock, patch

import pytest

from kbbridge.config.config import Credentials as ConfigCredentials
from kbbridge.core.orchestration.models import Credentials as ModelCredentials
from kbbridge.core.orchestration.pipeline import DirectApproachProcessor


class TestConfigCredentialsRerankingAvailability:
    """Test is_reranking_available() in config.Credentials"""

    def test_reranking_available_with_both_credentials(self):
        """Test reranking is available when both rerank_url and rerank_model are provided"""
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        assert creds.is_reranking_available() is True

    def test_reranking_unavailable_missing_rerank_url(self):
        """Test reranking is unavailable when rerank_url is missing"""
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url=None,
            rerank_model="test-model",
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_missing_rerank_model(self):
        """Test reranking is unavailable when rerank_model is missing"""
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url="https://rerank.com",
            rerank_model=None,
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_missing_both(self):
        """Test reranking is unavailable when both rerank credentials are missing"""
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url=None,
            rerank_model=None,
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_empty_rerank_url(self):
        """Test reranking is unavailable when rerank_url is empty string"""
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url="",
            rerank_model="test-model",
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_empty_rerank_model(self):
        """Test reranking is unavailable when rerank_model is empty string"""
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url="https://rerank.com",
            rerank_model="",
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_empty_both(self):
        """Test reranking is unavailable when both rerank credentials are empty strings"""
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url="",
            rerank_model="",
        )

        assert creds.is_reranking_available() is False


class TestModelCredentialsRerankingAvailability:
    """Test is_reranking_available() in models.Credentials"""

    def test_reranking_available_with_both_credentials(self):
        """Test reranking is available when both rerank_url and rerank_model are provided"""
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        assert creds.is_reranking_available() is True

    def test_reranking_unavailable_missing_rerank_url(self):
        """Test reranking is unavailable when rerank_url is missing"""
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url=None,
            rerank_model="test-model",
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_missing_rerank_model(self):
        """Test reranking is unavailable when rerank_model is missing"""
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url="https://rerank.com",
            rerank_model=None,
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_missing_both(self):
        """Test reranking is unavailable when both rerank credentials are missing"""
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url=None,
            rerank_model=None,
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_empty_rerank_url(self):
        """Test reranking is unavailable when rerank_url is empty string"""
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url="",
            rerank_model="test-model",
        )

        assert creds.is_reranking_available() is False

    def test_reranking_unavailable_empty_rerank_model(self):
        """Test reranking is unavailable when rerank_model is empty string"""
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url="https://rerank.com",
            rerank_model="",
        )

        assert creds.is_reranking_available() is False


class TestServerFileDiscoverRerankingNormalization:
    """Test reranking normalization in server.py file_discover tool"""

    @pytest.mark.asyncio
    async def test_file_discover_reranking_disabled_when_credentials_missing(self):
        """Test file_discover disables reranking when credentials are missing"""
        import kbbridge.server as server_module

        file_discover = server_module.file_discover.fn

        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        # Credentials without reranking
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url=None,
            rerank_model=None,
        )

        with patch("kbbridge.server.get_current_credentials", return_value=creds):
            with patch("kbbridge.server.file_discover_service") as mock_service:
                mock_service.return_value = {"success": True, "files": []}

                await file_discover(
                    query="test",
                    dataset_id="test-dataset",
                    ctx=mock_ctx,
                    do_file_rerank=True,  # User requests reranking
                )

                # Verify reranking was disabled
                mock_ctx.info.assert_any_call(
                    "File reranking disabled: RERANK_URL or RERANK_MODEL not configured"
                )
                # Verify service was called with do_file_rerank=False
                mock_service.assert_called_once()
                call_kwargs = mock_service.call_args.kwargs
                assert call_kwargs["do_file_rerank"] is False

    @pytest.mark.asyncio
    async def test_file_discover_reranking_enabled_when_credentials_available(self):
        """Test file_discover enables reranking when credentials are available"""
        import kbbridge.server as server_module

        file_discover = server_module.file_discover.fn

        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        # Credentials with reranking
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        with patch("kbbridge.server.get_current_credentials", return_value=creds):
            with patch("kbbridge.server.file_discover_service") as mock_service:
                mock_service.return_value = {"success": True, "files": []}

                await file_discover(
                    query="test",
                    dataset_id="test-dataset",
                    ctx=mock_ctx,
                    do_file_rerank=True,
                )

                # Verify service was called with do_file_rerank=True
                mock_service.assert_called_once()
                call_kwargs = mock_service.call_args.kwargs
                assert call_kwargs["do_file_rerank"] is True

    @pytest.mark.asyncio
    async def test_file_discover_reranking_already_disabled(self):
        """Test file_discover doesn't change reranking when already disabled"""
        import kbbridge.server as server_module

        file_discover = server_module.file_discover.fn

        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        # Credentials without reranking
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url=None,
            rerank_model=None,
        )

        with patch("kbbridge.server.get_current_credentials", return_value=creds):
            with patch("kbbridge.server.file_discover_service") as mock_service:
                mock_service.return_value = {"success": True, "files": []}

                await file_discover(
                    query="test",
                    dataset_id="test-dataset",
                    ctx=mock_ctx,
                    do_file_rerank=False,  # User already disabled reranking
                )

                # Verify no info message about disabling reranking
                info_calls = [str(call) for call in mock_ctx.info.call_args_list]
                assert not any(
                    "File reranking disabled" in str(call) for call in info_calls
                )
                # Verify service was called with do_file_rerank=False
                mock_service.assert_called_once()
                call_kwargs = mock_service.call_args.kwargs
                assert call_kwargs["do_file_rerank"] is False


class TestServerRetrieverRerankingNormalization:
    """Test reranking normalization in server.py retriever tool"""

    @pytest.mark.asyncio
    async def test_retriever_reranking_disabled_when_credentials_missing(self):
        """Test retriever disables reranking when credentials are missing"""
        import kbbridge.server as server_module

        retriever = server_module.retriever.fn

        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        # Credentials without reranking
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url=None,
            rerank_model=None,
        )

        with patch("kbbridge.server.get_current_credentials", return_value=creds):
            with patch("kbbridge.server.retriever_service") as mock_service:
                mock_service.return_value = {"result": []}

                await retriever(
                    dataset_id="test-dataset",
                    query="test query",
                    ctx=mock_ctx,
                    does_rerank=True,  # User requests reranking
                )

                # Verify reranking was disabled
                mock_ctx.info.assert_any_call(
                    "Reranking disabled: RERANK_URL or RERANK_MODEL not configured"
                )
                # Verify service was called with does_rerank=False
                mock_service.assert_called_once()
                call_kwargs = mock_service.call_args.kwargs
                assert call_kwargs["does_rerank"] is False

    @pytest.mark.asyncio
    async def test_retriever_reranking_enabled_when_credentials_available(self):
        """Test retriever enables reranking when credentials are available"""
        import kbbridge.server as server_module

        retriever = server_module.retriever.fn

        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        # Credentials with reranking
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        with patch("kbbridge.server.get_current_credentials", return_value=creds):
            with patch("kbbridge.server.retriever_service") as mock_service:
                mock_service.return_value = {"result": []}

                await retriever(
                    dataset_id="test-dataset",
                    query="test query",
                    ctx=mock_ctx,
                    does_rerank=True,
                )

                # Verify service was called with does_rerank=True
                mock_service.assert_called_once()
                call_kwargs = mock_service.call_args.kwargs
                assert call_kwargs["does_rerank"] is True

    @pytest.mark.asyncio
    async def test_retriever_reranking_already_disabled(self):
        """Test retriever doesn't change reranking when already disabled"""
        import kbbridge.server as server_module

        retriever = server_module.retriever.fn

        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        # Credentials without reranking
        creds = ConfigCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url=None,
            rerank_model=None,
        )

        with patch("kbbridge.server.get_current_credentials", return_value=creds):
            with patch("kbbridge.server.retriever_service") as mock_service:
                mock_service.return_value = {"result": []}

                await retriever(
                    dataset_id="test-dataset",
                    query="test query",
                    ctx=mock_ctx,
                    does_rerank=False,  # User already disabled reranking
                )

                # Verify no info message about disabling reranking
                info_calls = [str(call) for call in mock_ctx.info.call_args_list]
                assert not any("Reranking disabled" in str(call) for call in info_calls)
                # Verify service was called with does_rerank=False
                mock_service.assert_called_once()
                call_kwargs = mock_service.call_args.kwargs
                assert call_kwargs["does_rerank"] is False


class TestDirectApproachProcessorRerankingNormalization:
    """Test reranking normalization in DirectApproachProcessor"""

    def test_direct_processor_reranking_disabled_when_credentials_missing(self):
        """Test DirectApproachProcessor disables reranking when credentials are missing"""
        mock_retriever = Mock()
        mock_answer_extractor = Mock()

        # Credentials without reranking
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url=None,
            rerank_model=None,
        )

        processor = DirectApproachProcessor(
            mock_retriever, mock_answer_extractor, credentials=creds
        )

        # Mock retriever.retrieve to return success
        mock_retriever.retrieve.return_value = {
            "success": True,
            "result": [{"content": "test", "document_name": "test.pdf"}],
        }
        mock_answer_extractor.extract.return_value = {
            "success": True,
            "answer": "Test answer",
        }

        processor.process("test query", "test-dataset", None, 10)

        # Verify retriever was called with does_rerank=False
        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["does_rerank"] is False

    def test_direct_processor_reranking_enabled_when_credentials_available(self):
        """Test DirectApproachProcessor enables reranking when credentials are available"""
        mock_retriever = Mock()
        mock_answer_extractor = Mock()

        # Credentials with reranking
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        processor = DirectApproachProcessor(
            mock_retriever, mock_answer_extractor, credentials=creds
        )

        # Mock retriever.retrieve to return success
        mock_retriever.retrieve.return_value = {
            "success": True,
            "result": [{"content": "test", "document_name": "test.pdf"}],
        }
        mock_answer_extractor.extract.return_value = {
            "success": True,
            "answer": "Test answer",
        }

        processor.process("test query", "test-dataset", None, 10)

        # Verify retriever was called with does_rerank=True
        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["does_rerank"] is True

    def test_direct_processor_reranking_disabled_when_no_credentials(self):
        """Test DirectApproachProcessor disables reranking when credentials are None"""
        mock_retriever = Mock()
        mock_answer_extractor = Mock()

        processor = DirectApproachProcessor(
            mock_retriever, mock_answer_extractor, credentials=None
        )

        # Mock retriever.retrieve to return success
        mock_retriever.retrieve.return_value = {
            "success": True,
            "result": [{"content": "test", "document_name": "test.pdf"}],
        }
        mock_answer_extractor.extract.return_value = {
            "success": True,
            "answer": "Test answer",
        }

        processor.process("test query", "test-dataset", None, 10)

        # Verify retriever was called with does_rerank=False
        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["does_rerank"] is False

    def test_direct_processor_uses_call_method_when_retrieve_not_available(self):
        """Test DirectApproachProcessor uses call() when retriever doesn't have retrieve()"""
        mock_retriever = Mock()
        del mock_retriever.retrieve  # Remove retrieve method
        mock_retriever.call = Mock(return_value={"success": True, "result": []})
        mock_answer_extractor = Mock()

        # Credentials without reranking
        creds = ModelCredentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            rerank_url=None,
            rerank_model=None,
        )

        processor = DirectApproachProcessor(
            mock_retriever, mock_answer_extractor, credentials=creds
        )

        mock_answer_extractor.extract.return_value = {
            "success": True,
            "answer": "Test answer",
        }

        processor.process("test query", "test-dataset", None, 10)

        # Verify call() was used with does_rerank=False
        mock_retriever.call.assert_called_once()
        call_kwargs = mock_retriever.call.call_args.kwargs
        assert call_kwargs["does_rerank"] is False
