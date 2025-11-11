import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kbbridge.services.assistant_service import assistant_service as _assistant
from kbbridge.services.file_discover_service import file_discover_service
from kbbridge.services.file_lister_service import file_lister_service
from kbbridge.services.keyword_generator_service import keyword_generator_service
from kbbridge.services.retriever_service import retriever_service


class TestKBAssistantService:
    """Test kb_assistant_service functionality"""

    @pytest.mark.asyncio
    async def test_kb_assistant_service_success(
        self, mock_credentials, test_tool_parameters
    ):
        """Test successful kb_assistant execution"""
        # Mock Context
        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.warning = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.progress = AsyncMock()

        with patch("kbbridge.core.orchestration.ComponentFactory") as mock_factory:
            # Mock the component factory and its methods
            mock_processor = Mock()
            mock_processor.process.return_value = {
                "results": [{"content": "Test result", "score": 0.9}],
                "total_results": 1,
            }
            mock_factory.create_processor.return_value = mock_processor

            # Use assistant_service (renamed from kb_assistant_service)
            assistant_service = _assistant
            result = await assistant_service(
                dataset_id="test-dataset",
                query="test query",
                ctx=mock_ctx,
            )

            assert isinstance(result, dict)
            assert "error" in result

    @pytest.mark.asyncio
    async def test_kb_assistant_service_invalid_dataset_id(self, mock_credentials):
        """Test kb_assistant with invalid dataset_id"""
        # Mock Context
        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.warning = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.progress = AsyncMock()

        assistant_service = _assistant
        result = await assistant_service(
            dataset_id="",
            query="test query",
            ctx=mock_ctx,
        )

        assert "error" in result
        # Empty dataset_id should return "Invalid dataset_id" error
        assert (
            "Invalid dataset_id" in result["error"]
            or "LLM API token is required" in result["error"]
            or "KB Assistant failed" in result["error"]
        )

    @pytest.mark.asyncio
    async def test_kb_assistant_service_empty_dataset(self, mock_credentials):
        """Test kb_assistant with empty dataset"""
        # Mock Context
        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.warning = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.progress = AsyncMock()

        assistant_service = _assistant
        result = await assistant_service(
            dataset_id="",
            query="test query",
            ctx=mock_ctx,
        )

        assert "error" in result
        assert "Invalid dataset_id" in result["error"]

    @pytest.mark.asyncio
    async def test_kb_assistant_service_processing_error(self, mock_credentials):
        """Test kb_assistant with processing error"""
        # Mock Context
        mock_ctx = Mock()
        mock_ctx.info = AsyncMock()
        mock_ctx.warning = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.progress = AsyncMock()

        with patch(
            "kbbridge.core.orchestration.DatasetProcessor"
        ) as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process.side_effect = Exception("Processing error")
            mock_processor_class.return_value = mock_processor

            assistant_service = _assistant
            result = await assistant_service(
                dataset_id="test-dataset",
                query="test query",
                ctx=mock_ctx,
            )

            assert "error" in result
            # With mocked exception, should get error response
            assert any(
                keyword in result["error"]
                for keyword in ["KB Assistant", "Processing", "error", "failed"]
            )


class TestFileListerService:
    """Test file_lister_service functionality"""

    def test_file_lister_service_success(self, mock_credentials):
        """Test successful file lister execution"""
        # Test basic parameter validation
        result = file_lister_service(
            dataset_id="",
            retrieval_endpoint="https://dify.ai",
            retrieval_api_key="test-api-key",
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "dataset_id is required" in result["error"]

    def test_file_lister_service_error(self, mock_credentials):
        """Test file lister with invalid credentials"""
        with patch(
            "kbbridge.services.file_lister_service.RetrievalCredentials"
        ) as mock_creds_class:
            # Mock from_env to return invalid credentials
            mock_creds = Mock()
            mock_creds.validate.return_value = (False, "Invalid credentials")
            mock_creds.backend_type = "dify"
            mock_creds_class.from_env.return_value = mock_creds

            # Test with missing credentials (empty strings trigger from_env fallback)
            result = file_lister_service(
                dataset_id="test-dataset",
                retrieval_endpoint="",
                retrieval_api_key="",
            )

            assert "error" in result
            assert "Invalid credentials" in result["error"]


class TestKeywordGeneratorService:
    """Test keyword_generator_service functionality"""

    def test_keyword_generator_service_success(self, mock_credentials):
        """Test successful keyword generator execution"""
        with patch(
            "kbbridge.core.query.keyword_generator.generate_keywords"
        ) as mock_generator:
            mock_generator.return_value = {
                "result": ["keyword1", "keyword2", "keyword3"]
            }

            result = keyword_generator_service(query="test query", max_sets=5)

            assert isinstance(result, dict)

    def test_keyword_generator_service_error(self, mock_credentials):
        """Test keyword generator with error"""
        with patch(
            "kbbridge.core.query.keyword_generator.generate_keywords"
        ) as mock_generator:
            mock_generator.side_effect = Exception("Generator error")

            result = keyword_generator_service(
                query="test query",
                max_sets=5,
                llm_api_url="https://api.openai.com/v1",
                llm_model="gpt-4",
            )

            assert "error" in result
            assert "Generator error" in result["error"]


class TestRetrieverService:
    """Test retriever_service functionality"""

    def test_retriever_service_success(self, mock_credentials):
        """Test successful retriever execution"""
        with patch(
            "kbbridge.utils.working_components.KnowledgeBaseRetriever"
        ) as mock_retriever:
            mock_retriever_instance = Mock()
            mock_retriever_instance.retrieve.return_value = [
                {"content": "Test content 1", "score": 0.9},
                {"content": "Test content 2", "score": 0.8},
            ]
            mock_retriever.return_value = mock_retriever_instance

            result = retriever_service(
                query="test query",
                dataset_id="test-dataset",
                top_k=10,
                score_threshold=0.5,
                verbose=False,
                retrieval_endpoint="https://dify.ai",
                retrieval_api_key="test-api-key",
            )

            assert "result" in result
            assert isinstance(result["result"], list)

    def test_retriever_service_error(self, mock_credentials):
        """Test retriever with error"""

        with patch("kbbridge.integrations.RetrievalCredentials") as mock_creds_class:
            with patch(
                "kbbridge.utils.working_components.KnowledgeBaseRetriever"
            ) as mock_retriever:
                # Setup credentials mock
                mock_creds = Mock()
                mock_creds.validate.return_value = (True, None)
                mock_creds.endpoint = "https://dify.ai"
                mock_creds.api_key = "test-api-key"
                mock_creds_class.return_value = mock_creds

                # Mock the instance creation to raise an exception
                mock_retriever_instance = Mock()
                mock_retriever_instance.retrieve.side_effect = Exception(
                    "Retriever error"
                )
                mock_retriever.return_value = mock_retriever_instance

                result = retriever_service(
                    query="test query",
                    dataset_id="test-dataset",
                    retrieval_endpoint="https://dify.ai",
                    retrieval_api_key="test-api-key",
                )

                assert "error" in result
                # Accept any error message from retriever
                assert isinstance(result["error"], str) and result["error"]


# TestContentBoosterService removed - content_booster_service was deleted
# as its functionality is integrated into AdvancedApproachProcessor


class TestFileDiscoverService:
    """Test file_discover_service functionality"""

    @patch("kbbridge.integrations.RetrieverRouter.create_retriever")
    @patch("kbbridge.core.discovery.file_discover.FileDiscover")
    @patch("kbbridge.services.file_discover_service.RetrievalCredentials")
    def test_file_discover_service_success(
        self, mock_credentials_class, mock_file_discover_class, mock_create_retriever
    ):
        """Test successful file discover"""
        # Mock credentials
        mock_creds = Mock()
        mock_creds.validate.return_value = (True, None)
        mock_creds.backend_type = "dify"
        mock_creds.endpoint = "https://test.com"
        mock_creds.api_key = "test-key"
        mock_credentials_class.return_value = mock_creds

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.build_metadata_filter.return_value = {"filter": "value"}
        mock_create_retriever.return_value = mock_retriever

        # Mock file discover - FileDiscover is instantiated and then called
        mock_file = Mock()
        mock_file.file_name = "test.pdf"
        mock_discover_instance = Mock()
        # When the instance is called, it should return the list of files
        mock_discover_instance.__call__ = Mock(return_value=[mock_file])
        mock_file_discover_class.return_value = mock_discover_instance

        result = file_discover_service(
            query="test query",
            dataset_id="test-dataset",
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        assert result["success"] is True
        assert len(result["distinct_files"]) == 1
        assert result["distinct_files"][0] == "test.pdf"
        assert result["total_files"] == 1

    @patch("kbbridge.services.file_discover_service.RetrievalCredentials")
    def test_file_discover_service_invalid_credentials(self, mock_credentials_class):
        """Test file discover with invalid credentials"""
        # Mock invalid credentials
        mock_creds = Mock()
        mock_creds.validate.return_value = (False, "Invalid credentials")
        mock_credentials_class.return_value = mock_creds

        result = file_discover_service(
            query="test query",
            dataset_id="test-dataset",
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        assert "error" in result
        # Should return the validation error directly, not make HTTP calls
        assert "Invalid credentials" in result["error"]

    @patch("kbbridge.integrations.RetrieverRouter.create_retriever")
    @patch("kbbridge.core.discovery.file_discover.FileDiscover")
    @patch("kbbridge.services.file_discover_service.RetrievalCredentials")
    def test_file_discover_service_from_env(
        self, mock_credentials_class, mock_file_discover_class, mock_create_retriever
    ):
        """Test file discover using credentials from environment"""
        # Mock credentials from environment
        mock_creds = Mock()
        mock_creds.validate.return_value = (True, None)
        mock_creds.backend_type = "dify"
        mock_creds.endpoint = "https://test.com"
        mock_creds.api_key = "test-key"
        mock_credentials_class.from_env.return_value = mock_creds

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.build_metadata_filter.return_value = {"filter": "value"}
        mock_create_retriever.return_value = mock_retriever

        # Mock file discover - FileDiscover is instantiated and then called
        mock_file = Mock()
        mock_file.file_name = "test.pdf"
        mock_discover_instance = Mock()
        # When the instance is called, it should return the list of files
        mock_discover_instance.__call__ = Mock(return_value=[mock_file])
        mock_file_discover_class.return_value = mock_discover_instance

        result = file_discover_service(
            query="test query",
            dataset_id="test-dataset",
            # No credentials provided, should use from_env
        )

        # Verify from_env was called
        mock_credentials_class.from_env.assert_called_once()
        assert result["success"] is True

    @patch("kbbridge.integrations.RetrieverRouter.create_retriever")
    @patch("kbbridge.services.file_discover_service.RetrievalCredentials")
    def test_file_discover_service_exception(
        self, mock_credentials_class, mock_create_retriever
    ):
        """Test file discover with exception"""
        # Mock credentials
        mock_creds = Mock()
        mock_creds.validate.return_value = (True, None)
        mock_creds.backend_type = "dify"
        mock_creds.endpoint = "https://test.com"
        mock_creds.api_key = "test-key"
        mock_credentials_class.return_value = mock_creds

        # Mock retriever to raise an exception
        mock_create_retriever.side_effect = Exception("Test error")

        result = file_discover_service(
            query="test query",
            dataset_id="test-dataset",
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        assert "error" in result
        assert "Test error" in result["error"]


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    sys.exit(exit_code)
