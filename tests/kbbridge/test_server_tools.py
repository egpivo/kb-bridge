"""
Test server MCP tools functionality
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kbbridge.config.config import Credentials


class TestServerAssistantTool:
    """Test assistant MCP tool"""

    @pytest.mark.asyncio
    async def test_assistant_without_credentials(self):
        """Test assistant tool without credentials"""
        import kbbridge.server as server_module

        # FastMCP wraps functions in FunctionTool, access via .fn attribute
        assistant_func = server_module.assistant.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        with patch("kbbridge.server.get_current_credentials", return_value=None):
            result = await assistant_func(
                resource_id="test",
                query="test query",
                ctx=mock_ctx,
            )

            assert "Error: No credentials available" in result
            mock_ctx.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_assistant_success(self):
        """Test assistant tool success"""
        import kbbridge.server as server_module

        assistant = server_module.assistant.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        mock_result = {"answer": "Test answer", "sources": []}

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.assistant_service", return_value=mock_result
            ) as mock_service:
                result = await assistant(
                    resource_id="test",
                    query="test query",
                    ctx=mock_ctx,
                )

                assert json.loads(result) == mock_result
                mock_service.assert_called_once()

    @pytest.mark.asyncio
    async def test_assistant_timeout(self):
        """Test assistant tool timeout"""
        import kbbridge.server as server_module

        assistant = server_module.assistant.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.assistant_service",
                side_effect=asyncio.TimeoutError(),
            ):
                result = await assistant(
                    resource_id="test",
                    query="test query",
                    ctx=mock_ctx,
                )

                data = json.loads(result)
                assert data["error"] == "Request timeout"
                assert data["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_assistant_exception(self):
        """Test assistant tool exception handling"""
        import kbbridge.server as server_module

        assistant = server_module.assistant.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.assistant_service",
                side_effect=ValueError("Test error"),
            ):
                result = await assistant(
                    resource_id="test",
                    query="test query",
                    ctx=mock_ctx,
                )

                data = json.loads(result)
                assert data["error"] == "Tool execution failed"
                assert "Test error" in data["message"]

    @pytest.mark.asyncio
    async def test_assistant_with_custom_instructions(self):
        """Test assistant tool with custom instructions"""
        import kbbridge.server as server_module

        assistant = server_module.assistant.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        mock_result = {"answer": "Test answer"}

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch("kbbridge.server.assistant_service", return_value=mock_result):
                result = await assistant(
                    resource_id="test",
                    query="test query",
                    ctx=mock_ctx,
                    custom_instructions="Use formal language",
                )

                assert json.loads(result) == mock_result
                # Verify custom instructions were logged
                assert any(
                    "custom instructions" in str(call).lower()
                    for call in mock_ctx.info.call_args_list
                )

    @pytest.mark.asyncio
    async def test_assistant_with_query_rewriting(self):
        """Test assistant tool with query rewriting enabled"""
        import kbbridge.server as server_module

        assistant = server_module.assistant.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        mock_result = {"answer": "Test answer"}

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch("kbbridge.server.assistant_service", return_value=mock_result):
                result = await assistant(
                    resource_id="test",
                    query="test query",
                    ctx=mock_ctx,
                    enable_query_rewriting=True,
                )

                assert json.loads(result) == mock_result
                # Verify query rewriting was logged
                assert any(
                    "query rewriting" in str(call).lower()
                    for call in mock_ctx.info.call_args_list
                )


class TestServerFileDiscoverTool:
    """Test file_discover MCP tool"""

    @pytest.mark.asyncio
    async def test_file_discover_without_credentials(self):
        """Test file_discover tool without credentials"""
        import kbbridge.server as server_module

        file_discover = server_module.file_discover.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        with patch("kbbridge.server.get_current_credentials", return_value=None):
            result = await file_discover(
                query="test query",
                resource_id="test-dataset",
                ctx=mock_ctx,
            )

            assert "Error: No credentials available" in result

    @pytest.mark.asyncio
    async def test_file_discover_success(self):
        """Test file_discover tool success"""
        import kbbridge.server as server_module

        file_discover = server_module.file_discover.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        mock_result = {"success": True, "distinct_files": ["file1.pdf"]}

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.file_discover_service", return_value=mock_result
            ):
                result = await file_discover(
                    query="test query",
                    resource_id="test-dataset",
                    ctx=mock_ctx,
                )

                assert json.loads(result) == mock_result

    @pytest.mark.asyncio
    async def test_file_discover_exception(self):
        """Test file_discover tool exception handling"""
        import kbbridge.server as server_module

        file_discover = server_module.file_discover.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.file_discover_service",
                side_effect=ValueError("Test error"),
            ):
                result = await file_discover(
                    query="test query",
                    resource_id="test-dataset",
                    ctx=mock_ctx,
                )

                data = json.loads(result)
                assert data["error"] == "Tool execution failed"
                assert "Test error" in data["message"]


class TestServerFileListerTool:
    """Test file_lister MCP tool"""

    @pytest.mark.asyncio
    async def test_file_lister_without_credentials(self):
        """Test file_lister tool without credentials"""
        import kbbridge.server as server_module

        file_lister = server_module.file_lister.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        with patch("kbbridge.server.get_current_credentials", return_value=None):
            result = await file_lister(
                resource_id="test-dataset",
                ctx=mock_ctx,
            )

            assert "Error: No credentials available" in result

    @pytest.mark.asyncio
    async def test_file_lister_success(self):
        """Test file_lister tool success"""
        import kbbridge.server as server_module

        file_lister = server_module.file_lister.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        mock_result = {"files": ["file1.pdf", "file2.pdf"]}

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch("kbbridge.server.file_lister_service", return_value=mock_result):
                result = await file_lister(
                    resource_id="test-dataset",
                    ctx=mock_ctx,
                    limit=10,
                    offset=0,
                )

                assert json.loads(result) == mock_result

    @pytest.mark.asyncio
    async def test_file_lister_exception(self):
        """Test file_lister tool exception handling"""
        import kbbridge.server as server_module

        file_lister = server_module.file_lister.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.file_lister_service",
                side_effect=ValueError("Test error"),
            ):
                result = await file_lister(
                    resource_id="test-dataset",
                    ctx=mock_ctx,
                )

                data = json.loads(result)
                assert data["error"] == "Tool execution failed"
                assert "Test error" in data["message"]


class TestServerKeywordGeneratorTool:
    """Test keyword_generator MCP tool"""

    @pytest.mark.asyncio
    async def test_keyword_generator_without_credentials(self):
        """Test keyword_generator tool without credentials"""
        import kbbridge.server as server_module

        keyword_generator = server_module.keyword_generator.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        with patch("kbbridge.server.get_current_credentials", return_value=None):
            result = await keyword_generator(
                query="test query",
                ctx=mock_ctx,
            )

            assert "Error: No credentials available" in result

    @pytest.mark.asyncio
    async def test_keyword_generator_success(self):
        """Test keyword_generator tool success"""
        import kbbridge.server as server_module

        keyword_generator = server_module.keyword_generator.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            llm_api_token="test-token",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        mock_result = {"keywords": [["keyword1", "keyword2"]]}

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.keyword_generator_service", return_value=mock_result
            ):
                result = await keyword_generator(
                    query="test query",
                    ctx=mock_ctx,
                    max_sets=5,
                )

                assert json.loads(result) == mock_result

    @pytest.mark.asyncio
    async def test_keyword_generator_exception(self):
        """Test keyword_generator tool exception handling"""
        import kbbridge.server as server_module

        keyword_generator = server_module.keyword_generator.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.keyword_generator_service",
                side_effect=ValueError("Test error"),
            ):
                result = await keyword_generator(
                    query="test query",
                    ctx=mock_ctx,
                )

                data = json.loads(result)
                assert data["error"] == "Tool execution failed"
                assert "Test error" in data["message"]


class TestServerRetrieverTool:
    """Test retriever MCP tool"""

    @pytest.mark.asyncio
    async def test_retriever_without_credentials(self):
        """Test retriever tool without credentials"""
        import kbbridge.server as server_module

        retriever = server_module.retriever.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        with patch("kbbridge.server.get_current_credentials", return_value=None):
            result = await retriever(
                resource_id="test-dataset",
                query="test query",
                ctx=mock_ctx,
            )

            assert "Error: No credentials available" in result

    @pytest.mark.asyncio
    async def test_retriever_success(self):
        """Test retriever tool success"""
        import kbbridge.server as server_module

        retriever = server_module.retriever.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
            llm_api_token="test-token",
            rerank_url="https://rerank.com",
            rerank_model="test-model",
        )

        mock_result = {"results": [{"content": "test content"}]}

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch("kbbridge.server.retriever_service", return_value=mock_result):
                result = await retriever(
                    resource_id="test-dataset",
                    query="test query",
                    ctx=mock_ctx,
                )

                assert json.loads(result) == mock_result

    @pytest.mark.asyncio
    async def test_retriever_exception(self):
        """Test retriever tool exception handling"""
        import kbbridge.server as server_module

        retriever = server_module.retriever.fn

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.error = AsyncMock()

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
        )

        with patch("kbbridge.server.get_current_credentials", return_value=credentials):
            with patch(
                "kbbridge.server.retriever_service",
                side_effect=ValueError("Test error"),
            ):
                result = await retriever(
                    resource_id="test-dataset",
                    query="test query",
                    ctx=mock_ctx,
                )

                data = json.loads(result)
                assert data["error"] == "Tool execution failed"
                assert "Test error" in data["message"]
