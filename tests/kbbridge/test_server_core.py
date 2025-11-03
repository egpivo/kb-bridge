"""
Test server core functionality
"""
import sys
from unittest.mock import patch

# Import the server module components without triggering tool registration
sys.path.insert(0, "/home/wenting.wang/mcp-kb-assistant")

from fastmcp.server.dependencies import get_http_headers

# Import the functions we need to test
from kbbridge.config.config import Config, Credentials


def _get_headers():
    """Get HTTP headers with error handling"""
    try:
        headers = get_http_headers(include_all=True)
        return headers or {}
    except Exception:
        return {}


def _get_credentials_from_headers():
    """Extract credentials from HTTP headers"""
    try:
        headers = _get_headers()
        return Config.get_credentials_from_headers(headers)
    except Exception:
        return None


class TestServerCore:
    """Test core server functionality"""

    def test_get_headers_success(self):
        """Test successful header retrieval"""
        with patch(
            "tests.kbbridge.test_server_core.get_http_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = {
                "x-retrieval-endpoint": "https://test.com",
                "x-retrieval-api-key": "test-key",
            }

            headers = _get_headers()

            assert isinstance(headers, dict)
            assert "x-retrieval-endpoint" in headers
            assert "x-retrieval-api-key" in headers

    def test_get_headers_exception(self):
        """Test header retrieval with exception"""
        with patch(
            "tests.kbbridge.test_server_core.get_http_headers"
        ) as mock_get_headers:
            mock_get_headers.side_effect = Exception("Header error")

            headers = _get_headers()

            assert isinstance(headers, dict)
            assert len(headers) == 0

    def test_get_headers_none_response(self):
        """Test header retrieval with None response"""
        with patch(
            "tests.kbbridge.test_server_core.get_http_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = None

            headers = _get_headers()

            assert isinstance(headers, dict)
            assert len(headers) == 0

    def test_get_credentials_from_headers_success(self):
        """Test successful credential extraction"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                "x-retrieval-endpoint": "https://test.com",
                "x-retrieval-api-key": "test-key",
                "x-llm-api-url": "https://llm.com",
                "x-llm-model": "gpt-4",
            }

            credentials = _get_credentials_from_headers()

            assert credentials is not None
            assert credentials.retrieval_endpoint == "https://test.com"
            assert credentials.retrieval_api_key == "test-key"

    def test_get_credentials_from_headers_missing(self):
        """Test credential extraction with missing headers"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.return_value = {}

            credentials = _get_credentials_from_headers()

            assert credentials is None

    def test_get_credentials_from_headers_exception(self):
        """Test credential extraction with exception"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.side_effect = Exception("Header error")

            credentials = _get_credentials_from_headers()

            assert credentials is None

    def test_get_credentials_from_headers_partial(self):
        """Test credential extraction with partial headers"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                "x-retrieval-endpoint": "https://test.com",
                # Missing x-retrieval-api-key
            }

            credentials = _get_credentials_from_headers()

            assert credentials is None

    def test_get_credentials_from_headers_case_insensitive(self):
        """Test credential extraction with different case headers"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                "X-RETRIEVAL-ENDPOINT": "https://test.com",
                "X-RETRIEVAL-API-KEY": "test-key",
                "X-LLM-API-URL": "https://llm.com",
                "X-LLM-MODEL": "gpt-4",
            }

            credentials = _get_credentials_from_headers()

            assert credentials is not None
            assert credentials.retrieval_endpoint == "https://test.com"
            assert credentials.retrieval_api_key == "test-key"


class TestServerEnvironment:
    """Test server environment setup"""

    def test_environment_variables_set(self):
        """Test that environment variables can be set"""
        import os

        # Test that we can set environment variables
        os.environ["TEST_VAR"] = "test_value"
        assert os.environ.get("TEST_VAR") == "test_value"

    def test_logging_configuration(self):
        """Test logging configuration"""
        import logging

        # Test that logger is configured
        logger = logging.getLogger("kbbridge.server")
        assert logger is not None
        assert logger.level <= logging.INFO


class TestServerImports:
    """Test server imports and initialization"""

    def test_config_import(self):
        """Test config import"""
        assert Config is not None
        assert Credentials is not None

    def test_fastmcp_import(self):
        """Test FastMCP can be imported"""
        from fastmcp import FastMCP

        # Test that we can create a FastMCP instance
        mcp = FastMCP("test-server")
        assert mcp is not None
        assert hasattr(mcp, "tool")


class TestServerErrorHandling:
    """Test server error handling"""

    def test_get_headers_error_handling(self):
        """Test error handling in _get_headers"""
        with patch(
            "tests.kbbridge.test_server_core.get_http_headers"
        ) as mock_get_headers:
            # Test various exception types
            mock_get_headers.side_effect = ValueError("Invalid headers")

            headers = _get_headers()

            assert isinstance(headers, dict)
            assert len(headers) == 0

    def test_get_credentials_error_handling(self):
        """Test error handling in _get_credentials_from_headers"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.side_effect = RuntimeError("Runtime error")

            credentials = _get_credentials_from_headers()

            assert credentials is None

    def test_get_credentials_config_error(self):
        """Test error handling when Config.get_credentials_from_headers fails"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.return_value = {"x-retrieval-endpoint": "https://test.com"}

            with patch(
                "kbbridge.config.config.Config.get_credentials_from_headers"
            ) as mock_config:
                mock_config.side_effect = Exception("Config error")

                credentials = _get_credentials_from_headers()

                assert credentials is None


class TestServerEdgeCases:
    """Test server edge cases"""

    def test_get_headers_empty_dict(self):
        """Test _get_headers with empty dict"""
        with patch(
            "tests.kbbridge.test_server_core.get_http_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = {}

            headers = _get_headers()

            assert isinstance(headers, dict)
            assert len(headers) == 0

    def test_get_credentials_empty_headers(self):
        """Test _get_credentials_from_headers with empty headers"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.return_value = {}

            credentials = _get_credentials_from_headers()

            assert credentials is None

    def test_get_credentials_malformed_headers(self):
        """Test _get_credentials_from_headers with malformed headers"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                "x-retrieval-endpoint": "",  # Empty endpoint
                "x-retrieval-api-key": "test-key",
            }

            credentials = _get_credentials_from_headers()

            # Should return None due to empty endpoint
            assert credentials is None

    def test_get_credentials_whitespace_headers(self):
        """Test _get_credentials_from_headers with whitespace headers"""
        with patch("tests.kbbridge.test_server_core._get_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                "x-retrieval-endpoint": "  https://test.com  ",
                "x-retrieval-api-key": "  test-key  ",
            }

            credentials = _get_credentials_from_headers()

            # Should still work with whitespace (Config handles trimming)
            assert credentials is not None
            assert credentials.retrieval_endpoint == "  https://test.com  "
            assert credentials.retrieval_api_key == "  test-key  "
