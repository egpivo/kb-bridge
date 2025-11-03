import json
from unittest.mock import patch

from kbbridge.config.config import Credentials
from kbbridge.middleware import (
    AuthMiddleware,
    ErrorMiddleware,
    error_middleware,
    optional_auth,
    require_auth,
)
from kbbridge.middleware._auth_core import auth_middleware


class TestAuthMiddleware:
    """Test authentication middleware functionality"""

    def test_init(self):
        """Test middleware initialization"""
        middleware = AuthMiddleware()
        assert middleware._session_credentials is None

    def test_get_credentials_from_request(self):
        """Test credential extraction from request"""
        middleware = AuthMiddleware()

        # Test with mock headers - patch where it's used, not where it's defined
        with patch("kbbridge.middleware._auth_core.get_http_headers") as mock_headers:
            mock_headers.return_value = {
                "x-retrieval-endpoint": "https://test.com",
                "x-retrieval-api-key": "test-key",
                "x-llm-api-url": "https://llm.com",
                "x-llm-model": "gpt-4",
            }

            credentials = middleware.get_credentials_from_request()
            assert credentials is not None
            assert credentials.retrieval_endpoint == "https://test.com"
            assert credentials.retrieval_api_key == "test-key"

    def test_session_credentials(self):
        """Test session credential management"""
        middleware = AuthMiddleware()
        credentials = Credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )

        # Test setting session credentials
        middleware.set_session_credentials(credentials)
        retrieved = middleware.get_session_credentials()
        assert retrieved == credentials

        # Test clearing session credentials
        middleware.clear_session_credentials()
        assert middleware.get_session_credentials() is None

    def test_validate_credentials(self):
        """Test credential validation"""
        middleware = AuthMiddleware()

        # Test valid credentials
        valid_credentials = Credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )
        validation = middleware.validate_credentials(valid_credentials)
        assert validation["valid"] is True
        assert validation["errors"] == []

        # Test invalid credentials
        invalid_credentials = Credentials(retrieval_endpoint="", retrieval_api_key="test-key")
        validation = middleware.validate_credentials(invalid_credentials)
        assert validation["valid"] is False
        assert "Missing required retrieval backend credentials" in validation["errors"]

    def test_create_auth_error_response(self):
        """Test authentication error response creation"""
        middleware = AuthMiddleware()

        response = middleware.create_auth_error_response(
            "Test error", ["Error 1", "Error 2"]
        )
        data = json.loads(response)

        assert data["error"] == "Authentication failed"
        assert data["status"] == "error"
        assert data["message"] == "Test error"
        assert data["errors"] == ["Error 1", "Error 2"]
        assert "required_headers" in data


class TestErrorMiddleware:
    """Test error middleware functionality"""

    def test_handle_error(self):
        """Test general error handling"""
        middleware = ErrorMiddleware()

        response = middleware.handle_error(Exception("Test error"), "test_tool")
        data = json.loads(response)

        assert data["error"] == "Tool execution failed"
        assert data["status"] == "error"
        assert data["tool"] == "test_tool"
        assert data["message"] == "Test error"
        assert data["type"] == "Exception"

    def test_handle_auth_error(self):
        """Test authentication error handling"""
        middleware = ErrorMiddleware()

        response = middleware.handle_auth_error("Auth failed", ["Invalid key"])
        data = json.loads(response)

        assert data["error"] == "Authentication failed"
        assert data["status"] == "error"
        assert data["message"] == "Auth failed"
        assert data["errors"] == ["Invalid key"]

    def test_handle_validation_error(self):
        """Test validation error handling"""
        middleware = ErrorMiddleware()

        response = middleware.handle_validation_error("Invalid input", "query")
        data = json.loads(response)

        assert data["error"] == "Validation failed"
        assert data["status"] == "error"
        assert data["message"] == "Invalid input"
        assert data["field"] == "query"


class TestToolDecorators:
    """Test tool decorators functionality"""

    def test_require_auth_decorator_without_credentials(self):
        """Test require_auth decorator without credentials"""

        @require_auth
        def test_tool(query: str):
            return f"Processed: {query}"

        with patch(
            "kbbridge.middleware._auth_core.auth_middleware.get_available_credentials",
            return_value=None,
        ):
            result = test_tool("test query")
            data = json.loads(result)
            assert data["error"] == "Authentication failed"

    def test_require_auth_decorator_with_invalid_credentials(self):
        """Test require_auth decorator with invalid credentials"""

        @require_auth
        def test_tool(query: str):
            return f"Processed: {query}"

        invalid_credentials = Credentials(retrieval_endpoint="", retrieval_api_key="test-key")

        with patch(
            "kbbridge.middleware._auth_core.auth_middleware.get_available_credentials",
            return_value=invalid_credentials,
        ):
            with patch(
                "kbbridge.middleware._auth_core.auth_middleware.validate_credentials",
                return_value={"valid": False, "errors": ["Invalid endpoint"]},
            ):
                result = test_tool("test query")
                data = json.loads(result)
                assert data["error"] == "Authentication failed"

    def test_require_auth_decorator_with_valid_credentials(self):
        """Test require_auth decorator with valid credentials"""

        @require_auth
        def test_tool(query: str):
            from kbbridge.server import get_current_credentials

            credentials = get_current_credentials()
            endpoint = credentials.retrieval_endpoint if credentials else None
            return f"Processed: {query} with endpoint: {endpoint}"

        valid_credentials = Credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )

        with patch(
            "kbbridge.middleware._auth_core.auth_middleware.get_available_credentials",
            return_value=valid_credentials,
        ):
            with patch(
                "kbbridge.middleware._auth_core.auth_middleware.validate_credentials",
                return_value={"valid": True, "errors": []},
            ):
                result = test_tool("test query")
                assert "Processed: test query with endpoint: https://test.com" in result

    def test_optional_auth_decorator_without_credentials(self):
        """Test optional_auth decorator without credentials"""

        @optional_auth
        def test_tool(query: str, dify_endpoint: str = None):
            return f"Processed: {query} with endpoint: {dify_endpoint or 'no-auth'}"

        with patch(
            "kbbridge.middleware._auth_core.auth_middleware.get_available_credentials",
            return_value=None,
        ):
            result = test_tool("test query")
            assert "Processed: test query with endpoint: no-auth" in result

    def test_optional_auth_decorator_with_credentials(self):
        """Test optional_auth decorator with credentials"""

        @optional_auth
        def test_tool(query: str):
            from kbbridge.server import get_current_credentials

            credentials = get_current_credentials()
            endpoint = credentials.retrieval_endpoint if credentials else "no-auth"
            return f"Processed: {query} with endpoint: {endpoint}"

        valid_credentials = Credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )

        with patch(
            "kbbridge.middleware._auth_core.auth_middleware.get_available_credentials",
            return_value=valid_credentials,
        ):
            result = test_tool("test query")
            assert "Processed: test query with endpoint: https://test.com" in result

    def test_decorator_error_handling(self):
        """Test decorator error handling"""

        @require_auth
        def test_tool(
            query: str,
            dify_endpoint: str = None,
            dify_api_key: str = None,
            llm_api_url: str = None,
            llm_model: str = None,
            llm_api_token: str = None,
            rerank_url: str = None,
            rerank_model: str = None,
        ):
            raise ValueError("Test error")

        valid_credentials = Credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )

        with patch(
            "kbbridge.middleware._auth_core.auth_middleware.get_available_credentials",
            return_value=valid_credentials,
        ):
            with patch(
                "kbbridge.middleware._auth_core.auth_middleware.validate_credentials",
                return_value={"valid": True, "errors": []},
            ):
                result = test_tool("test query")
                data = json.loads(result)
                assert data["error"] == "Tool execution failed"
                assert "Test error" in data["message"]


class TestMiddlewareIntegration:
    """Test middleware integration"""

    def test_global_middleware_instances(self):
        """Test that global middleware instances are properly initialized"""
        assert isinstance(auth_middleware, AuthMiddleware)
        assert isinstance(error_middleware, ErrorMiddleware)

    def test_decorator_functions(self):
        """Test that decorator functions are callable"""
        assert callable(require_auth)
        assert callable(optional_auth)
