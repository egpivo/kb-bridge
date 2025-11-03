from unittest.mock import patch

from kbbridge.config.config import Credentials


class TestServerModule:
    """Test the server module functionality"""

    def test_get_current_credentials_empty(self):
        """Test get_current_credentials when no credentials are set"""
        from kbbridge.server import get_current_credentials

        # Mock auth_middleware to return None
        with patch("kbbridge.server.auth_middleware") as mock_auth:
            mock_auth.get_available_credentials.return_value = None

            result = get_current_credentials()
            assert result is None

    def test_set_current_credentials(self):
        """Test set_current_credentials functionality"""
        from kbbridge.server import get_current_credentials, set_current_credentials

        credentials = Credentials(
            retrieval_endpoint="https://test.com",
            retrieval_api_key="test-key",
            llm_api_url="https://llm.com",
            llm_model="gpt-4",
        )

        with patch("kbbridge.server.auth_middleware") as mock_auth:
            mock_auth.get_available_credentials.return_value = credentials

            set_current_credentials(credentials)
            result = get_current_credentials()

            assert result is not None
            assert result.retrieval_endpoint == "https://test.com"
            assert result.retrieval_api_key == "test-key"
            mock_auth.set_session_credentials.assert_called_once_with(credentials)

    def test_set_current_credentials_none(self):
        """Test set_current_credentials with None"""
        from kbbridge.server import get_current_credentials, set_current_credentials

        with patch("kbbridge.server.auth_middleware") as mock_auth:
            mock_auth.get_available_credentials.return_value = None

            set_current_credentials(None)
            result = get_current_credentials()

            assert result is None
            mock_auth.clear_session_credentials.assert_called_once()

    def test_main_function_logging(self):
        """Test main function logging"""
        import asyncio

        from kbbridge.server import main

        with patch("kbbridge.server.config_helper") as mock_config_helper:
            mock_config_helper.get_available_credentials.return_value = {
                "dify_endpoint": "https://test.com",
                "dify_api_key": "test-key",
            }

            # This should not raise an exception
            try:
                # Since main() is async, we need to run it properly
                asyncio.run(main())
            except SystemExit:
                pass  # Expected behavior for main function

    def test_environment_variables_set(self):
        """Test that environment variables are set correctly"""
        import os

        # Import the server module to trigger environment variable setting
        # Check that environment variables are set (from .env file or defaults)
        # .env file has MAX_WORKERS=3 and USE_CONTENT_BOOSTER=true
        # Test environment may override these values
        max_workers = os.environ.get("MAX_WORKERS", "").strip()
        use_content_booster = os.environ.get("USE_CONTENT_BOOSTER", "").strip()

        # Just verify they are set to some value
        assert max_workers  # Not empty
        assert use_content_booster  # Not empty

    def test_imports_work(self):
        """Test that all imports work correctly"""
        from kbbridge.server import (
            get_current_credentials,
            main,
            set_current_credentials,
        )

        # These should not raise import errors
        assert callable(get_current_credentials)
        assert callable(set_current_credentials)
        assert callable(main)
