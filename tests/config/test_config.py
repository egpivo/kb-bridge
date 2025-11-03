
"""
Test Suite for Configuration Modules
Tests all configuration modules with comprehensive coverage
"""

from kbbridge.config.config import Config


class TestConfig:
    """Test Config functionality"""

    def test_init(self):
        """Test config initialization"""
        config = Config()
        assert config is not None

    def test_get_credentials_from_headers_success(self):
        """Test successful credential extraction from headers"""
        headers = {
            "X-RETRIEVAL-ENDPOINT": "https://dify.com",
            "X-RETRIEVAL-API-KEY": "test-api-key",
            "X-LLM-API-URL": "https://api.openai.com/v1",
            "X-LLM-MODEL": "gpt-4",
            "X-RERANK-URL": "https://rerank.example.com",
            "X-RERANK-MODEL": "test-model",
        }

        credentials = Config.get_credentials_from_headers(headers)

        assert credentials is not None
        assert credentials.retrieval_endpoint == "https://dify.com"
        assert credentials.retrieval_api_key == "test-api-key"
        assert credentials.llm_api_url == "https://api.openai.com/v1"
        assert credentials.llm_model == "gpt-4"
        assert credentials.rerank_url == "https://rerank.example.com"
        assert credentials.rerank_model == "test-model"

    def test_get_credentials_from_headers_missing_fields(self):
        """Test credential extraction with missing fields"""
        headers = {
            "X-RETRIEVAL-ENDPOINT": "https://dify.com",
            "X-RETRIEVAL-API-KEY": "test-api-key"
            # Missing other required fields
        }

        credentials = Config.get_credentials_from_headers(headers)

        # Should still create credentials with available fields
        assert credentials is not None
        assert credentials.retrieval_endpoint == "https://dify.com"
        assert credentials.retrieval_api_key == "test-api-key"

    def test_get_credentials_from_headers_empty(self):
        """Test credential extraction with empty headers"""
        headers = {}

        credentials = Config.get_credentials_from_headers(headers)

        # Should return None for empty headers
        assert credentials is None

    def test_get_credentials_from_headers_none(self):
        """Test credential extraction with None headers"""
        credentials = Config.get_credentials_from_headers(None)

        assert credentials is None

    def test_get_credentials_from_headers_invalid_format(self):
        """Test credential extraction with invalid header format"""
        headers = {"INVALID-HEADER": "value"}

        credentials = Config.get_credentials_from_headers(headers)

        # Should return None for invalid format
        assert credentials is None

    def test_validate_credentials_success(self):
        """Test successful credential validation"""
        headers = {
            "X-RETRIEVAL-ENDPOINT": "https://dify.com",
            "X-RETRIEVAL-API-KEY": "test-api-key",
            "X-LLM-API-URL": "https://api.openai.com/v1",
            "X-LLM-MODEL": "gpt-4",
        }

        credentials = Config.get_credentials_from_headers(headers)
        is_valid = Config.validate_credentials(credentials)

        assert is_valid is True

    def test_validate_credentials_missing_required(self):
        """Test credential validation with missing required fields"""
        headers = {
            "X-RETRIEVAL-ENDPOINT": "https://dify.com"
            # Missing required API key
        }

        credentials = Config.get_credentials_from_headers(headers)
        is_valid = Config.validate_credentials(credentials)

        assert is_valid is False

    def test_validate_credentials_none(self):
        """Test credential validation with None credentials"""
        is_valid = Config.validate_credentials(None)

        assert is_valid is False

    def test_get_default_config(self):
        """Test getting default configuration"""
        config = Config.get_default_config()

        assert config is not None
        assert isinstance(config, dict)

    def test_set_config(self):
        """Test setting configuration"""
        config = Config()
        new_config = {"test_key": "test_value"}

        config.set_config(new_config)

        # Should update internal configuration
        assert config.config == new_config

    def test_get_config_value(self):
        """Test getting configuration value"""
        config = Config()
        config.set_config({"test_key": "test_value"})

        value = config.get_config_value("test_key")

        assert value == "test_value"

    def test_get_config_value_default(self):
        """Test getting configuration value with default"""
        config = Config()

        value = config.get_config_value("nonexistent_key", "default_value")

        assert value == "default_value"

    def test_get_config_value_none(self):
        """Test getting configuration value that doesn't exist"""
        config = Config()

        value = config.get_config_value("nonexistent_key")

        assert value is None

    def test_update_config(self):
        """Test updating configuration"""
        config = Config()
        config.set_config({"key1": "value1", "key2": "value2"})

        config.update_config({"key2": "new_value2", "key3": "value3"})

        assert config.get_config_value("key1") == "value1"
        assert config.get_config_value("key2") == "new_value2"
        assert config.get_config_value("key3") == "value3"

    def test_reset_config(self):
        """Test resetting configuration"""
        config = Config()
        config.set_config({"test_key": "test_value"})

        config.reset_config()

        # Should reset to default configuration
        assert config.config == Config.get_default_config()

    def test_export_config(self):
        """Test exporting configuration"""
        config = Config()
        config.set_config({"key1": "value1", "key2": "value2"})

        exported = config.export_config()

        assert isinstance(exported, dict)
        assert exported["key1"] == "value1"
        assert exported["key2"] == "value2"

    def test_import_config(self):
        """Test importing configuration"""
        config = Config()
        imported_config = {"imported_key": "imported_value"}

        config.import_config(imported_config)

        assert config.get_config_value("imported_key") == "imported_value"

    def test_import_config_invalid(self):
        """Test importing invalid configuration"""
        config = Config()

        # Should handle invalid configuration gracefully
        config.import_config("invalid_config")

        # Should not crash and maintain current state
        assert config is not None

    def test_get_default_credentials_success(self):
        """Test getting default credentials from environment variables"""
        import os

        # Set environment variables
        os.environ["RETRIEVAL_ENDPOINT"] = "https://test.com"
        os.environ["RETRIEVAL_API_KEY"] = "test-key"
        os.environ["LLM_API_URL"] = "https://llm.com"
        os.environ["LLM_MODEL"] = "gpt-4"
        os.environ["LLM_API_TOKEN"] = "test-token"
        os.environ["RERANK_URL"] = "https://rerank.com"
        os.environ["RERANK_MODEL"] = "test-model"

        try:
            credentials = Config.get_default_credentials()
            assert credentials is not None
            assert credentials.retrieval_endpoint == "https://test.com"
            assert credentials.retrieval_api_key == "test-key"
            assert credentials.llm_api_url == "https://llm.com"
            assert credentials.llm_model == "gpt-4"
            assert credentials.llm_api_token == "test-token"
            assert credentials.rerank_url == "https://rerank.com"
            assert credentials.rerank_model == "test-model"
        finally:
            # Clean up environment variables
            for key in [
                "DIFY_ENDPOINT",
                "DIFY_API_KEY",
                "LLM_API_URL",
                "LLM_MODEL",
                "LLM_API_TOKEN",
                "RERANK_URL",
                "RERANK_MODEL",
            ]:
                os.environ.pop(key, None)

    def test_get_default_credentials_missing_required(self):
        """Test getting default credentials with missing required variables"""
        import os

        # Clear all environment variables first
        for key in ["DIFY_ENDPOINT", "DIFY_API_KEY", "LLM_API_URL", "LLM_MODEL"]:
            os.environ.pop(key, None)

        try:
            credentials = Config.get_default_credentials()
            assert credentials is None
        finally:
            # Clean up environment variables
            for key in ["DIFY_ENDPOINT", "DIFY_API_KEY", "LLM_API_URL", "LLM_MODEL"]:
                os.environ.pop(key, None)

    def test_get_default_credentials_none_values(self):
        """Test getting default credentials with None values"""
        import os

        # Set environment variables to None
        os.environ["DIFY_ENDPOINT"] = ""
        os.environ["DIFY_API_KEY"] = ""
        os.environ["LLM_API_URL"] = ""
        os.environ["LLM_MODEL"] = ""

        try:
            credentials = Config.get_default_credentials()
            assert credentials is None
        finally:
            # Clean up environment variables
            for key in ["DIFY_ENDPOINT", "DIFY_API_KEY", "LLM_API_URL", "LLM_MODEL"]:
                os.environ.pop(key, None)

    def test_get_search_config(self):
        """Test getting search configuration"""
        config = Config.get_search_config()

        assert isinstance(config, dict)
        # Should return a dictionary with search configuration

    def test_import_config_exception(self):
        """Test import_config with exception handling"""
        config = Config()

        # Test with None input (should trigger exception)
        result = config.import_config(None)
        assert result is False
