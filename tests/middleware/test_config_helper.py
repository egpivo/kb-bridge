
"""
Test config_helper module functionality
"""

import json
import os
import tempfile

from kbbridge.middleware.credential_manager import MCPConfigHelper


class TestMCPConfigHelper:
    """Test MCPConfigHelper class"""

    def test_init_default_config_file(self):
        """Test initialization with default config file"""
        helper = MCPConfigHelper()

        assert helper.config_file.endswith(".mcp_kb_config.json")
        assert isinstance(helper.config, dict)

    def test_init_custom_config_file(self):
        """Test initialization with custom config file"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            config_file = tmp.name

        try:
            helper = MCPConfigHelper(config_file)
            assert helper.config_file == config_file
        finally:
            os.unlink(config_file)

    def test_load_config_file_exists(self):
        """Test loading config from existing file"""
        config_data = {"dify_endpoint": "https://test.com", "dify_api_key": "test-key"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(config_data, tmp)
            config_file = tmp.name

        try:
            helper = MCPConfigHelper(config_file)
            assert helper.config == config_data
        finally:
            os.unlink(config_file)

    def test_load_config_file_not_exists(self):
        """Test loading config when file doesn't exist"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            config_file = tmp.name  # File will be deleted

        helper = MCPConfigHelper(config_file)
        assert isinstance(helper.config, dict)
        # When file doesn't exist, config should be empty
        assert len(helper.config) == 0

    def test_save_config(self):
        """Test saving configuration"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            config_file = tmp.name

        try:
            helper = MCPConfigHelper(config_file)
            helper.config["test_key"] = "test_value"
            helper.save_config()

            # Verify file was written
            with open(config_file, "r") as f:
                saved_config = json.load(f)

            assert saved_config["test_key"] == "test_value"
        finally:
            os.unlink(config_file)

    def test_get_credentials(self):
        """Test getting credentials"""
        helper = MCPConfigHelper()
        helper.set_credentials(retrieval_endpoint="https://test.com")

        credentials = helper.get_credentials()
        assert credentials["RETRIEVAL_ENDPOINT"] == "https://test.com"

    def test_get_credentials_empty(self):
        """Test getting credentials when empty"""
        helper = MCPConfigHelper()

        credentials = helper.get_credentials()
        assert isinstance(credentials, dict)

    def test_set_credentials(self):
        """Test setting credentials"""
        helper = MCPConfigHelper()

        helper.set_credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )
        credentials = helper.get_credentials()
        assert credentials["RETRIEVAL_ENDPOINT"] == "https://test.com"
        assert credentials["RETRIEVAL_API_KEY"] == "test-key"

    def test_validate_credentials(self):
        """Test credential validation"""
        helper = MCPConfigHelper()

        # Test with missing credentials
        validation = helper.validate_credentials()
        assert isinstance(validation, dict)
        assert "valid" in validation

        # Test with valid credentials
        helper.set_credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )
        validation = helper.validate_credentials()
        assert isinstance(validation, dict)
        assert "valid" in validation

    def test_get_credentials_export(self):
        """Test getting credentials (export functionality)"""
        helper = MCPConfigHelper()
        helper.set_credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )

        exported = helper.get_credentials()
        assert isinstance(exported, dict)
        assert "RETRIEVAL_ENDPOINT" in exported
        assert "RETRIEVAL_API_KEY" in exported

    def test_set_credentials_import(self):
        """Test setting credentials (import functionality)"""
        helper = MCPConfigHelper()

        helper.set_credentials(
            retrieval_endpoint="https://test.com", retrieval_api_key="test-key"
        )
        credentials = helper.get_credentials()
        assert credentials["RETRIEVAL_ENDPOINT"] == "https://test.com"
        assert credentials["RETRIEVAL_API_KEY"] == "test-key"


class TestConfigHelperEdgeCases:
    """Test edge cases and error conditions"""

    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write("invalid json content")
            config_file = tmp.name

        try:
            helper = MCPConfigHelper(config_file)
            # Should fall back to default config
            assert isinstance(helper.config, dict)
        finally:
            os.unlink(config_file)

    def test_save_config_permission_error(self):
        """Test saving config with permission error"""
        helper = MCPConfigHelper("/root/invalid/path/config.json")

        # Should handle permission error gracefully
        try:
            helper.save_config()
        except (PermissionError, OSError):
            pass  # Expected behavior

    def test_validate_credentials_partial(self):
        """Test validation with partial credentials"""
        helper = MCPConfigHelper()
        helper.set_credentials(retrieval_endpoint="https://test.com")
        # Missing dify_api_key

        validation = helper.validate_credentials()
        assert isinstance(validation, dict)
        assert "valid" in validation

    def test_get_credentials_empty(self):
        """Test getting empty credentials"""
        helper = MCPConfigHelper()

        exported = helper.get_credentials()
        assert isinstance(exported, dict)

    def test_set_credentials_invalid(self):
        """Test setting credentials with invalid input"""
        helper = MCPConfigHelper()

        # Should handle invalid input gracefully
        helper.set_credentials()
        credentials = helper.get_credentials()
        assert isinstance(credentials, dict)
