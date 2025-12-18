"""Tests for orchestration services module."""

from unittest.mock import patch

from kbbridge.config.constants import AssistantDefaults
from kbbridge.core.orchestration.services import ParameterValidator


class TestParameterValidator:
    """Test ParameterValidator class."""

    def test_validate_config_with_file_discovery_evaluation(self):
        """Test validate_config with file discovery evaluation parameters."""
        tool_parameters = {
            "resource_id": "test-id",
            "query": "test query",
            "enable_file_discovery_evaluation": True,
            "file_discovery_evaluation_threshold": 0.8,
        }

        config = ParameterValidator.validate_config(tool_parameters)

        assert config.enable_file_discovery_evaluation is True
        assert config.file_discovery_evaluation_threshold == 0.8

    def test_validate_config_file_discovery_threshold_invalid_high(self):
        """Test validate_config with invalid high threshold."""
        tool_parameters = {
            "resource_id": "test-id",
            "query": "test query",
            "file_discovery_evaluation_threshold": 1.5,  # Invalid: > 1.0
        }

        with patch("kbbridge.core.orchestration.services.logger") as mock_logger:
            config = ParameterValidator.validate_config(tool_parameters)

            assert (
                config.file_discovery_evaluation_threshold
                == AssistantDefaults.FILE_DISCOVERY_EVALUATION_THRESHOLD.value
            )
            mock_logger.warning.assert_called_once()

    def test_validate_config_file_discovery_threshold_invalid_low(self):
        """Test validate_config with invalid low threshold."""
        tool_parameters = {
            "resource_id": "test-id",
            "query": "test query",
            "file_discovery_evaluation_threshold": -0.1,  # Invalid: < 0.0
        }

        with patch("kbbridge.core.orchestration.services.logger") as mock_logger:
            config = ParameterValidator.validate_config(tool_parameters)

            assert (
                config.file_discovery_evaluation_threshold
                == AssistantDefaults.FILE_DISCOVERY_EVALUATION_THRESHOLD.value
            )
            mock_logger.warning.assert_called_once()

    def test_validate_config_file_discovery_evaluation_defaults(self):
        """Test validate_config uses defaults when parameters not provided."""
        tool_parameters = {
            "resource_id": "test-id",
            "query": "test query",
        }

        config = ParameterValidator.validate_config(tool_parameters)

        assert (
            config.enable_file_discovery_evaluation
            == AssistantDefaults.ENABLE_FILE_DISCOVERY_EVALUATION.value
        )
        assert (
            config.file_discovery_evaluation_threshold
            == AssistantDefaults.FILE_DISCOVERY_EVALUATION_THRESHOLD.value
        )
