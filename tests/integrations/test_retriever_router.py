"""
Tests for RetrieverRouter

Comprehensive test coverage for retriever routing functionality.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from kbbridge.integrations.retriever_router import (
    RetrieverRouter,
    create_retriever_from_env,
    make_retriever,
)


class TestRetrieverRouterRegistry:
    """Test retriever registry functionality"""

    def test_register_retriever(self):
        """Test registering a retriever"""
        mock_retriever_class = MagicMock()
        backend_type = "test_backend"

        RetrieverRouter.register_retriever(backend_type, mock_retriever_class)

        assert backend_type.lower() in RetrieverRouter._retrievers
        assert RetrieverRouter._retrievers[backend_type.lower()] == mock_retriever_class

    def test_register_retriever_lowercase(self):
        """Test that backend types are converted to lowercase"""
        mock_retriever_class = MagicMock()
        backend_type = "TEST_BACKEND"

        RetrieverRouter.register_retriever(backend_type, mock_retriever_class)

        assert backend_type.lower() in RetrieverRouter._retrievers
        assert "TEST_BACKEND" not in RetrieverRouter._retrievers

    def test_get_available_backends(self):
        """Test getting list of available backends"""
        # Register a test retriever
        mock_retriever_class = MagicMock()
        RetrieverRouter.register_retriever("test_backend", mock_retriever_class)

        backends = RetrieverRouter.get_available_backends()

        assert isinstance(backends, list)
        assert "test_backend" in backends


class TestRetrieverRouterCreateRetriever:
    """Test create_retriever method"""

    def test_create_retriever_with_backend_type(self):
        """Test creating retriever with explicit backend type"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("test_backend", mock_retriever_class)

        with patch.dict(os.environ, {}, clear=True):
            result = RetrieverRouter.create_retriever(
                dataset_id="test-dataset",
                backend_type="test_backend",
                endpoint="https://test.com",
                api_key="test-key",
            )

        assert result == mock_instance
        mock_retriever_class.assert_called_once()

    def test_create_retriever_from_env_var(self):
        """Test creating retriever using RETRIEVER_BACKEND env var"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("dify", mock_retriever_class)

        with patch.dict(os.environ, {"RETRIEVER_BACKEND": "dify"}, clear=True):
            result = RetrieverRouter.create_retriever(
                dataset_id="test-dataset",
                endpoint="https://test.com",
                api_key="test-key",
            )

        assert result == mock_instance
        mock_retriever_class.assert_called_once()

    def test_create_retriever_default_backend(self):
        """Test creating retriever with default backend (dify)"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("dify", mock_retriever_class)

        with patch.dict(os.environ, {}, clear=True):
            result = RetrieverRouter.create_retriever(
                dataset_id="test-dataset",
                endpoint="https://test.com",
                api_key="test-key",
            )

        assert result == mock_instance

    def test_create_retriever_unsupported_backend(self):
        """Test creating retriever with unsupported backend raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                RetrieverRouter.create_retriever(
                    dataset_id="test-dataset",
                    backend_type="unsupported_backend",
                )

        assert "Unsupported backend type" in str(exc_info.value)
        assert "unsupported_backend" in str(exc_info.value)


class TestRetrieverRouterBuildConfig:
    """Test _build_config method for different backends"""

    def test_build_config_dify_with_retrieval_vars(self):
        """Test building config for Dify backend with RETRIEVAL_* vars"""
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_ENDPOINT": "https://retrieval.com",
                "RETRIEVAL_API_KEY": "retrieval-key",
            },
            clear=True,
        ):
            config = RetrieverRouter._build_config("dify", "test-dataset", timeout=60)

        assert config["dataset_id"] == "test-dataset"
        assert config["endpoint"] == "https://retrieval.com"
        assert config["api_key"] == "retrieval-key"
        assert config["timeout"] == 60

    def test_build_config_dify_with_dify_vars_fallback(self):
        """Test building config for Dify backend with DIFY_* vars as fallback"""
        with patch.dict(
            os.environ,
            {
                "DIFY_ENDPOINT": "https://dify.com",
                "DIFY_API_KEY": "dify-key",
            },
            clear=True,
        ):
            config = RetrieverRouter._build_config("dify", "test-dataset")

        assert config["endpoint"] == "https://dify.com"
        assert config["api_key"] == "dify-key"
        assert config["timeout"] == 30  # default

    def test_build_config_dify_with_kwargs(self):
        """Test building config for Dify backend with kwargs"""
        with patch.dict(os.environ, {}, clear=True):
            config = RetrieverRouter._build_config(
                "dify",
                "test-dataset",
                endpoint="https://kwarg.com",
                api_key="kwarg-key",
                timeout=45,
            )

        assert config["endpoint"] == "https://kwarg.com"
        assert config["api_key"] == "kwarg-key"
        assert config["timeout"] == 45

    def test_build_config_dify_prefers_retrieval_over_dify(self):
        """Test that RETRIEVAL_* vars are preferred over DIFY_* vars"""
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_ENDPOINT": "https://retrieval.com",
                "RETRIEVAL_API_KEY": "retrieval-key",
                "DIFY_ENDPOINT": "https://dify.com",
                "DIFY_API_KEY": "dify-key",
            },
            clear=True,
        ):
            config = RetrieverRouter._build_config("dify", "test-dataset")

        assert config["endpoint"] == "https://retrieval.com"
        assert config["api_key"] == "retrieval-key"

    def test_build_config_dify_missing_endpoint(self):
        """Test that missing endpoint raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                RetrieverRouter._build_config("dify", "test-dataset")

        assert "RETRIEVAL_ENDPOINT or DIFY_ENDPOINT" in str(exc_info.value)
        assert "required for Dify backend" in str(exc_info.value)

    def test_build_config_dify_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        with patch.dict(
            os.environ, {"RETRIEVAL_ENDPOINT": "https://test.com"}, clear=True
        ):
            with pytest.raises(ValueError) as exc_info:
                RetrieverRouter._build_config("dify", "test-dataset")

        assert "RETRIEVAL_API_KEY or DIFY_API_KEY" in str(exc_info.value)
        assert "required for Dify backend" in str(exc_info.value)

    def test_build_config_opensearch(self):
        """Test building config for OpenSearch backend"""
        with patch.dict(
            os.environ,
            {
                "OPENSEARCH_ENDPOINT": "https://opensearch.com",
                "OPENSEARCH_AUTH": "auth-key",
            },
            clear=True,
        ):
            config = RetrieverRouter._build_config("opensearch", "test-dataset")

        assert config["dataset_id"] == "test-dataset"
        assert config["endpoint"] == "https://opensearch.com"
        assert config["auth"] == "auth-key"
        assert config["index_name"] == "test-dataset"  # defaults to dataset_id
        assert config["timeout"] == 30

    def test_build_config_opensearch_with_kwargs(self):
        """Test building config for OpenSearch backend with kwargs"""
        with patch.dict(os.environ, {}, clear=True):
            config = RetrieverRouter._build_config(
                "opensearch",
                "test-dataset",
                endpoint="https://kwarg-os.com",
                auth="kwarg-auth",
                index_name="custom-index",
                timeout=60,
            )

        assert config["endpoint"] == "https://kwarg-os.com"
        assert config["auth"] == "kwarg-auth"
        assert config["index_name"] == "custom-index"
        assert config["timeout"] == 60

    def test_build_config_opensearch_with_index_env_var(self):
        """Test building config for OpenSearch with OPENSEARCH_INDEX env var"""
        with patch.dict(
            os.environ,
            {
                "OPENSEARCH_ENDPOINT": "https://opensearch.com",
                "OPENSEARCH_AUTH": "auth-key",
                "OPENSEARCH_INDEX": "env-index",
            },
            clear=True,
        ):
            config = RetrieverRouter._build_config("opensearch", "test-dataset")

        assert config["index_name"] == "env-index"

    def test_build_config_opensearch_missing_endpoint(self):
        """Test that missing OpenSearch endpoint raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                RetrieverRouter._build_config("opensearch", "test-dataset")

        assert "OPENSEARCH_ENDPOINT" in str(exc_info.value)
        assert "required for OpenSearch backend" in str(exc_info.value)

    def test_build_config_n8n(self):
        """Test building config for n8n backend"""
        with patch.dict(
            os.environ,
            {
                "N8N_WEBHOOK_URL": "https://n8n.com/webhook",
                "N8N_API_KEY": "n8n-key",
            },
            clear=True,
        ):
            config = RetrieverRouter._build_config("n8n", "test-dataset")

        assert config["dataset_id"] == "test-dataset"
        assert config["webhook_url"] == "https://n8n.com/webhook"
        assert config["api_key"] == "n8n-key"
        assert config["timeout"] == 30

    def test_build_config_n8n_with_kwargs(self):
        """Test building config for n8n backend with kwargs"""
        with patch.dict(os.environ, {}, clear=True):
            config = RetrieverRouter._build_config(
                "n8n",
                "test-dataset",
                webhook_url="https://kwarg-n8n.com/webhook",
                api_key="kwarg-n8n-key",
                timeout=45,
            )

        assert config["webhook_url"] == "https://kwarg-n8n.com/webhook"
        assert config["api_key"] == "kwarg-n8n-key"
        assert config["timeout"] == 45

    def test_build_config_n8n_missing_webhook_url(self):
        """Test that missing n8n webhook URL raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                RetrieverRouter._build_config("n8n", "test-dataset")

        assert "N8N_WEBHOOK_URL" in str(exc_info.value)
        assert "required for n8n backend" in str(exc_info.value)

    def test_build_config_custom_backend(self):
        """Test building config for custom backend (passes through kwargs)"""
        config = RetrieverRouter._build_config(
            "custom_backend",
            "test-dataset",
            custom_param="custom_value",
            another_param=123,
            timeout=60,
        )

        assert config["dataset_id"] == "test-dataset"
        assert config["custom_param"] == "custom_value"
        assert config["another_param"] == 123
        assert config["timeout"] == 60


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_make_retriever(self):
        """Test make_retriever convenience function"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("test_kind", mock_retriever_class)

        with patch.dict(os.environ, {}, clear=True):
            result = make_retriever(
                "test_kind",
                dataset_id="test-dataset",
                endpoint="https://test.com",
                api_key="test-key",
            )

        assert result == mock_instance
        mock_retriever_class.assert_called_once()

    def test_make_retriever_default_dataset_id(self):
        """Test make_retriever with default dataset_id"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("test_kind", mock_retriever_class)

        with patch.dict(os.environ, {}, clear=True):
            result = make_retriever(
                "test_kind",
                endpoint="https://test.com",
                api_key="test-key",
            )

        # Check that dataset_id was set to "default"
        call_kwargs = mock_retriever_class.call_args[1]
        assert call_kwargs["dataset_id"] == "default"

    def test_create_retriever_from_env(self):
        """Test create_retriever_from_env convenience function"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("dify", mock_retriever_class)

        with patch.dict(os.environ, {"RETRIEVER_BACKEND": "dify"}, clear=True):
            result = create_retriever_from_env(
                dataset_id="test-dataset",
                endpoint="https://test.com",
                api_key="test-key",
            )

        assert result == mock_instance
        mock_retriever_class.assert_called_once()

    def test_create_retriever_from_env_default_backend(self):
        """Test create_retriever_from_env with default backend"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("dify", mock_retriever_class)

        with patch.dict(os.environ, {}, clear=True):
            result = create_retriever_from_env(
                dataset_id="test-dataset",
                endpoint="https://test.com",
                api_key="test-key",
            )

        assert result == mock_instance


class TestIntegrationScenarios:
    """Test integration scenarios with multiple configurations"""

    def test_dify_backend_full_flow(self):
        """Test full flow for Dify backend creation"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("dify", mock_retriever_class)

        with patch.dict(
            os.environ,
            {
                "RETRIEVER_BACKEND": "dify",
                "RETRIEVAL_ENDPOINT": "https://test.com",
                "RETRIEVAL_API_KEY": "test-key",
            },
            clear=True,
        ):
            retriever = RetrieverRouter.create_retriever(dataset_id="test-dataset")

        assert retriever == mock_instance
        call_kwargs = mock_retriever_class.call_args[1]
        assert call_kwargs["dataset_id"] == "test-dataset"
        assert call_kwargs["endpoint"] == "https://test.com"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["timeout"] == 30

    def test_backend_type_case_insensitive(self):
        """Test that backend type is case insensitive"""
        mock_retriever_class = MagicMock()
        mock_instance = MagicMock()
        mock_retriever_class.return_value = mock_instance

        RetrieverRouter.register_retriever("test_backend", mock_retriever_class)

        with patch.dict(os.environ, {}, clear=True):
            # Try with different cases
            result1 = RetrieverRouter.create_retriever(
                dataset_id="test",
                backend_type="TEST_BACKEND",
                endpoint="https://test.com",
                api_key="key",
            )
            result2 = RetrieverRouter.create_retriever(
                dataset_id="test",
                backend_type="Test_Backend",
                endpoint="https://test.com",
                api_key="key",
            )

        assert result1 == mock_instance
        assert result2 == mock_instance
