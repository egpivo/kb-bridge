"""
Tests for DifyBackendAdapter
"""

from unittest.mock import Mock, patch

import pytest

from kbbridge.integrations.credentials import RetrievalCredentials
from kbbridge.integrations.dify import DifyBackendAdapter
from kbbridge.integrations.dify.dify_credentials import DifyCredentials


class TestDifyBackendAdapter:
    """Test DifyBackendAdapter class"""

    def test_init_with_retrieval_credentials(self):
        """Test initialization with RetrievalCredentials"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id="test-resource")
        assert adapter.resource_id == "test-resource"
        assert adapter.credentials == creds

    def test_init_with_dify_credentials(self):
        """Test initialization with DifyCredentials (backward compatibility)"""
        dify_creds = DifyCredentials(endpoint="https://test.com", api_key="test-key")
        adapter = DifyBackendAdapter(dify_creds, resource_id="test-resource")
        assert adapter.resource_id == "test-resource"
        assert isinstance(adapter.credentials, RetrievalCredentials)

    @patch("kbbridge.integrations.dify.dify_backend_adapter.DifyCredentials.from_env")
    def test_init_with_none_credentials(self, mock_from_env):
        """Test initialization with None credentials loads from env"""
        mock_from_env.return_value = DifyCredentials(
            endpoint="https://test.com", api_key="test-key"
        )
        adapter = DifyBackendAdapter(None, resource_id="test-resource")
        assert adapter.resource_id == "test-resource"
        assert isinstance(adapter.credentials, RetrievalCredentials)

    def test_init_resource_bound(self):
        """Test resource-bound initialization creates retriever"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        with patch(
            "kbbridge.integrations.dify.dify_backend_adapter.DifyRetriever"
        ) as mock_retriever:
            adapter = DifyBackendAdapter(creds, resource_id="test-resource")
            assert adapter._dify_retriever is not None
            mock_retriever.assert_called_once()

    def test_init_non_resource_bound(self):
        """Test non-resource-bound initialization doesn't create retriever"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id=None)
        assert adapter._dify_retriever is None

    def test_get_retriever_resource_bound(self):
        """Test _get_retriever returns existing retriever when resource-bound"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id="test-resource")
        retriever = adapter._get_retriever()
        assert retriever == adapter._dify_retriever

    def test_get_retriever_non_resource_bound(self):
        """Test _get_retriever creates retriever when not resource-bound"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id=None)
        with patch(
            "kbbridge.integrations.dify.dify_backend_adapter.DifyRetriever"
        ) as mock_retriever:
            retriever = adapter._get_retriever(resource_id="test-resource")
            mock_retriever.assert_called_once()
            assert retriever is not None

    def test_get_retriever_raises_when_no_resource_id(self):
        """Test _get_retriever raises ValueError when resource_id not provided"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id=None)
        with pytest.raises(ValueError, match="resource_id is required"):
            adapter._get_retriever()

    def test_search_with_dataset_id_backward_compatibility(self):
        """Test search() accepts dataset_id for backward compatibility"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id=None)
        mock_retriever = Mock()
        mock_retriever.build_metadata_filter.return_value = None
        mock_retriever.call.return_value = {"records": []}

        with patch.object(adapter, "_get_retriever", return_value=mock_retriever):
            adapter.search(query="test", dataset_id="test-dataset")
            mock_retriever.call.assert_called_once()

    def test_list_files_with_dataset_id_backward_compatibility(self):
        """Test list_files() accepts dataset_id for backward compatibility"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id=None)
        mock_retriever = Mock()
        mock_retriever.list_files.return_value = ["file1.pdf", "file2.pdf"]

        with patch.object(adapter, "_get_retriever", return_value=mock_retriever):
            files = adapter.list_files(dataset_id="test-dataset")
            assert files == ["file1.pdf", "file2.pdf"]
            mock_retriever.list_files.assert_called_once()

    def test_build_metadata_filter_non_resource_bound(self):
        """Test build_metadata_filter() creates temp retriever when not resource-bound"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id=None)
        mock_retriever = Mock()
        mock_retriever.build_metadata_filter.return_value = {"conditions": []}

        with patch(
            "kbbridge.integrations.dify.dify_backend_adapter.DifyRetriever",
            return_value=mock_retriever,
        ):
            result = adapter.build_metadata_filter(document_name="test.pdf")
            assert result == {"conditions": []}
            mock_retriever.build_metadata_filter.assert_called_once_with(
                document_name="test.pdf"
            )

    def test_create_retriever(self):
        """Test create_retriever() backward compatibility method"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id="test-resource")
        with patch(
            "kbbridge.integrations.dify.dify_backend_adapter.DifyRetriever"
        ) as mock_retriever:
            retriever = adapter.create_retriever("test-dataset", timeout=60)
            mock_retriever.assert_called_once_with(
                endpoint="https://test.com",
                api_key="test-key",
                dataset_id="test-dataset",
                timeout=60,
            )

    def test_get_credentials_summary(self):
        """Test get_credentials_summary() method"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter(creds, resource_id="test-resource")
        summary = adapter.get_credentials_summary()
        assert isinstance(summary, dict)
        # DifyCredentials.get_masked_summary() returns dict with dify-specific keys
        assert len(summary) > 0

    @patch(
        "kbbridge.integrations.dify.dify_backend_adapter.RetrievalCredentials.from_env"
    )
    def test_from_env(self, mock_from_env):
        """Test from_env() class method"""
        mock_from_env.return_value = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = DifyBackendAdapter.from_env(resource_id="test-resource")
        assert adapter.resource_id == "test-resource"
        mock_from_env.assert_called_once()

    def test_from_params(self):
        """Test from_params() class method"""
        adapter = DifyBackendAdapter.from_params(
            dify_endpoint="https://test.com",
            dify_api_key="test-key",
            resource_id="test-resource",
        )
        assert adapter.resource_id == "test-resource"
        assert adapter.credentials.endpoint == "https://test.com"
        assert adapter.credentials.api_key == "test-key"

    def test_create_dify_adapter_function(self):
        """Test create_dify_adapter() convenience function"""
        from kbbridge.integrations.dify.dify_backend_adapter import create_dify_adapter

        adapter = create_dify_adapter(
            dify_endpoint="https://test.com",
            dify_api_key="test-key",
            resource_id="test-resource",
        )
        assert adapter.resource_id == "test-resource"
        assert adapter.credentials.endpoint == "https://test.com"
