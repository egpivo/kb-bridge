"""
Tests for BackendAdapter and BackendAdapterFactory
"""


import pytest

from kbbridge.integrations.backend_adapter import BackendAdapter, BackendAdapterFactory
from kbbridge.integrations.credentials import RetrievalCredentials


class TestBackendAdapter:
    """Test BackendAdapter base class"""

    def test_backend_id_property_with_resource_id(self):
        """Test _backend_id property when resource_id is provided"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )

        class ConcreteAdapter(BackendAdapter):
            def search(self, **kwargs):
                pass

            def list_files(self, **kwargs):
                pass

            def build_metadata_filter(self, **kwargs):
                pass

        adapter = ConcreteAdapter(creds, resource_id="test-resource")
        assert adapter._backend_id == "test-resource"

    def test_backend_id_property_without_resource_id(self):
        """Test _backend_id property raises ValueError when resource_id is None"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )

        class ConcreteAdapter(BackendAdapter):
            def search(self, **kwargs):
                pass

            def list_files(self, **kwargs):
                pass

            def build_metadata_filter(self, **kwargs):
                pass

        adapter = ConcreteAdapter(creds, resource_id=None)
        with pytest.raises(ValueError, match="resource_id is required"):
            _ = adapter._backend_id


class TestBackendAdapterFactory:
    """Test BackendAdapterFactory"""

    def test_create_dify_backend(self):
        """Test creating Dify backend adapter"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = BackendAdapterFactory.create("test-resource", creds)
        assert adapter.resource_id == "test-resource"
        assert adapter.credentials == creds

    def test_create_with_backend_type_override(self):
        """Test creating adapter with backend_type override"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="dify"
        )
        adapter = BackendAdapterFactory.create(
            "test-resource", creds, backend_type="dify"
        )
        assert adapter.resource_id == "test-resource"

    def test_create_opensearch_backend(self):
        """Test creating OpenSearch backend adapter raises NotImplementedError"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="opensearch"
        )
        with pytest.raises(NotImplementedError, match="OpenSearch backend adapter"):
            BackendAdapterFactory.create("test-resource", creds)

    def test_create_n8n_backend(self):
        """Test creating n8n backend adapter raises NotImplementedError"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="n8n"
        )
        with pytest.raises(NotImplementedError, match="n8n backend adapter"):
            BackendAdapterFactory.create("test-resource", creds)

    def test_create_unsupported_backend(self):
        """Test creating unsupported backend raises ValueError"""
        creds = RetrievalCredentials(
            endpoint="https://test.com", api_key="test-key", backend_type="unknown"
        )
        with pytest.raises(ValueError, match="Unsupported backend type"):
            BackendAdapterFactory.create("test-resource", creds)
