
"""
Test Suite for Core Models
Tests model classes and data structures
"""

from kbbridge.core.orchestration.models import Credentials, ProcessingConfig


class TestCredentials:
    """Test Credentials model"""

    def test_credentials_creation(self):
        """Test creating credentials with all fields"""
        creds = Credentials(
            retrieval_endpoint="https://dify-instance",
            retrieval_api_key="test-key",
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
            llm_temperature=0.7,
            llm_timeout=30,
            rerank_url="https://rerank.example.com",
            rerank_model="test-model",
        )

        assert creds.retrieval_endpoint == "https://dify-instance"
        assert creds.retrieval_api_key == "test-key"
        assert creds.llm_api_url == "https://api.openai.com/v1"
        assert creds.llm_model == "gpt-4"
        assert creds.llm_api_token == "test-token"
        assert creds.llm_temperature == 0.7
        assert creds.llm_timeout == 30
        assert creds.rerank_url == "https://rerank.example.com"
        assert creds.rerank_model == "test-model"

    def test_credentials_minimal(self):
        """Test creating credentials with minimal required fields"""
        creds = Credentials(
            retrieval_endpoint="https://dify-instance",
            retrieval_api_key="test-key",
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
        )

        assert creds.retrieval_endpoint == "https://dify-instance"
        assert creds.retrieval_api_key == "test-key"
        assert creds.llm_api_url == "https://api.openai.com/v1"
        assert creds.llm_model == "gpt-4"


class TestProcessingConfig:
    """Test ProcessingConfig model"""

    def test_processing_config_creation(self):
        """Test creating processing config"""
        config = ProcessingConfig(
            dataset_info='[{"id": "test-dataset", "source_path": "/test/path"}]',
            query="test query",
            max_workers=5,
            verbose=True,
            use_content_booster=True,
            max_boost_keywords=10,
            score_threshold=0.7,
            top_k=20,
        )

        assert (
            config.dataset_info
            == '[{"id": "test-dataset", "source_path": "/test/path"}]'
        )
        assert config.query == "test query"
        assert config.max_workers == 5
        assert config.verbose is True
        assert config.use_content_booster is True
        assert config.max_boost_keywords == 10
        assert config.score_threshold == 0.7
        assert config.top_k == 20

    def test_processing_config_defaults(self):
        """Test processing config with default values"""
        config = ProcessingConfig(
            dataset_info='[{"id": "test-dataset", "source_path": "/test/path"}]',
            query="test query",
        )

        # Should have reasonable defaults
        assert (
            config.dataset_info
            == '[{"id": "test-dataset", "source_path": "/test/path"}]'
        )
        assert config.query == "test query"
        assert config.max_workers >= 1
        assert isinstance(config.verbose, bool)
        assert isinstance(config.use_content_booster, bool)


# Additional model tests can be added when more models are available
