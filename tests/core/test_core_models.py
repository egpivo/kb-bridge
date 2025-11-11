"""
Test Suite for Core Models
Tests model classes and data structures
"""

from kbbridge.core.orchestration.models import (
    CandidateAnswer,
    Credentials,
    ProcessingConfig,
)


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
            resource_id="test-dataset",
            query="test query",
            max_workers=5,
            verbose=True,
            use_content_booster=True,
            max_boost_keywords=10,
            score_threshold=0.7,
            top_k=20,
        )

        assert config.resource_id == "test-dataset"
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
            resource_id="test-dataset",
            query="test query",
        )

        # Should have reasonable defaults
        assert config.resource_id == "test-dataset"
        assert config.query == "test query"
        assert config.max_workers >= 1
        assert isinstance(config.verbose, bool)
        assert isinstance(config.use_content_booster, bool)


class TestCandidateAnswer:
    """Test CandidateAnswer model"""

    def test_to_dict_includes_dataset_id(self):
        """Test to_dict() includes dataset_id for backward compatibility"""
        candidate = CandidateAnswer(
            source="direct",
            answer="Test answer",
            success=True,
            resource_id="test-resource",
        )
        result = candidate.to_dict()
        assert result["resource_id"] == "test-resource"
        assert result["dataset_id"] == "test-resource"  # Backward compatibility

    def test_from_dict_prefers_resource_id(self):
        """Test from_dict() prefers resource_id over dataset_id"""
        data = {
            "source": "direct",
            "answer": "Test answer",
            "success": True,
            "resource_id": "test-resource",
            "dataset_id": "old-dataset",  # Should be ignored
        }
        candidate = CandidateAnswer.from_dict(data)
        assert candidate.resource_id == "test-resource"

    def test_from_dict_falls_back_to_dataset_id(self):
        """Test from_dict() falls back to dataset_id if resource_id not present"""
        data = {
            "source": "direct",
            "answer": "Test answer",
            "success": True,
            "dataset_id": "test-dataset",  # Old format
        }
        candidate = CandidateAnswer.from_dict(data)
        assert candidate.resource_id == "test-dataset"


# Additional model tests can be added when more models are available
