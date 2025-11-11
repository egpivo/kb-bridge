"""
Test Suite for Extraction Modules
Tests content clustering, paragraph location, and dataset parsing modules
"""

from unittest.mock import Mock, patch

from kbbridge.core.extraction.content_cluster import ContentCluster
from kbbridge.core.extraction.paragraph_locator import ParagraphLocator
from kbbridge.core.utils.json_utils import parse_dataset_info

# Test fixtures for required parameters
TEST_LLM_API_TOKEN = "test-api-token"


class TestContentCluster:
    """Test ContentCluster functionality"""

    def test_init(self):
        """Test content cluster initialization"""
        cluster = ContentCluster(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )
        assert cluster is not None

    def test_init_with_use_cot(self):
        """Test content cluster initialization with Chain of Thought (covers line 106)"""
        cluster = ContentCluster(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
            use_cot=True,  # This should use ChainOfThought predictor
        )
        assert cluster is not None
        # Verify it uses ChainOfThought predictor
        import dspy

        assert isinstance(cluster.predictor, dspy.ChainOfThought)

    def test_init_missing_api_token(self):
        """Test content cluster initialization without API token (covers line 113)"""
        from pytest import raises

        with raises(ValueError, match="LLM API token is required"):
            ContentCluster(
                llm_api_url="https://api.openai.com/v1",
                llm_model="gpt-4",
                llm_api_token=None,  # Missing token should raise ValueError
            )

    def test_cluster_single_anchor(self):
        """Test clustering with single anchor - insufficient data (covers line 158)"""
        cluster = ContentCluster(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        result = cluster.cluster(
            anchor_list=["single_anchor"], original_query="test query"
        )

        assert result["success"] is True
        assert result["cluster_method"] == "insufficient_data"
        assert len(result["clustered_anchors"]) == 1

    def test_cluster_empty_clusters_fallback(self):
        """Test clustering fallback when no valid clusters (covers line 190)"""
        cluster = ContentCluster(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock DSPy predictor to return empty clusters
        with patch.object(cluster, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.clusters = []  # Empty clusters
            mock_result.total_clusters = 0
            mock_result.unclustered = ["content1", "content2"]
            mock_predictor.return_value = mock_result

            result = cluster.cluster(
                anchor_list=["content1", "content2"], original_query="test query"
            )

            assert result["success"] is True
            assert result["cluster_method"] == "fallback"
            assert len(result["clustered_anchors"]) == 2

    def test_cluster_content_success(self, mock_credentials):
        """Test successful content clustering"""
        cluster = ContentCluster(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock DSPy predictor
        with patch.object(cluster, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_cluster_info = Mock()
            mock_cluster_info.theme = "test theme"
            mock_cluster_info.anchors = ["content1", "content2"]
            mock_cluster_info.description = "test cluster"
            mock_result.clusters = [mock_cluster_info]
            mock_result.total_clusters = 1
            mock_result.unclustered = []
            mock_predictor.return_value = mock_result

            result = cluster.cluster(
                anchor_list=["content1", "content2"], original_query="test query"
            )

            assert isinstance(result, dict)
            assert result["success"] is True
            assert "clustered_anchors" in result

    def test_cluster_content_error(self, mock_credentials):
        """Test content clustering with error"""
        cluster = ContentCluster(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock DSPy predictor to raise an exception
        with patch.object(cluster, "predictor", side_effect=Exception("API error")):
            result = cluster.cluster(
                anchor_list=["content1", "content2"], original_query="test query"
            )

            assert result["success"] is False
            assert "error" in result


class TestParagraphLocator:
    """Test ParagraphLocator functionality"""

    def test_init(self):
        """Test paragraph locator initialization"""
        locator = ParagraphLocator(
            llm_api_url="https://api.openai.com/v1", llm_model="gpt-4"
        )
        assert locator is not None

    def test_locate_paragraphs_success(self, mock_credentials):
        """Test successful paragraph location"""
        locator = ParagraphLocator(
            llm_api_url="https://api.openai.com/v1", llm_model="gpt-4"
        )

        # Mock the predictor to return a location
        with patch.object(locator, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.location = (
                "Section 5.2 Termination — confidentiality obligations"
            )
            mock_predictor.return_value = mock_result

            result = locator.locate(document_content="test content", query="test query")

            assert isinstance(result, dict)
            assert result["success"] is True
            assert "anchor_section" in result
            assert "anchor_term" in result

    def test_locate_paragraphs_error(self, mock_credentials):
        """Test paragraph location with error"""
        locator = ParagraphLocator(
            llm_api_url="https://api.openai.com/v1", llm_model="gpt-4"
        )

        # Mock the predictor to raise an exception
        with patch.object(locator, "predictor") as mock_predictor:
            mock_predictor.side_effect = Exception("DSPy error")

            result = locator.locate(document_content="test content", query="test query")

            assert "error" in result or not result.get("success", True)
            assert result.get("success") is False


class TestDatasetParser:
    """Test dataset parser functionality"""

    def test_parse_dataset_info_valid(self):
        """Test parsing valid dataset info"""
        dataset_info = '[{"id": "test-dataset"}]'
        result = parse_dataset_info(dataset_info)

        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "test-dataset"

    def test_parse_dataset_info_empty(self):
        """Test parsing empty dataset info"""
        result = parse_dataset_info("[]")
        assert result == []

    def test_parse_dataset_info_invalid_format(self):
        """Test parsing invalid dataset info format"""
        result = parse_dataset_info("invalid")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_parse_dataset_info_missing_id(self):
        """Test parsing dataset info with missing id"""
        dataset_info = '[{"other_field": "value"}]'
        result = parse_dataset_info(dataset_info)
        assert isinstance(result, list)

    def test_parse_dataset_info_with_uuids(self):
        """Test parsing dataset info with UUIDs (covers lines 25-27)"""
        # Test with UUIDs in the string
        dataset_info = '["550e8400-e29b-41d4-a716-446655440000", "6ba7b810-9dad-11d1-80b4-00c04fd430c8"]'
        result = parse_dataset_info(dataset_info)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert result[1]["id"] == "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

    def test_parse_dataset_info_json_unwrapping(self):
        """Test parsing dataset info with JSON unwrapping (covers lines 30-41)"""
        # Test with nested JSON strings
        dataset_info = '"[{\\"id\\": \\"test-dataset\\"}]"'
        result = parse_dataset_info(dataset_info)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "test-dataset"

    def test_parse_dataset_info_single_quotes(self):
        """Test parsing dataset info with single quotes (covers lines 34-35)"""
        # Test with single quotes
        dataset_info = '\'[{"id": "test-dataset"}]\''
        result = parse_dataset_info(dataset_info)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "test-dataset"

    def test_parse_dataset_info_non_string_input(self):
        """Test parsing dataset info with non-string input (covers lines 21-22)"""
        # Test with non-string input - this should cause an exception
        try:
            result = parse_dataset_info(123)
            # If no exception, check the result
            assert isinstance(result, list)
        except AttributeError:
            # Expected behavior for non-string input
            pass

    def test_parse_dataset_info_empty_string(self):
        """Test parsing dataset info with empty string"""
        result = parse_dataset_info("")

        assert isinstance(result, list)
