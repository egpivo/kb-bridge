"""
Test Configuration and Fixtures
Provides common test fixtures and configuration for all test modules
"""

import os
from unittest.mock import patch

import pytest

from kbbridge.config.env_loader import load_env_file

load_env_file()


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--test-port",
        action="store",
        default="9004",
        help="Port to use for test server (default: 9004, dev uses 9003)",
    )


@pytest.fixture(scope="session")
def test_port(request):
    """Get the test port from command line or environment."""
    # Check environment variable first, then command line, then default
    env_port = os.getenv("TEST_PORT")
    if env_port:
        return int(env_port)
    return int(request.config.getoption("--test-port"))


@pytest.fixture(scope="session")
def test_server_url(test_port):
    """Get the full test server URL."""
    return f"http://0.0.0.0:{test_port}"


@pytest.fixture(scope="session")
def test_mcp_endpoint(test_server_url):
    """Get the test MCP endpoint URL."""
    return f"{test_server_url}/mcp"


@pytest.fixture
def mock_credentials():
    """Mock credentials for testing"""
    # Create a simple mock credentials object to avoid import issues
    class MockCredentials:
        def __init__(self):
            self.retrieval_endpoint = "https://dify.com"
            self.retrieval_api_key = "test-api-key"
            self.llm_api_url = "https://api.openai.com/v1"
            self.llm_model = "gpt-4"
            self.llm_api_token = "test-token"
            self.llm_temperature = 0.7
            self.llm_timeout = 30
            self.rerank_url = "https://rerank.example.com"
            self.rerank_model = "test-rerank-model"

    return MockCredentials()


@pytest.fixture
def mock_dify_response():
    """Mock Dify API response"""
    return {
        "records": [
            {
                "segment": {
                    "content": "Test content 1",
                    "document": {"doc_metadata": {"document_name": "test_doc1.pdf"}},
                }
            },
            {
                "segment": {
                    "content": "Test content 2",
                    "document": {"doc_metadata": {"document_name": "test_doc2.pdf"}},
                }
            },
        ]
    }


@pytest.fixture
def mock_dify_error_response():
    """Mock Dify API error response"""
    return {"error": "Test error message", "details": {"field": ["Validation error"]}}


@pytest.fixture
def test_tool_parameters():
    """Test tool parameters"""
    return {
        "dataset_id": "test-dataset-id",
        "query": "test query",
        "verbose": True,
        "score_threshold": 0.7,
        "top_k": 10,
        "max_workers": 3,
        "use_content_booster": True,
        "max_boost_keywords": 5,
    }


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for testing"""
    with patch("requests.post") as mock_post:
        yield mock_post


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for testing"""
    with patch("requests.get") as mock_get:
        yield mock_get


# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


# Environment variables for testing
@pytest.fixture(autouse=True)  # Re-enabled but now safer
def setup_test_env():
    """Set up test environment variables"""
    # Store original values to restore later
    original_values = {}
    test_keys = ["DIFY_ENDPOINT", "DIFY_API_KEY", "LLM_API_URL", "LLM_MODEL"]

    # Store original values
    for key in test_keys:
        original_values[key] = os.environ.get(key)

    # Only set test defaults if no value exists AND we're in a test environment
    # This prevents overwriting user's environment variables
    if os.getenv("PYTEST_CURRENT_TEST"):  # Only set defaults during pytest runs
        os.environ.setdefault("RETRIEVAL_ENDPOINT", "https://dify.com")
        os.environ.setdefault("RETRIEVAL_API_KEY", "test-api-key")
        os.environ.setdefault("LLM_API_URL", "https://api.openai.com/v1")
        os.environ.setdefault("LLM_MODEL", "gpt-4")

    yield

    # Restore original values
    for key in test_keys:
        if original_values[key] is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_values[key]
