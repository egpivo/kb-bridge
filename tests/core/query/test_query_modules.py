"""
Test Suite for Query Modules
Tests keyword generation, intention extraction, and query rewriting modules
"""

from unittest.mock import Mock, patch

import pytest

from kbbridge.core.query.intention_extractor import UserIntentionExtractor
from kbbridge.core.query.keyword_generator import (
    ContentBoostKeywordGenerator,
    FileSearchKeywordGenerator,
)
from kbbridge.core.utils.json_utils import parse_json_from_markdown

# Test fixtures for required parameters
TEST_LLM_API_TOKEN = "test-api-token"


class TestFileSearchKeywordGenerator:
    """Test FileSearchKeywordGenerator functionality"""

    def test_init(self):
        """Test keyword generator initialization"""
        generator = FileSearchKeywordGenerator(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )
        assert generator is not None

    def test_generate_keywords_success(self, mock_credentials):
        """Test successful keyword generation"""
        generator = FileSearchKeywordGenerator(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock the predictor to return keyword sets
        with patch.object(generator, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.keyword_sets = [["keyword1", "keyword2"], ["keyword3"]]
            mock_predictor.return_value = mock_result

            keywords = generator.generate(
                query="test query", max_sets=5, document_name="test_document.pdf"
            )

            assert isinstance(keywords, dict)
            assert keywords.get("success") is True

    def test_generate_keywords_api_error(self, mock_credentials):
        """Test keyword generation with API error"""
        generator = FileSearchKeywordGenerator(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock the predictor to raise an exception (simulating API error)
        with patch.object(generator, "predictor") as mock_predictor:
            mock_predictor.side_effect = Exception("DSPy API error")

            keywords = generator.generate(
                query="test query", max_sets=5, document_name="test_document.pdf"
            )

            assert isinstance(keywords, dict)
            assert keywords.get("success") is False

    def test_generate_keywords_network_error(self, mock_credentials):
        """Test keyword generation with network error"""
        generator = FileSearchKeywordGenerator(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock the predictor to raise a network-style exception
        with patch.object(generator, "predictor") as mock_predictor:
            mock_predictor.side_effect = Exception("Network error")

            keywords = generator.generate(
                query="test query", max_sets=5, document_name="test_document.pdf"
            )

            assert isinstance(keywords, dict)
            assert keywords.get("success") is False

    def test_generate_keywords_dspy_success(self, mock_credentials):
        """Test DSPy-backed keyword generation success path"""
        pytest.importorskip("dspy")

        # Mock the DSPy predictor to return keyword sets
        with patch("dspy.Predict") as mock_predict_cls:
            mock_predictor = Mock()
            mock_result = Mock()
            mock_result.keyword_sets = [["keyword1", "keyword2"]]
            mock_predictor.return_value = mock_result
            mock_predict_cls.return_value = mock_predictor

            generator = FileSearchKeywordGenerator(
                llm_api_url="https://api.openai.com/v1",
                llm_model="gpt-4",
                llm_api_token=TEST_LLM_API_TOKEN,
            )

            result = generator.generate(query="test query", max_sets=2)

            assert result["success"] is True
            assert result["keyword_sets"] == [["keyword1", "keyword2"]]

    def test_generate_keywords_dspy_failure_fallback(self, mock_credentials):
        """Test DSPy failure returns error (no HTTP fallback in pure DSPy)"""
        pytest.importorskip("dspy")

        # Mock the DSPy predictor to raise an error
        with patch("dspy.Predict") as mock_predict_cls:
            mock_predictor = Mock()
            mock_predictor.side_effect = RuntimeError("dspy boom")
            mock_predict_cls.return_value = mock_predictor

            generator = FileSearchKeywordGenerator(
                llm_api_url="https://api.openai.com/v1",
                llm_model="gpt-4",
                llm_api_token=TEST_LLM_API_TOKEN,
            )

            result = generator.generate(query="test query", max_sets=1)

            # Should return error, not fallback to HTTP
            assert result["success"] is False
            assert "error" in result


class TestContentBoostKeywordGenerator:
    """Test ContentBoostKeywordGenerator functionality"""

    def test_init(self):
        """Test content boost generator initialization"""
        generator = ContentBoostKeywordGenerator(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )
        assert generator is not None

    def test_generate_boost_keywords_success(self, mock_credentials):
        """Test successful boost keyword generation"""
        generator = ContentBoostKeywordGenerator(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock the predictor to return keyword sets
        with patch.object(generator, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.keyword_sets = [["boost1", "boost2"], ["boost3"]]
            mock_predictor.return_value = mock_result

            keywords = generator.generate(
                query="test query", max_sets=5, document_name="test_document.pdf"
            )

            assert isinstance(keywords, dict)
            assert keywords.get("success") is True

    def test_generate_boost_keywords_error(self, mock_credentials):
        """Test boost keyword generation with error"""
        generator = ContentBoostKeywordGenerator(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock the predictor to raise an exception
        with patch.object(generator, "predictor") as mock_predictor:
            mock_predictor.side_effect = Exception("DSPy API error")

            keywords = generator.generate(
                query="test query", max_sets=5, document_name="test_document.pdf"
            )

            assert isinstance(keywords, dict)
            assert keywords.get("success") is False


class TestUserIntentionExtractor:
    """Test UserIntentionExtractor functionality"""

    def test_init(self):
        """Test user intention extractor initialization"""
        extractor = UserIntentionExtractor(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )
        assert extractor is not None

    def test_extract_intention_success(self, mock_credentials):
        """Test successful intention extraction"""
        extractor = UserIntentionExtractor(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock DSPy predictor
        with patch.object(extractor, "predictor") as mock_predictor:
            mock_result = Mock()
            mock_result.intent = "search documents"
            mock_result.information_type = "document search"
            mock_result.relevant_documents = ["document1.pdf", "document2.pdf"]
            mock_result.query_complexity = "simple"
            mock_result.suggested_approach = "single search"
            mock_result.should_decompose = False
            mock_result.sub_queries = []
            mock_result.updated_query = "test query"
            mock_predictor.return_value = mock_result

            result = extractor.extract_intention(
                user_query="test query", doc_names=["document1.pdf", "document2.pdf"]
            )

            assert result["success"] is True
            assert result["updated_query"] == "test query"

    def test_extract_intention_error(self, mock_credentials):
        """Test intention extraction with error"""
        extractor = UserIntentionExtractor(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token=TEST_LLM_API_TOKEN,
        )

        # Mock DSPy predictor to raise an exception
        with patch.object(extractor, "predictor", side_effect=Exception("API error")):
            result = extractor.extract_intention(
                user_query="test query", doc_names=["document1.pdf", "document2.pdf"]
            )

            assert result["success"] is False
            assert "error" in result


class TestParseJsonFromMarkdown:
    """Test parse_json_from_markdown function"""

    def test_parse_json_from_markdown_json_block(self):
        """Test parsing JSON from markdown json block (covers lines 14-15)"""
        json_string = '```json\n[["keyword1", "keyword2"]]\n```'
        result = parse_json_from_markdown(json_string)

        assert "result" in result
        assert result["result"] == [["keyword1", "keyword2"]]

    def test_parse_json_from_markdown_generic_block(self):
        """Test parsing JSON from generic markdown block (covers lines 16-17)"""
        json_string = '```\n[["keyword1", "keyword2"]]\n```'
        result = parse_json_from_markdown(json_string)

        assert "result" in result
        assert result["result"] == [["keyword1", "keyword2"]]

    def test_parse_json_from_markdown_no_match(self):
        """Test parsing JSON with no match (covers line 18)"""
        json_string = "No JSON here"

        try:
            parse_json_from_markdown(json_string)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No JSON array found" in str(e)

    def test_parse_json_from_markdown_invalid_json(self):
        """Test parsing invalid JSON (covers lines 24-27)"""
        json_string = '```json\n[["keyword1", 123]]\n```'

        try:
            parse_json_from_markdown(json_string)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not an array of keyword sets" in str(e)

    def test_parse_json_from_markdown_not_list(self):
        """Test parsing JSON that is not a list (covers lines 24-27)"""
        json_string = '```json\n{"keyword": "value"}\n```'

        try:
            parse_json_from_markdown(json_string)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # The function raises "No JSON array found" when the JSON is not an array
            assert "No JSON array found" in str(e)


class TestLLMQueryRewriter:
    """Test LLMQueryRewriter functionality"""

    def test_init_with_token(self):
        """Test rewriter initialization with explicit token"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )
        assert rewriter is not None
        assert rewriter.llm_api_token == "test-token"

    def test_init_without_token_raises_error(self):
        """Test rewriter initialization without token raises error"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="LLM API token is required"):
                LLMQueryRewriter(
                    llm_api_url="https://api.openai.com/v1",
                    llm_model="gpt-4",
                    llm_api_token=None,
                )

    def test_init_with_env_token(self):
        """Test rewriter initialization with token from env"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        with patch.dict("os.environ", {"LLM_API_TOKEN": "env-token"}):
            rewriter = LLMQueryRewriter(
                llm_api_url="https://api.openai.com/v1",
                llm_model="gpt-4",
                llm_api_token=None,
            )
            assert rewriter.llm_api_token == "env-token"

    def test_init_with_cot_enabled(self):
        """Test rewriter initialization with chain-of-thought enabled"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        with patch("dspy.ChainOfThought") as mock_cot:
            rewriter = LLMQueryRewriter(
                llm_api_url="https://api.openai.com/v1",
                llm_model="gpt-4",
                llm_api_token="test-token",
                use_cot=True,
            )
            # Should use ChainOfThought for predictors
            assert mock_cot.call_count >= 3  # analyzer, expander, relaxer

    def test_rewrite_query_no_change_strategy(self):
        """Test query rewriting with NO_CHANGE strategy"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        # Mock the analyzer to return NO_CHANGE
        with patch.object(rewriter, "analyzer") as mock_analyzer:
            mock_result = Mock()
            mock_result.strategy = "NO_CHANGE"
            mock_result.confidence = 0.9
            mock_result.reason = "Query is already optimal"
            mock_result.query_analysis = "Simple query"
            mock_analyzer.return_value = mock_result

            result = rewriter.rewrite_query("simple query")

            assert result.strategy.value == "no_change"
            assert result.rewritten_query == "simple query"
            assert result.confidence == 0.9

    def test_rewrite_query_expansion_strategy(self):
        """Test query rewriting with EXPANSION strategy"""
        from kbbridge.core.query.models import RewriteStrategy
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        # Mock the analyzer to return EXPANSION
        with patch.object(rewriter, "analyzer") as mock_analyzer:
            mock_analysis = Mock()
            mock_analysis.strategy = "EXPANSION"
            mock_analysis.confidence = 0.8
            mock_analysis.reason = "Query too narrow"
            mock_analysis.query_analysis = "Narrow query"
            mock_analyzer.return_value = mock_analysis

            # Mock the expander
            with patch.object(rewriter, "expander") as mock_expander:
                mock_expand_result = Mock()
                mock_expand_result.expanded_query = "expanded query with synonyms"
                mock_expander.return_value = mock_expand_result

                result = rewriter.rewrite_query("narrow query")

                assert result.strategy == RewriteStrategy.EXPANSION
                assert result.rewritten_query == "expanded query with synonyms"
                assert result.confidence == 0.8

    def test_rewrite_query_relaxation_strategy(self):
        """Test query rewriting with RELAXATION strategy"""
        from kbbridge.core.query.models import RewriteStrategy
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        # Mock the analyzer to return RELAXATION
        with patch.object(rewriter, "analyzer") as mock_analyzer:
            mock_analysis = Mock()
            mock_analysis.strategy = "RELAXATION"
            mock_analysis.confidence = 0.7
            mock_analysis.reason = "Query too specific"
            mock_analysis.query_analysis = "Overly specific query"
            mock_analyzer.return_value = mock_analysis

            # Mock the relaxer
            with patch.object(rewriter, "relaxer") as mock_relaxer:
                mock_relax_result = Mock()
                mock_relax_result.relaxed_query = "relaxed broader query"
                mock_relaxer.return_value = mock_relax_result

                result = rewriter.rewrite_query("very specific detailed query")

                assert result.strategy == RewriteStrategy.RELAXATION
                assert result.rewritten_query == "relaxed broader query"
                assert result.confidence == 0.7

    def test_rewrite_query_with_context(self):
        """Test query rewriting with custom context"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        with patch.object(rewriter, "analyzer") as mock_analyzer:
            mock_result = Mock()
            mock_result.strategy = "NO_CHANGE"
            mock_result.confidence = 0.9
            mock_result.reason = "Context-aware"
            mock_result.query_analysis = "Legal domain query"
            mock_analyzer.return_value = mock_result

            result = rewriter.rewrite_query("query", context="Legal documents")

            # Verify analyzer was called with custom context
            mock_analyzer.assert_called_once()
            call_kwargs = mock_analyzer.call_args[1]
            assert call_kwargs["context"] == "Legal documents"

    def test_rewrite_query_missing_attributes(self):
        """Test query rewriting when analysis result has missing attributes"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        # Mock analyzer with missing attributes
        with patch.object(rewriter, "analyzer") as mock_analyzer:
            mock_result = Mock(spec=[])  # Empty spec = no attributes
            mock_analyzer.return_value = mock_result

            result = rewriter.rewrite_query("query")

            # Should use defaults
            assert result.strategy.value == "no_change"
            assert result.confidence == 0.5

    def test_rewrite_query_expander_missing_attribute(self):
        """Test expansion when expander result has missing attribute"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        with patch.object(rewriter, "analyzer") as mock_analyzer:
            mock_analysis = Mock()
            mock_analysis.strategy = "EXPANSION"
            mock_analysis.confidence = 0.8
            mock_analysis.reason = "Expand"
            mock_analysis.query_analysis = "Analysis"
            mock_analyzer.return_value = mock_analysis

            with patch.object(rewriter, "expander") as mock_expander:
                mock_result = Mock(spec=[])  # No expanded_query attribute
                mock_expander.return_value = mock_result

                result = rewriter.rewrite_query("query")

                # Should fallback to original query
                assert result.rewritten_query == "query"

    def test_rewrite_query_relaxer_missing_attribute(self):
        """Test relaxation when relaxer result has missing attribute"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        with patch.object(rewriter, "analyzer") as mock_analyzer:
            mock_analysis = Mock()
            mock_analysis.strategy = "RELAXATION"
            mock_analysis.confidence = 0.7
            mock_analysis.reason = "Relax"
            mock_analysis.query_analysis = "Analysis"
            mock_analyzer.return_value = mock_analysis

            with patch.object(rewriter, "relaxer") as mock_relaxer:
                mock_result = Mock(spec=[])  # No relaxed_query attribute
                mock_relaxer.return_value = mock_result

                result = rewriter.rewrite_query("query")

                # Should fallback to original query
                assert result.rewritten_query == "query"

    def test_rewrite_query_exception_handling(self):
        """Test query rewriting with exception during analysis"""
        from kbbridge.core.query.models import RewriteStrategy
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        with patch.object(rewriter, "analyzer", side_effect=Exception("API error")):
            result = rewriter.rewrite_query("query")

            assert result.strategy == RewriteStrategy.NO_CHANGE
            assert result.rewritten_query == "query"
            assert result.confidence == 0.0
            assert "Rewrite failed" in result.reason
            assert "API error" in result.metadata["error"]

    def test_rewrite_query_with_whitespace(self):
        """Test query rewriting strips whitespace from result"""
        from kbbridge.core.query.rewriter import LLMQueryRewriter

        rewriter = LLMQueryRewriter(
            llm_api_url="https://api.openai.com/v1",
            llm_model="gpt-4",
            llm_api_token="test-token",
        )

        with patch.object(rewriter, "analyzer") as mock_analyzer:
            mock_analysis = Mock()
            mock_analysis.strategy = "EXPANSION"
            mock_analysis.confidence = 0.8
            mock_analysis.reason = "Expand"
            mock_analysis.query_analysis = "Analysis"
            mock_analyzer.return_value = mock_analysis

            with patch.object(rewriter, "expander") as mock_expander:
                mock_result = Mock()
                mock_result.expanded_query = "  expanded query with spaces  "
                mock_expander.return_value = mock_result

                result = rewriter.rewrite_query("query")

                # Should strip whitespace
                assert result.rewritten_query == "expanded query with spaces"
