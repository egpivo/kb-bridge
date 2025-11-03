"""
Tests for retriever.py to improve coverage
"""
from kbbridge.utils.formatting import format_search_results


class TestRetrieverCoverage:
    """Tests to improve coverage for retriever.py"""

    def test_format_search_results_dict_input(self):
        """Test format_search_results with dict input (covers lines 13-14)"""
        results = {
            "records": [
                {
                    "segment": {
                        "content": "test content",
                        "document": {"doc_metadata": {"document_name": "test.doc"}},
                    }
                }
            ]
        }

        result = format_search_results(results)

        assert "result" in result
        assert len(result["result"]) == 1
        assert result["result"][0]["content"] == "test content"

    def test_format_search_results_empty_records(self):
        """Test format_search_results with empty records (covers lines 28-34)"""
        results = [{"records": []}]

        result = format_search_results(results)

        assert "result" in result
        assert result["result"] == []

    def test_format_search_results_exception_handling(self):
        """Test format_search_results exception handling (covers lines 39-41)"""
        # Test with malformed data that causes exception
        results = [{"invalid": "data"}]

        result = format_search_results(results)

        assert "result" in result
        # The function handles the exception gracefully and returns empty result
        assert result["result"] == []
