"""
Additional tests to reach 80% coverage threshold
"""
from kbbridge.utils.formatting import format_search_results


class TestAdditionalCoverage:
    """Additional tests to reach 80% coverage"""

    def test_format_search_results_missing_records_key(self):
        """Test format_search_results when records key is missing (covers lines 28, 32-34)"""
        # Test with dict that doesn't have 'records' key
        results = {"invalid": "data"}

        result = format_search_results(results)

        assert "result" in result
        assert result["result"] == []

    def test_format_search_results_empty_list(self):
        """Test format_search_results with empty list (covers line 10)"""
        results = []

        result = format_search_results(results)

        assert "result" in result
        assert result["result"] == []

    def test_format_search_results_none_input(self):
        """Test format_search_results with None input"""
        result = format_search_results(None)

        assert "result" in result
        # The function handles None input gracefully
        assert result["result"] == []
