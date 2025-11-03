"""
Additional test for kb_utils.py to ensure 100% coverage
"""
from kbbridge.utils.kb_utils import build_context_from_segments, format_debug_details


class TestKBUtilsAdditional:
    """Additional tests for kb_utils.py"""

    def test_format_debug_details_edge_cases(self):
        """Test format_debug_details with edge cases"""
        # Test with empty list
        result = format_debug_details([])
        assert result == []

        # Test with None values
        result = format_debug_details([None, "test"])
        assert len(result) == 2
        assert "None" in result[0]
        assert "test" in result[1]

    def test_build_context_from_segments_edge_cases(self):
        """Test build_context_from_segments with edge cases"""
        # Test with segments containing None values
        segments = [
            {"content": "test", "document_name": None},
            {"content": None, "document_name": "doc"},
            {"content": "", "document_name": ""},
        ]

        result = build_context_from_segments(segments)
        assert isinstance(result, str)

        # Test verbose mode
        result_verbose = build_context_from_segments(segments, verbose=True)
        assert "Context built from" in result_verbose
