
"""
Test Suite for Utility Modules
Tests utility functions and helpers
"""

from kbbridge.utils.kb_utils import build_context_from_segments, format_debug_details


class TestFormatDebugDetails:
    """Test format_debug_details function"""

    def test_format_string_details(self):
        """Test formatting string details"""
        details = ["Detail 1", "Detail 2", "Detail 3"]
        result = format_debug_details(details)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(item.startswith("  - ") for item in result)
        assert "Detail 1" in result[0]

    def test_format_with_custom_prefix(self):
        """Test formatting with custom prefix"""
        details = ["Item 1", "Item 2"]
        result = format_debug_details(details, prefix="* ")

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(item.startswith("* ") for item in result)

    def test_format_multiline_details(self):
        """Test formatting multiline details"""
        details = ["Line 1\nLine 2", "Single line"]
        result = format_debug_details(details)

        assert isinstance(result, list)
        assert len(result) == 3  # Should split multiline
        assert "  - Line 1" in result
        assert "  - Line 2" in result
        assert "  - Single line" in result

    def test_format_empty_details(self):
        """Test formatting empty details"""
        result = format_debug_details([])
        assert result == []

    def test_format_non_string_details(self):
        """Test formatting non-string details"""
        details = [123, {"key": "value"}, None]
        result = format_debug_details(details)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "  - 123" in result
        assert "  - {'key': 'value'}" in result
        assert "  - None" in result


class TestBuildContextFromSegments:
    """Test build_context_from_segments function"""

    def test_build_context_simple(self):
        """Test building context from simple segments"""
        segments = [
            {"content": "First content", "document_name": "doc1.pdf"},
            {"content": "Second content", "document_name": "doc2.pdf"},
        ]

        result = build_context_from_segments(segments)

        assert isinstance(result, str)
        assert "First content" in result
        assert "Second content" in result
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result

    def test_build_context_no_document_names(self):
        """Test building context without document names"""
        segments = [{"content": "Content 1"}, {"content": "Content 2"}]

        result = build_context_from_segments(segments)

        assert isinstance(result, str)
        assert "Content 1" in result
        assert "Content 2" in result

    def test_build_context_verbose(self):
        """Test building context with verbose mode"""
        segments = [{"content": "Test content", "document_name": "test.pdf"}]

        result = build_context_from_segments(segments, verbose=True)

        assert isinstance(result, str)
        assert "segments" in result
        assert "characters" in result
        assert "Test content" in result

    def test_build_context_empty_segments(self):
        """Test building context from empty segments"""
        result = build_context_from_segments([])
        assert result == ""

    def test_build_context_empty_content(self):
        """Test building context with empty content segments"""
        segments = [
            {"content": "", "document_name": "empty.pdf"},
            {"content": "Valid content", "document_name": "valid.pdf"},
        ]

        result = build_context_from_segments(segments)

        assert isinstance(result, str)
        assert "Valid content" in result
        # Empty content should be handled gracefully

    def test_build_context_missing_fields(self):
        """Test building context with missing fields"""
        segments = [{"content": "Content only"}, {"document_name": "Name only"}, {}]

        result = build_context_from_segments(segments)

        assert isinstance(result, str)
        # Should handle missing fields gracefully

    def test_utils_integration(self):
        """Test integration between utility functions"""
        # Test that utilities work together
        details = ["Step 1", "Step 2"]
        formatted = format_debug_details(details)

        segments = [{"content": "Test content", "document_name": "test.pdf"}]
        context = build_context_from_segments(segments)

        assert isinstance(formatted, list)
        assert isinstance(context, str)
        assert len(formatted) == 2
        assert "Test content" in context
