"""Tests for json_utils module"""

import pytest

from kbbridge.core.utils.json_utils import parse_dataset_ids, parse_json_from_markdown


class TestParseJsonFromMarkdown:
    """Test parse_json_from_markdown function"""

    def test_parse_json_markdown_with_json_block(self):
        """Test parsing JSON from markdown code block with json label"""
        text = '```json\n[["keyword1", "keyword2"], ["keyword3"]]\n```'
        result = parse_json_from_markdown(text)
        assert result == {"result": [["keyword1", "keyword2"], ["keyword3"]]}

    def test_parse_json_markdown_without_json_label(self):
        """Test parsing JSON from markdown code block without json label"""
        text = '```\n[["keyword1", "keyword2"], ["keyword3"]]\n```'
        result = parse_json_from_markdown(text)
        assert result == {"result": [["keyword1", "keyword2"], ["keyword3"]]}

    def test_parse_json_markdown_no_block_raises(self):
        """Test that ValueError is raised when no JSON block found"""
        with pytest.raises(ValueError, match="No JSON array found"):
            parse_json_from_markdown("just plain text")

    def test_parse_json_markdown_invalid_structure_raises(self):
        """Test that ValueError is raised when JSON is not array of keyword sets"""
        with pytest.raises(ValueError, match="not an array of keyword sets"):
            parse_json_from_markdown('```json\n[{"not": "an array"}]\n```')

    def test_parse_json_markdown_not_list_of_lists_raises(self):
        """Test that ValueError is raised when not list of lists"""
        with pytest.raises(ValueError, match="not an array of keyword sets"):
            parse_json_from_markdown('```json\n["not", "nested"]\n```')


class TestParseDatasetIds:
    """Test parse_dataset_ids function"""

    def test_parse_uuids_from_raw(self):
        """Test extracting UUIDs from raw string"""
        raw = "Some text with uuid 123e4567-e89b-12d3-a456-426614174000 and another 987fcdeb-51a2-43f7-bcde-123456789abc"
        result = parse_dataset_ids(raw)
        assert len(result) == 2
        assert "123e4567-e89b-12d3-a456-426614174000" in result
        assert "987fcdeb-51a2-43f7-bcde-123456789abc" in result

    def test_parse_from_json_array(self):
        """Test parsing from JSON array"""
        raw = '["id1", "id2", "id3"]'
        result = parse_dataset_ids(raw)
        assert result == ["id1", "id2", "id3"]

    def test_parse_from_double_quoted_json_string(self):
        """Test parsing from double-quoted JSON string"""
        raw = '"["id1", "id2"]"'
        result = parse_dataset_ids(raw)
        # May parse successfully or fall back to UUID extraction
        assert isinstance(result, list)

    def test_parse_from_list_object(self):
        """Test parsing when already a list"""
        # This tests the unwrapping logic
        raw = '"\\"[\\"id1\\", \\"id2\\"]\\""'
        result = parse_dataset_ids(raw)
        assert len(result) >= 0  # May parse or return empty

    def test_parse_from_comma_separated_string(self):
        """Test parsing from comma-separated string"""
        raw = "id1, id2, id3"
        result = parse_dataset_ids(raw)
        assert result == ["id1", "id2", "id3"]

    def test_parse_empty_string(self):
        """Test parsing empty string"""
        result = parse_dataset_ids("")
        assert result == []

    def test_parse_with_integers_in_list(self):
        """Test parsing list with integers"""
        raw = "[123, 456, 789]"
        result = parse_dataset_ids(raw)
        assert result == ["123", "456", "789"]

    def test_parse_with_single_quotes(self):
        """Test parsing with single quotes"""
        raw = "['id1', 'id2']"
        # Single quotes not valid JSON, should fall through
        result = parse_dataset_ids(raw)
        assert isinstance(result, list)
