"""Tests for json_utils module"""

import pytest

from kbbridge.core.utils.json_utils import parse_json_from_markdown


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
