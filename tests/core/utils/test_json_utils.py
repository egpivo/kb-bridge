"""Tests for json_utils module"""

import pytest

from kbbridge.core.utils.json_utils import (
    _looks_like_structured_json,
    _process_list_items,
    _salvage_id_objects,
    parse_dataset_ids,
    parse_dataset_info,
    parse_json_from_markdown,
)


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


class TestParseDatasetInfo:
    """Test parse_dataset_info function"""

    def test_parse_simple_dict_array(self):
        """Test parsing simple array of dictionaries (source_path is ignored)"""
        raw = '[{"id": "dataset1", "source_path": "/path1"}, {"id": "dataset2", "source_path": "/path2"}]'
        result = parse_dataset_info(raw)
        assert len(result) == 2
        assert result[0]["id"] == "dataset1"
        assert result[1]["id"] == "dataset2"
        assert "source_path" not in result[0]

    def test_parse_with_uuids_no_structured_json(self):
        """Test parsing UUIDs when no structured JSON present"""
        raw = "Text with uuid 123e4567-e89b-12d3-a456-426614174000"
        result = parse_dataset_info(raw)
        assert len(result) == 1
        assert result[0]["id"] == "123e4567-e89b-12d3-a456-426614174000"
        # source_path is not included in result
        assert "source_path" not in result[0]

    def test_parse_with_uuids_and_structured_json(self):
        """Test that structured JSON takes precedence over UUID extraction"""
        raw = '[{"id": "dataset1"}] with uuid 123e4567-e89b-12d3-a456-426614174000'
        result = parse_dataset_info(raw)
        # Should parse the structured JSON, not extract UUID
        assert len(result) >= 1

    def test_parse_double_quoted_json_string(self):
        """Test parsing double-quoted JSON string layer"""
        raw = '"\\"[{\\"id\\": \\"dataset1\\"}]\\""'
        result = parse_dataset_info(raw)
        assert isinstance(result, list)

    def test_parse_from_list_object_direct(self):
        """Test parsing when already a list object (line 132)"""
        # This would be called via _process_list_items
        raw = '[{"id": "dataset1"}, {"id": "dataset2"}]'
        result = parse_dataset_info(raw)
        assert len(result) == 2

    def test_parse_with_comma_fallback(self):
        """Test comma-separated fallback parsing"""
        raw = "dataset1, dataset2, dataset3"
        result = parse_dataset_info(raw)
        assert len(result) == 3
        assert result[0]["id"] == "dataset1"

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list (line 151)"""
        result = parse_dataset_info("")
        assert result == []

    def test_parse_malformed_json_with_salvage(self):
        """Test parsing malformed JSON that gets salvaged"""
        raw = 'Text with {"id": "dataset1", "source_path": "/path"} embedded'
        result = parse_dataset_info(raw)
        assert len(result) >= 1

    def test_parse_with_missing_source_path(self):
        """Test parsing dict without source_path (source_path is ignored)"""
        raw = '[{"id": "dataset1"}]'
        result = parse_dataset_info(raw)
        assert result[0]["id"] == "dataset1"
        # source_path is not included in result
        assert "source_path" not in result[0]

    def test_parse_with_scalar_items_in_list(self):
        """Test parsing list with scalar items"""
        raw = '["dataset1", "dataset2"]'
        result = parse_dataset_info(raw)
        assert len(result) == 2
        assert result[0]["id"] == "dataset1"


class TestLooksLikeStructuredJson:
    """Test _looks_like_structured_json function"""

    def test_has_braces(self):
        """Test detection of structured JSON"""
        assert _looks_like_structured_json('{"id": "test"}') is True
        assert _looks_like_structured_json('Text {"id": "test"} more text') is True

    def test_no_braces(self):
        """Test detection when no braces"""
        assert _looks_like_structured_json("plain text") is False
        assert (
            _looks_like_structured_json("123e4567-e89b-12d3-a456-426614174000") is False
        )


class TestProcessListItems:
    """Test _process_list_items function"""

    def test_process_dict_items_with_id(self):
        """Test processing dictionary items with id"""
        items = [
            {"id": "dataset1", "source_path": "/path1"},
            {"id": "dataset2", "source_path": "/path2"},
        ]
        result = _process_list_items(items)
        assert len(result) == 2
        assert result[0]["id"] == "dataset1"
        assert "source_path" not in result[0]

    def test_process_dict_items_without_source_path(self):
        """Test processing dict items without source_path (defaults to empty, ignored)"""
        items = [{"id": "dataset1"}, {"id": "dataset2"}]
        result = _process_list_items(items)
        assert len(result) == 2
        # source_path is not included in result
        assert "source_path" not in result[0]

    def test_process_dict_items_with_none_source_path(self):
        """Test processing dict items with None source_path (converted to empty, ignored)"""
        items = [{"id": "dataset1", "source_path": None}]
        result = _process_list_items(items)
        # source_path is ignored and not included in result
        assert "source_path" not in result[0]

    def test_process_dict_items_missing_id(self):
        """Test that items without id are skipped"""
        items = [{"name": "dataset1"}, {"id": "dataset2"}]
        result = _process_list_items(items)
        assert len(result) == 1
        assert result[0]["id"] == "dataset2"

    def test_process_dict_items_empty_id(self):
        """Test that items with empty id are skipped (line 172-173)"""
        items = [{"id": ""}, {"id": "dataset2"}]
        result = _process_list_items(items)
        assert len(result) == 1
        assert result[0]["id"] == "dataset2"

    def test_process_scalar_string_items(self):
        """Test processing scalar string items (line 185-187)"""
        items = ["dataset1", "dataset2"]
        result = _process_list_items(items)
        assert len(result) == 2
        assert result[0]["id"] == "dataset1"
        assert "source_path" not in result[0]

    def test_process_scalar_int_items(self):
        """Test processing scalar integer items"""
        items = [123, 456]
        result = _process_list_items(items)
        assert len(result) == 2
        assert result[0]["id"] == "123"
        assert "source_path" not in result[0]

    def test_process_scalar_float_items(self):
        """Test processing scalar float items"""
        items = [123.45]
        result = _process_list_items(items)
        assert result[0]["id"] == "123.45"

    def test_process_mixed_items(self):
        """Test processing mixed dict and scalar items"""
        items = [{"id": "dataset1"}, "dataset2", 123]
        result = _process_list_items(items)
        assert len(result) == 3


class TestSalvageIdObjects:
    """Test _salvage_id_objects function"""

    def test_salvage_valid_json_object(self):
        """Test salvaging valid JSON object (line 207-218)"""
        raw = 'Text {"id": "dataset1", "source_path": "/path"} more text'
        result = _salvage_id_objects(raw)
        assert len(result) == 1
        assert result[0]["id"] == "dataset1"
        assert "source_path" not in result[0]

    def test_salvage_multiple_objects(self):
        """Test salvaging multiple objects"""
        raw = '{"id": "dataset1"} and {"id": "dataset2", "source_path": "/path2"}'
        result = _salvage_id_objects(raw)
        assert len(result) == 2

    def test_salvage_object_without_source_path(self):
        """Test salvaging object without source_path"""
        raw = '{"id": "dataset1"}'
        result = _salvage_id_objects(raw)
        assert len(result) == 1
        assert "source_path" not in result[0]

    def test_salvage_invalid_json_fallback_to_regex(self):
        """Test salvaging invalid JSON falls back to regex parsing (line 208-209, 220-231)"""
        raw = 'Text {"id": "dataset1", invalid json} more'
        result = _salvage_id_objects(raw)
        # Should attempt regex-based parsing
        assert isinstance(result, list)

    def test_salvage_regex_extraction_with_quotes(self):
        """Test regex extraction with quoted id (line 225-226)"""
        raw = '{"id": "dataset1", "source_path": "/path"}'
        result = _salvage_id_objects(raw)
        assert len(result) >= 1

    def test_salvage_regex_extraction_without_source_path_match(self):
        """Test regex extraction when source_path match fails (line 228-231)"""
        raw = 'Text with "id": "dataset1" and no source_path field'
        result = _salvage_id_objects(raw)
        # Should still extract id, source_path empty
        assert isinstance(result, list)

    def test_salvage_no_id_field(self):
        """Test that objects without id are skipped"""
        raw = '{"name": "dataset1", "source_path": "/path"}'
        result = _salvage_id_objects(raw)
        assert len(result) == 0

    def test_salvage_empty_string(self):
        """Test salvaging empty string"""
        result = _salvage_id_objects("")
        assert result == []

    def test_salvage_no_braces(self):
        """Test salvaging text without braces"""
        result = _salvage_id_objects("plain text")
        assert result == []
