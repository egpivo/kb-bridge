import logging

from kbbridge.utils.formatting import format_search_results


class TestFormatSearchResults:
    def test_handles_empty_results(self, caplog):
        """Should return empty list when no records provided."""
        with caplog.at_level(logging.WARNING):
            result = format_search_results([])

        assert result == {"result": []}
        assert any("No results provided" in msg for msg in caplog.messages)

    def test_formats_dict_payload(self, caplog):
        """Accept dict payloads with nested segment metadata."""
        payload = {
            "records": [
                {
                    "segment": {
                        "content": "Clause 1",
                        "document": {
                            "name": "doc-a.pdf",
                            "doc_metadata": {"document_name": "DocA"},
                        },
                    }
                },
                {"segment": None},
            ]
        }

        result = format_search_results(payload)

        assert result["result"][0]["content"] == "Clause 1"
        assert result["result"][0]["document_name"] == "doc-a.pdf"

    def test_handles_unexpected_structure(self):
        """Returns diagnostic payload when structure raises errors."""

        class Opaque:
            pass

        malformed = Opaque()
        output = format_search_results(malformed)  # type: ignore[arg-type]

        assert output["result"] == []
        assert "format_error" in output
        assert output["raw_results"] is malformed
