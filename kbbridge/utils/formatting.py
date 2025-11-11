import logging

logger = logging.getLogger(__name__)


def format_search_results(results: list) -> dict:
    """Format search results according to the specified structure"""
    try:
        if not results:
            return {"result": []}

        # Handle case where results might be a dict instead of list
        if isinstance(results, dict):
            records = results.get("records", [])
        else:
            records = results[0].get("records", []) if results else []

        segments = []
        for record in records:
            try:
                # Handle case where record is None
                segment = record.get("segment") or {}
                if segment:
                    content = segment.get("content", "")
                    doc_metadata = segment.get("document", {}).get("doc_metadata", {})
                    if doc_metadata:
                        document_name = doc_metadata.get("document_name", "")
                    else:
                        document_name = ""
                    segments.append(
                        {"content": content, "document_name": document_name}
                    )
            except Exception as e:
                logger.debug(f"Skipping problematic record: {e}", exc_info=True)
                continue

        return {
            "result": segments,
        }
    except Exception as e:
        # Return error information
        return {"result": [], "format_error": str(e), "raw_results": results}


__all__ = ["format_search_results"]
