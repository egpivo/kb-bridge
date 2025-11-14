import logging

logger = logging.getLogger(__name__)


def format_search_results(results: list) -> dict:
    """Format search results according to the specified structure"""
    try:
        if not results:
            logger.warning("format_search_results: No results provided")
            return {"result": []}

        # Handle case where results might be a dict instead of list
        if isinstance(results, dict):
            records = results.get("records", [])
        else:
            records = results[0].get("records", []) if results else []

        logger.warning(f"format_search_results: Processing {len(records)} records")

        segments = []
        skipped_count = 0
        for i, record in enumerate(records):
            try:
                # Handle case where record is None
                segment = record.get("segment") or {}
                if segment:
                    content = segment.get("content", "")
                    doc = segment.get("document", {}) or {}
                    doc_metadata = doc.get("doc_metadata", {}) or {}

                    # Try multiple ways to get document_name (matching DifyRetriever.normalize_chunks)
                    document_name = (
                        doc.get("name", "")
                        or doc_metadata.get("document_name", "")
                        or ""
                    )

                    if i < 3:  # Log first 3 for debugging
                        logger.warning(
                            f"format_search_results: Record {i+1} - "
                            f"doc.name='{doc.get('name', '')}', "
                            f"doc_metadata.document_name='{doc_metadata.get('document_name', '')}', "
                            f"final document_name='{document_name}'"
                        )

                    segments.append(
                        {"content": content, "document_name": document_name}
                    )
                else:
                    skipped_count += 1
                    if i < 3:
                        logger.warning(
                            f"format_search_results: Record {i+1} has no segment"
                        )
            except Exception as e:
                skipped_count += 1
                logger.warning(
                    f"format_search_results: Skipping problematic record {i+1}: {e}",
                    exc_info=True,
                )
                continue

        logger.warning(
            f"format_search_results: Formatted {len(segments)} segments, skipped {skipped_count}"
        )
        return {
            "result": segments,
        }
    except Exception as e:
        # Return error information
        logger.error(f"format_search_results: Exception occurred: {e}", exc_info=True)
        return {"result": [], "format_error": str(e), "raw_results": results}


__all__ = ["format_search_results"]
