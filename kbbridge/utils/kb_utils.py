def format_debug_details(details: list, prefix: str = "  - ") -> list:
    """Format debug details with consistent indentation"""
    formatted = []
    for detail in details:
        if isinstance(detail, str):
            # Split multi-line details
            lines = detail.split("\n")
            for line in lines:
                formatted.append(f"{prefix}{line}")
        else:
            formatted.append(f"{prefix}{str(detail)}")
    return formatted


def build_context_from_segments(segments: list, verbose: bool = False) -> str:
    """Build context string from retrieved segments"""
    if not segments:
        return ""

    context_parts = []
    for i, segment in enumerate(segments):
        content = segment.get("content", "")
        document_name = segment.get("document_name", "")

        if content:
            if document_name:
                context_parts.append(f"--- Document: {document_name} ---\n{content}")
            else:
                context_parts.append(content)

    context = "\n\n".join(context_parts)

    if verbose:
        return f"Context built from {len(segments)} segments (total length: {len(context)} characters)\n\n{context}"

    return context
