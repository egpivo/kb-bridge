from typing import List

from kbbridge.integrations.retriever_base import ChunkHit


def build_file_surrogate_text(
    title: str, chunks: List[ChunkHit], max_chars: int = 800
) -> str:
    """Build a compact surrogate text for a file using its top chunks.

    This function creates a text representation of a file by combining its title
    and top-ranked chunks, useful for document reranking and similarity comparison.

    Args:
        title: File title/name
        chunks: Top chunks for this file
        max_chars: Max characters for the combined surrogate text

    Returns:
        Surrogate document text suitable for reranking

    Example:
        >>> chunks = [ChunkHit(content="First chunk", score=0.9),
        ...           ChunkHit(content="Second chunk", score=0.8)]
        >>> text = build_file_surrogate_text("document.pdf", chunks, max_chars=200)
        >>> print(text)
        title: document.pdf
        chunk: First chunk
        chunk: Second chunk
    """
    parts: List[str] = [f"title: {title}"]
    for c in chunks:
        # Use a small header per chunk to hint structure
        snippet = (c.content or "").strip().replace("\n", " ")
        if not snippet:
            continue
        parts.append(f"chunk: {snippet}")
        if sum(len(x) + 1 for x in parts) >= max_chars:
            break
    text = "\n".join(parts)
    return text[:max_chars]
