import json
import logging
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


def combine_rerank_results(
    rerank_response: Dict, all_docs: List[Dict], relevance_score_threshold: float
) -> List[Dict]:
    """Combine rerank results with original documents based on relevance score threshold"""
    combined = []
    results = rerank_response.get("results", [])
    for r in results:
        idx = r.get("index")
        score = r.get("relevance_score")

        # Skip invalid indices
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(all_docs):
            continue

        # Skip invalid scores
        if score is None or not isinstance(score, (int, float)):
            continue

        doc = all_docs[idx]
        if score >= relevance_score_threshold:
            combined.append({"index": idx, "relevance_score": score, "document": doc})
    return combined


def rerank_documents(
    query: str,
    documents: List[str],
    all_docs: List[Dict],
    relevance_score_threshold: float = 0.5,
    rerank_url: str = None,
    model: str = None,
) -> Dict[str, Any]:
    """
    Rerank documents using Jina reranker API

    Args:
        query: The search query
        documents: List of document texts to rerank
        all_docs: Original document objects
        relevance_score_threshold: Minimum relevance score to include
        rerank_url: URL of the reranking service (required)
        model: Model to use for reranking (required)

    Returns:
        Dict containing reranked results
    """
    # Validate required parameters
    if not rerank_url:
        raise ValueError(
            "rerank_url is required (should be provided by config/service layer)"
        )
    if not model:
        raise ValueError(
            "model is required (should be provided by config/service layer)"
        )

    # Ensure URL ends with /rerank if not already present
    url = rerank_url
    if not url.endswith("/rerank"):
        url = url.rstrip("/") + "/rerank"

    logger.debug(f"Rerank request: original_url={rerank_url}, normalized_url={url}")

    payload = {
        "query": query,
        "documents": documents,
        "return_documents": False,
        "model": model,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload, ensure_ascii=False)
        )
        logger.debug(
            f"Rerank response: status={response.status_code}, final_url={response.url}"
        )
        response.raise_for_status()
        rerank_response = response.json()

        # Combine rerank scores with original all_docs
        final_results = combine_rerank_results(
            rerank_response, all_docs, relevance_score_threshold
        )

        # Sort final_results by relevance_score (desc), then by index (asc) for stability
        final_results_sorted = sorted(
            final_results, key=lambda x: (-x["relevance_score"], x["index"])
        )

        reranked_all_docs = [
            item["document"]["document_name"] for item in final_results_sorted
        ]

        return {
            "success": True,
            "final_results": reranked_all_docs,
            "detailed_results": final_results_sorted,
            "rerank_response": rerank_response,
            "total_reranked": len(final_results_sorted),
        }
    except Exception as e:
        logger.error(f"Rerank request failed: {e}")
        logger.debug(f"Payload was: {json.dumps(payload, ensure_ascii=False)}")
        return {
            "success": False,
            "error": str(e),
            "final_results": [],
            "detailed_results": [],
            "total_reranked": 0,
        }


def rerank_files_by_names(
    query: str,
    file_names: List[str],
    relevance_score_threshold: float = 0.5,
    rerank_url: str = None,
    model: str = None,
) -> Dict[str, Any]:
    """
    Enhanced reranking function for file names with content-aware scoring

    Args:
        query: The search query
        file_names: List of file names to rerank
        relevance_score_threshold: Minimum relevance score to include
        rerank_url: URL of the reranking service (required)
        model: Model to use for reranking (required)

    Returns:
        Dict containing reranked file names
    """
    # Validate required parameters
    if not rerank_url:
        raise ValueError(
            "rerank_url is required (should be provided by config/service layer)"
        )
    if not model:
        raise ValueError(
            "model is required (should be provided by config/service layer)"
        )

    all_docs = [{"document_name": fname} for fname in file_names]
    documents = file_names  # Pass filenames as-is to the reranker

    return rerank_documents(
        query, documents, all_docs, relevance_score_threshold, rerank_url, model
    )
