"""
Retriever Service

This service provides retrieval functionality for knowledge bases.
Supports multiple backends via routing system.
"""

import os
from typing import Any, Dict, Optional

import kbbridge.integrations as integrations
import kbbridge.utils.working_components as working_components
from kbbridge.config.constants import (  # noqa: F401
    RetrieverDefaults,
    RetrieverSearchMethod,
)

# Re-exports for backward-compatible tests
from kbbridge.utils.formatting import format_search_results  # noqa: F401
from kbbridge.utils.working_components import (
    KnowledgeBaseRetriever as KnowledgeBaseRetriever,  # re-export for tests
)

# Backward-compatible default config expected by tests
DEFAULT_CONFIG: Dict[str, Any] = {
    "search_method": RetrieverDefaults.SEARCH_METHOD.value,
    "does_rerank": RetrieverDefaults.DOES_RERANK.value,
    "top_k": RetrieverDefaults.TOP_K.value,
    "score_threshold": RetrieverDefaults.SCORE_THRESHOLD.value,
    "weights": RetrieverDefaults.WEIGHTS.value,
    "document_name": "",
    "verbose": False,
}


def retriever_service(
    dataset_id: str,
    query: str,
    method: str = "hybrid_search",
    top_k: int = 20,
    verbose: bool = False,
    does_rerank: bool = False,
    score_threshold: Optional[float] = None,
    score_threshold_enabled: bool = False,
    weights: Optional[float] = None,
    document_name: str = "",
    timeout: int = 30,
    # Backend selection
    backend_type: Optional[str] = None,
    # Credentials (will be passed from environment or config)
    retrieval_endpoint: Optional[str] = None,
    retrieval_api_key: Optional[str] = None,
    opensearch_endpoint: Optional[str] = None,
    opensearch_auth: Optional[str] = None,
    n8n_webhook_url: Optional[str] = None,
    n8n_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve information from a knowledge base dataset.

    This tool retrieves relevant chunks from a knowledge base dataset using various
    search methods and optional reranking. Supports multiple backends via routing.

    Note: Reranking configuration (provider, model) is determined automatically
    by the backend adapter based on backend_type.

    Args:
        dataset_id: Dataset ID of the knowledge base to search
        query: Search query
        method: Search method (semantic_search, hybrid_search, keyword_search)
        top_k: Number of results to return
        does_rerank: Whether to enable reranking (adapter decides provider/model)
        score_threshold: Minimum score threshold
        score_threshold_enabled: Whether to enable score threshold
        weights: Search weights for hybrid search
        document_name: Filter by document name
        timeout: Timeout in seconds for the operation (default: 30)
        backend_type: Backend type ("dify", "opensearch", "n8n") - if None, uses RETRIEVER_BACKEND env var
        retrieval_endpoint: Dify API endpoint URL
        retrieval_api_key: Dify API key
        opensearch_endpoint: OpenSearch endpoint URL
        opensearch_auth: OpenSearch authentication
        n8n_webhook_url: n8n webhook URL
        n8n_api_key: n8n API key

    Returns:
        Dict containing the retrieval results with chunks and files
    """
    try:
        # Validate parameters
        if not dataset_id:
            return {"error": "dataset_id is required"}
        if not query:
            return {"error": "query is required"}

        # Create and validate generic credentials
        if retrieval_endpoint or retrieval_api_key:
            credentials = integrations.RetrievalCredentials(
                endpoint=retrieval_endpoint or "",
                api_key=retrieval_api_key or "",
                backend_type=backend_type or "dify",
            )
        elif opensearch_endpoint or opensearch_auth:
            credentials = integrations.RetrievalCredentials(
                endpoint=opensearch_endpoint or "",
                api_key=opensearch_auth or "",
                backend_type="opensearch",
            )
        elif n8n_webhook_url or n8n_api_key:
            credentials = integrations.RetrievalCredentials(
                endpoint=n8n_webhook_url or "",
                api_key=n8n_api_key or "",
                backend_type="n8n",
            )
        else:
            credentials = integrations.RetrievalCredentials.from_env(
                backend_type=backend_type
            )

        valid, error = credentials.validate()
        if not valid:
            return {"error": error}

        endpoint = credentials.endpoint
        api_key = credentials.api_key

        # TODO: Eventually replace with RetrieverRouter when tests are updated
        # Use module attribute so tests can patch it at
        # "kbbridge.utils.working_components.KnowledgeBaseRetriever"
        kb_retriever = working_components.KnowledgeBaseRetriever(endpoint, api_key)

        # Build optional metadata filter for document_name using retriever's method
        # This ensures correct format with comparison_operator and logical_operator
        metadata_filter = kb_retriever.build_metadata_filter(
            document_name=document_name
        )

        # Pass reranking config via **kwargs - adapter will use its own defaults
        resp = kb_retriever.retrieve(
            dataset_id=dataset_id,
            query=query,
            search_method=method,
            does_rerank=does_rerank,
            top_k=top_k,
            score_threshold_enabled=bool(score_threshold is not None),
            metadata_filter=metadata_filter,
            score_threshold=score_threshold,
            weights=weights,
        )

        # If metadata filter returned empty results and document_name was specified,
        # fall back to client-side filtering (metadata might be disabled in Dify)
        if document_name and resp and isinstance(resp, dict):
            records = resp.get("records", [])
            if not records and metadata_filter:
                # Metadata filter didn't work, try client-side filtering
                # Retrieve more results to filter from
                retrieve_top_k = top_k * 3
                resp = kb_retriever.retrieve(
                    dataset_id=dataset_id,
                    query=query,
                    search_method=method,
                    does_rerank=does_rerank,
                    top_k=retrieve_top_k,
                    score_threshold_enabled=bool(score_threshold is not None),
                    metadata_filter=None,  # Don't use metadata filter
                    score_threshold=score_threshold,
                    weights=weights,
                )

                # Filter by document_name client-side
                if resp and isinstance(resp, dict):
                    records = resp.get("records", [])
                    filtered_records = []
                    for record in records:
                        try:
                            segment = record.get("segment", {})
                            doc = (
                                segment.get("document", {})
                                if isinstance(segment, dict)
                                else {}
                            )
                            # Extract document name from doc.name (where Dify stores it)
                            doc_name = (
                                doc.get("name", "") if isinstance(doc, dict) else ""
                            )
                            # Match exact document name
                            if doc_name == document_name:
                                filtered_records.append(record)
                        except Exception:
                            continue

                    # Update response with filtered records
                    resp = {**resp, "records": filtered_records[:top_k]}
        # Format using working formatter expected by tests
        formatted = (
            format_search_results([resp]) if resp is not None else {"result": []}
        )
        return formatted

    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


def list_available_backends() -> Dict[str, Any]:
    """
    List available retriever backends.

    Returns:
        Dict containing available backends and current configuration
    """
    try:
        available_backends = integrations.RetrieverRouter.get_available_backends()
        current_backend = os.getenv("RETRIEVER_BACKEND", "dify")

        return {
            "available_backends": available_backends,
            "current_backend": current_backend,
            "environment_variables": {
                "RETRIEVER_BACKEND": os.getenv("RETRIEVER_BACKEND"),
                "RETRIEVAL_ENDPOINT": "***"
                if os.getenv("RETRIEVAL_ENDPOINT")
                else None,
                "RETRIEVAL_API_KEY": "***" if os.getenv("RETRIEVAL_API_KEY") else None,
                "OPENSEARCH_ENDPOINT": "***"
                if os.getenv("OPENSEARCH_ENDPOINT")
                else None,
                "OPENSEARCH_AUTH": "***" if os.getenv("OPENSEARCH_AUTH") else None,
                "N8N_WEBHOOK_URL": "***" if os.getenv("N8N_WEBHOOK_URL") else None,
                "N8N_API_KEY": "***" if os.getenv("N8N_API_KEY") else None,
            },
        }
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}
