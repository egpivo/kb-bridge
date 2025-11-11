import os
from typing import Any, Dict, Optional

import kbbridge.integrations as integrations
from kbbridge.config.constants import (  # noqa: F401
    RetrieverDefaults,
    RetrieverSearchMethod,
)
from kbbridge.utils.formatting import format_search_results  # noqa: F401

try:
    import kbbridge.utils.working_components as working_components

    KnowledgeBaseRetriever = working_components.KnowledgeBaseRetriever  # noqa: F401
except ImportError:
    KnowledgeBaseRetriever = None  # noqa: F401

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
    resource_id: str,
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
    backend_type: Optional[str] = None,
    retrieval_endpoint: Optional[str] = None,
    retrieval_api_key: Optional[str] = None,
    opensearch_endpoint: Optional[str] = None,
    opensearch_auth: Optional[str] = None,
    n8n_webhook_url: Optional[str] = None,
    n8n_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve information from a knowledge base resource."""
    try:
        if not resource_id:
            return {"error": "resource_id is required"}
        if not query:
            return {"error": "query is required"}

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

        from kbbridge.integrations.backend_adapter import BackendAdapterFactory

        adapter = BackendAdapterFactory.create(
            resource_id=resource_id, credentials=credentials, backend_type=backend_type
        )

        resp = adapter.search(
            query=query,
            method=method,
            top_k=top_k,
            does_rerank=does_rerank,
            document_name=document_name,
            score_threshold=score_threshold,
            weights=weights,
        )

        if document_name and resp and isinstance(resp, dict):
            records = resp.get("records", [])
            if not records:
                retrieve_top_k = top_k * 3
                resp = adapter.search(
                    query=query,
                    method=method,
                    top_k=retrieve_top_k,
                    does_rerank=does_rerank,
                    document_name="",
                    score_threshold=score_threshold,
                    weights=weights,
                )

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
                            doc_name = (
                                doc.get("name", "") if isinstance(doc, dict) else ""
                            )
                            if doc_name == document_name:
                                filtered_records.append(record)
                        except Exception:
                            continue

                    resp = {**resp, "records": filtered_records[:top_k]}

        formatted = (
            format_search_results([resp]) if resp is not None else {"result": []}
        )
        return formatted

    except NotImplementedError as e:
        return {"error": str(e)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


def list_available_backends() -> Dict[str, Any]:
    """List available retriever backends."""
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
