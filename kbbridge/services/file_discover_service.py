"""
File Discover Service

This service provides file discovery functionality using the integrations retriever
and the DSPy-based FileDiscover module.
"""

import logging
import os
from typing import Any, Dict, Optional

from kbbridge.core.discovery.file_reranker import rerank_files_by_names
from kbbridge.integrations import RetrievalCredentials, RetrieverRouter

logger = logging.getLogger(__name__)


def file_discover_service(
    query: str,
    dataset_id: str,
    top_k_recall: int = 100,
    top_k_return: int = 20,
    do_file_rerank: bool = True,
    relevance_score_threshold: float = 0.0,
    backend_type: Optional[str] = None,
    # Credentials
    retrieval_endpoint: Optional[str] = None,
    retrieval_api_key: Optional[str] = None,
    rerank_url: Optional[str] = None,
    rerank_model: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        # Create and validate generic credentials
        if retrieval_endpoint or retrieval_api_key:
            credentials = RetrievalCredentials(
                endpoint=retrieval_endpoint or "",
                api_key=retrieval_api_key or "",
                backend_type=backend_type or "dify",
            )
        else:
            credentials = RetrievalCredentials.from_env(backend_type=backend_type)

        valid, error = credentials.validate()
        if not valid:
            return {"error": error}

        # Get rerank settings from parameters or environment
        rerank_url = rerank_url or os.getenv("RERANK_URL")
        rerank_model = rerank_model or os.getenv("RERANK_MODEL")

        # Create retriever using RetrieverRouter
        retriever = RetrieverRouter.create_retriever(
            dataset_id=dataset_id,
            backend_type=credentials.backend_type,
            endpoint=credentials.endpoint,
            api_key=credentials.api_key,
            timeout=30,
        )

        def file_rerank_fn(query: str, documents, all_docs, **kw):
            names = [d.get("document_name", "") for d in all_docs]
            return rerank_files_by_names(
                query,
                file_names=names,
                relevance_score_threshold=kw.get(
                    "relevance_score_threshold", relevance_score_threshold
                ),
                rerank_url=kw.get("rerank_url", rerank_url),
                model=kw.get("model", rerank_model),
            )

        frf = (
            file_rerank_fn if (do_file_rerank and rerank_url and rerank_model) else None
        )

        # Access via package attribute so tests can patch
        # kbbridge.core.discovery.file_discover.FileDiscover without importing DSPy
        import kbbridge.core.discovery as discovery_pkg

        FileDiscover = getattr(discovery_pkg, "file_discover").FileDiscover

        discover = FileDiscover(retriever=retriever, file_rerank_fn=frf)
        metadata_filter = retriever.build_metadata_filter()

        # Call explicitly via __call__ so tests that patch __call__ directly work
        files = discover.__call__(
            query=query,
            search_method="semantic_search",
            top_k_recall=top_k_recall,
            top_k_return=top_k_return,
            do_chunk_rerank=False,
            do_file_rerank=bool(frf),
            metadata_filter=metadata_filter,
            rerank_url=rerank_url,
            rerank_model=rerank_model,
            relevance_score_threshold=relevance_score_threshold,
        )

        # Extract file names, filtering out empty ones
        file_names = [
            getattr(f, "file_name", "") for f in files if getattr(f, "file_name", "")
        ]

        if not file_names:
            logger.warning(
                f"file_discover returned {len(files)} files but none have file_name. "
                f"Files: {[f.__dict__ for f in files[:3]]}"
            )

        return {
            "success": True,
            "distinct_files": file_names,
            "total_files": len(file_names),
            "debug_info": {
                "files_found": len(files),
                "files_with_names": len(file_names),
                "sample_file_attrs": [
                    {
                        "file_name": getattr(f, "file_name", None),
                        "score": getattr(f, "score", None),
                        "has_chunks": bool(getattr(f, "chunks", [])),
                    }
                    for f in files[:5]
                ],
            }
            if not file_names
            else None,  # Only include debug info if empty
        }
    except Exception as e:
        return {"error": f"Error in file discover: {e}"}
