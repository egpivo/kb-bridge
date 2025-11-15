from typing import Any, Dict, Optional

from kbbridge.integrations import BackendAdapterFactory, RetrievalCredentials


def file_lister_service(
    resource_id: str,
    timeout: int = 30,
    limit: Optional[int] = None,
    offset: int = 0,
    backend_type: Optional[str] = None,
    retrieval_endpoint: Optional[str] = None,
    retrieval_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """List files in a knowledge base resource with pagination support."""
    try:
        if not resource_id:
            return {"error": "resource_id is required"}

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

        adapter = BackendAdapterFactory.create(
            resource_id=resource_id, credentials=credentials, backend_type=backend_type
        )

        files = adapter.list_files(timeout=timeout)

        # Apply pagination
        total_files = len(files)
        if offset > 0:
            files = files[offset:]
        if limit is not None:
            files = files[:limit]

        return {
            "files": files,
            "total": total_files,
            "limit": limit,
            "offset": offset,
            "returned": len(files),
        }

    except NotImplementedError as e:
        return {"error": str(e)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}
