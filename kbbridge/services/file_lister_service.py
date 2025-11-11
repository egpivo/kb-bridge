from typing import Any, Dict, Optional

from kbbridge.integrations import RetrievalCredentials


def file_lister_service(
    dataset_id: str,
    timeout: int = 30,
    backend_type: Optional[str] = None,
    retrieval_endpoint: Optional[str] = None,
    retrieval_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List files in a knowledge base dataset.

    This tool lists all files in a knowledge base dataset.

    Args:
        dataset_id: Dataset ID of the knowledge base to list files from
        timeout: Timeout in seconds for the operation (default: 30)
        backend_type: Backend type ("dify", "opensearch", etc.) - if None, uses RETRIEVAL_BACKEND env var
        retrieval_endpoint: Retrieval backend endpoint URL
        retrieval_api_key: Retrieval backend API key

    Returns:
        Dict containing the list of files and their metadata
    """
    try:
        # Validate parameters
        if not dataset_id:
            return {"error": "dataset_id is required"}

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

        # For Dify backend, use DifyAdapter for file listing
        if credentials.backend_type == "dify":
            from kbbridge.integrations import DifyAdapter, DifyCredentials

            dify_creds = DifyCredentials(
                endpoint=credentials.endpoint, api_key=credentials.api_key
            )
            adapter = DifyAdapter(credentials=dify_creds)
            files = adapter.list_files(dataset_id=dataset_id, timeout=timeout)
            return {"files": files}
        else:
            return {
                "error": f"File listing not supported for backend: {credentials.backend_type}"
            }

    except ValueError as e:
        # Credential validation errors
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}
