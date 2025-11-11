from typing import Any, Dict, List, Optional

from kbbridge.integrations.backend_adapter import BackendAdapter
from kbbridge.integrations.credentials import RetrievalCredentials
from kbbridge.integrations.dify.dify_adapter import DifyAdapter
from kbbridge.integrations.dify.dify_credentials import DifyCredentials


class DifyBackendAdapter(BackendAdapter):
    """Dify-specific backend adapter implementation."""

    def __init__(self, resource_id: str, credentials: RetrievalCredentials):
        super().__init__(resource_id, credentials)

        dify_creds = DifyCredentials(
            endpoint=credentials.endpoint, api_key=credentials.api_key
        )

        self._dify_adapter = DifyAdapter(credentials=dify_creds)
        self._dify_retriever = self._dify_adapter.create_retriever(
            dataset_id=self._backend_id,
        )

    def search(
        self,
        query: str,
        method: str = "hybrid_search",
        top_k: int = 20,
        does_rerank: bool = False,
        document_name: str = "",
        score_threshold: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
        **options
    ) -> Dict[str, Any]:
        """Search using Dify backend."""
        metadata_filter = self.build_metadata_filter(document_name=document_name)

        raw_result = self._dify_retriever.call(
            query=query,
            method=method,
            top_k=top_k,
            does_rerank=does_rerank,
            metadata_filter=metadata_filter,
            score_threshold=score_threshold,
            weights=weights,
            **options
        )

        return raw_result

    def list_files(self, timeout: int = 30) -> List[str]:
        """List files using Dify backend."""
        return self._dify_retriever.list_files(
            dataset_id=self._backend_id, timeout=timeout
        )

    def build_metadata_filter(
        self, document_name: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Build Dify-specific metadata filter from generic parameters."""
        return self._dify_retriever.build_metadata_filter(document_name=document_name)
