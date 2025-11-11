import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests

from kbbridge.integrations.retriever_base import ChunkHit, FileHit, Retriever

from .constants import DifyRetrieverDefaults

logger = logging.getLogger(__name__)


class DifyRetriever(Retriever):
    """Dify backend implementation of Retriever interface.

    Example:
        retriever = DifyRetriever(
            endpoint="https://api.dify.ai/v1",
            api_key="your-key",
            dataset_id="dataset-id",
            timeout=30
        )

        resp = retriever.call(query="...", method="semantic_search", top_k=10)
        chunks = retriever.normalize_chunks(resp)
        files = retriever.group_files(chunks, agg="max")
    """

    def __init__(self, endpoint: str, api_key: str, dataset_id: str, timeout: int = 30):
        """Initialize Dify retriever."""
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.timeout = timeout

    def call(self, *, query: str, method: str, top_k: int, **kw) -> Dict[str, Any]:
        """
        Call Dify retrieval API.

        Args:
            query: Search query
            method: Search method (semantic_search, hybrid_search, etc.)
            top_k: Number of results
            **kw: Additional parameters

        Returns:
            Dify API response
        """
        model = {
            "search_method": method,
            "reranking_enable": kw.get("does_rerank", False),
            "reranking_model": {
                "reranking_provider_name": kw.get(
                    "reranking_provider_name",
                    DifyRetrieverDefaults.RERANKING_PROVIDER_NAME.value,
                ),
                "reranking_model_name": kw.get(
                    "reranking_model_name",
                    DifyRetrieverDefaults.RERANKING_MODEL_NAME.value,
                ),
            },
            "top_k": int(top_k) if top_k and top_k > 0 else 20,
            "score_threshold_enabled": kw.get("score_threshold_enabled", False),
        }

        # Add optional parameters
        if kw.get("score_threshold") is not None:
            model["score_threshold"] = kw["score_threshold"]
        if kw.get("weights") is not None:
            model["weights"] = kw["weights"]
        if kw.get("metadata_filter") is not None:
            # Ensure metadata is enabled when using metadata filters
            self.enable_metadata(timeout=self.timeout)
            model["metadata_filtering_conditions"] = kw["metadata_filter"]

        payload = {"query": query, "retrieval_model": model}
        url = f"{self.endpoint}/v1/datasets/{self.dataset_id}/retrieve"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"Calling Dify API: {method}, top_k={top_k}")
        logger.debug(f"Dify API URL: {url}")
        logger.debug(f"Dify API Payload: {payload}")

        response = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Log the error details
            error_detail = ""
            try:
                error_detail = response.json()
                logger.error(f"Dify API error response: {error_detail}")
            except:
                error_detail = response.text
                logger.error(f"Dify API error text: {error_detail}")
            raise

        return response.json()

    def normalize_chunks(self, resp: Dict[str, Any]) -> List[ChunkHit]:
        chunks = []

        try:
            records = resp.get("records", [])

            for record in records:
                try:
                    segment = record.get("segment") or {}
                    if not isinstance(segment, dict):
                        continue

                    content = segment.get("content", "")
                    if not content:
                        continue

                    # Extract metadata
                    doc = segment.get("document") or {}
                    if not isinstance(doc, dict):
                        doc = {}
                    doc_metadata = doc.get("doc_metadata") or {}
                    if not isinstance(doc_metadata, dict):
                        doc_metadata = {}
                    document_name = doc.get("name", "") or doc_metadata.get(
                        "document_name", ""
                    )
                    score = record.get("score", 1.0)

                    chunk = ChunkHit(
                        content=content,
                        document_name=document_name,
                        score=float(score),
                        metadata=doc_metadata,
                    )

                    chunks.append(chunk)

                except Exception as e:
                    logger.warning(f"Error normalizing chunk: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing Dify response: {e}")
            chunks = []

        logger.info(f"Normalized {len(chunks)} chunks")
        return chunks

    def group_files(self, chunks: List[ChunkHit], agg: str = "max") -> List[FileHit]:
        # Group by document name, skipping chunks with empty document_name
        by_file = defaultdict(list)
        skipped = 0
        for chunk in chunks:
            if not chunk.document_name or not chunk.document_name.strip():
                skipped += 1
                continue
            by_file[chunk.document_name].append(chunk)

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped} chunks with empty document_name out of {len(chunks)} total chunks"
            )

        # Aggregate scores
        file_hits = []
        for file_name, file_chunks in by_file.items():
            if agg == "max":
                score = max(chunk.score for chunk in file_chunks)
            elif agg == "mean":
                score = sum(chunk.score for chunk in file_chunks) / len(file_chunks)
            elif agg == "sum":
                score = sum(chunk.score for chunk in file_chunks)
            else:
                score = max(chunk.score for chunk in file_chunks)

            file_hits.append(
                FileHit(file_name=file_name, score=score, chunks=file_chunks)
            )

        # Sort by score descending
        file_hits.sort(key=lambda f: f.score, reverse=True)

        return file_hits

    def build_metadata_filter(self, *, document_name: str = "") -> Optional[dict]:
        """
        Build Dify metadata filter.

        Args:
            document_name: Filter by document name

        Returns:
            Metadata filter dict or None
        """
        conditions = []

        if document_name.strip():
            conditions.append(
                {
                    "name": "document_name",
                    "comparison_operator": "contains",
                    "value": document_name,
                }
            )

        return {"conditions": conditions} if conditions else None

    def list_files(self, *, dataset_id: str, timeout: int = 30) -> List[str]:
        """List document names in the dataset using Dify Documents API."""
        url = f"{self.endpoint}/v1/datasets/{self.dataset_id}/documents"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            files = [doc.get("name") for doc in data if doc.get("name")]
            return files
        except Exception as e:
            logger.warning(f"Dify list_files failed: {e}")
            return []

    def enable_metadata(self, timeout: int = 30, force: bool = False) -> bool:
        """
        Enable built-in metadata for the dataset.

        Checks current status first and only enables if metadata is disabled.
        Use force=True to enable even if already enabled.

        Args:
            timeout: Request timeout in seconds
            force: If True, enable even if already enabled (default: False)

        Returns:
            True if metadata was enabled successfully or already enabled, False otherwise
        """
        # Check current status first
        if not force:
            status = self.check_metadata_status(timeout=timeout)
            if status and status.get("enabled"):
                logger.info(
                    f"Metadata already enabled for dataset {self.dataset_id}, skipping enable"
                )
                return True

        # Enable metadata
        url = f"{self.endpoint}/v1/datasets/{self.dataset_id}/metadata/built-in/enable"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            logger.info(f"Metadata enabled successfully for dataset {self.dataset_id}")
            return True
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = resp.json()
                logger.error(f"Dify enable_metadata error response: {error_detail}")
            except:
                error_detail = resp.text
                logger.error(f"Dify enable_metadata error text: {error_detail}")
            return False
        except Exception as e:
            logger.warning(f"Dify enable_metadata failed: {e}")
            return False

    def check_metadata_status(self, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Check metadata status for the dataset.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Metadata status dict if available, None otherwise
        """
        url = f"{self.endpoint}/v1/datasets/{self.dataset_id}/metadata/built-in"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"Metadata status for dataset {self.dataset_id}: {data}")
            return data
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = resp.json()
                logger.debug(
                    f"Dify check_metadata_status error response: {error_detail}"
                )
            except:
                error_detail = resp.text
                logger.debug(f"Dify check_metadata_status error text: {error_detail}")
            return None
        except Exception as e:
            logger.warning(f"Dify check_metadata_status failed: {e}")
            return None


def make_retriever(kind: str, **kwargs) -> Retriever:
    """
    Factory function to create a retriever instance.

    TODO:
     - Future: Add other backends: e.g., OpenSearchRetriever (from .opensearch.opensearch_retriever)

    Example:
        retriever = make_retriever(
            "dify",
            endpoint="https://api.dify.ai/v1",
            api_key="key",
            dataset_id="dataset-id"
        )
    """
    kind = kind.lower()

    if kind in ("dify", "dify_retriever"):
        return DifyRetriever(
            endpoint=kwargs["endpoint"],
            api_key=kwargs["api_key"],
            dataset_id=kwargs["dataset_id"],
            timeout=kwargs.get("timeout", 30),
        )

    raise ValueError(f"Unknown retriever type: {kind}")
