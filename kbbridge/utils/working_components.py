import json
import logging
from typing import Any, Dict, List, Optional

import requests

from kbbridge.config.constants import AssistantDefaults
from kbbridge.core.orchestration.models import Credentials, ProcessingConfig
from kbbridge.integrations.dify.constants import DifyRetrieverDefaults

logger = logging.getLogger(__name__)


def format_search_results(results: list) -> dict:
    """Format search results according to the specified structure"""
    try:
        if not results:
            return {"result": []}

        # Handle case where results might be a dict instead of list
        if isinstance(results, dict):
            records = results.get("records", [])
        else:
            records = results[0].get("records", []) if results else []

        segments = []
        for record in records:
            try:
                segment = record.get("segment")
                if segment:
                    content = segment.get("content", "")
                    doc_metadata = segment.get("document", {}).get("doc_metadata", {})
                    if doc_metadata:
                        document_name = doc_metadata.get("document_name", "")
                    else:
                        document_name = ""
                    segments.append(
                        {"content": content, "document_name": document_name}
                    )
            except Exception as e:
                logger.debug(f"Skipping problematic record: {e}", exc_info=True)
                continue

        return {
            "result": segments,
        }
    except Exception as e:
        # Return error information
        return {"result": [], "format_error": str(e), "raw_results": results}


class KnowledgeBaseRetriever:
    """
    Working knowledge base retrieval logic that calls the real Dify API
    """

    def __init__(self, endpoint: str, api_key: str):
        """
        Initialize the retriever

        Args:
            endpoint: Dify API endpoint
            api_key: Dify API key
        """
        self.endpoint = endpoint
        self.api_key = api_key

    def build_metadata_filter(self, *, document_name: str = "") -> Optional[dict]:
        """
        Build metadata filter for retrieval.

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

        if not conditions:
            return None

        return {"conditions": conditions, "logical_operator": "and"}

    def retrieve(
        self,
        dataset_id: str,
        query: str,
        search_method: str = "hybrid_search",
        does_rerank: bool = True,
        top_k: int = 10,
        reranking_provider_name: Optional[str] = None,
        reranking_model_name: Optional[str] = None,
        score_threshold_enabled: bool = False,
        metadata_filter: Optional[dict] = None,
        score_threshold: Optional[float] = None,
        weights: Optional[float] = None,
    ) -> dict:
        """
        Retrieve relevant documents from knowledge base using real Dify API

        Args:
            dataset_id: Target dataset ID
            query: Search query
            search_method: Search method (hybrid_search, semantic_search, etc.)
            does_rerank: Whether to rerank results
            top_k: Number of results to return
            reranking_provider_name: Reranking provider name (defaults to DifyRetrieverDefaults)
            reranking_model_name: Reranking model name (defaults to DifyRetrieverDefaults)
            score_threshold_enabled: Whether score threshold is enabled
            metadata_filter: Optional metadata filter
            score_threshold: Optional score threshold
            weights: Optional weights for hybrid search

        Returns:
            Dictionary containing retrieval results or error information
        """
        # Use DifyRetrieverDefaults if not provided
        if reranking_provider_name is None:
            reranking_provider_name = (
                DifyRetrieverDefaults.RERANKING_PROVIDER_NAME.value
            )
        if reranking_model_name is None:
            reranking_model_name = DifyRetrieverDefaults.RERANKING_MODEL_NAME.value

        # Disable reranking if provider or model is missing/invalid
        if does_rerank and (not reranking_provider_name or not reranking_model_name):
            does_rerank = False

        url = f"{self.endpoint.rstrip('/')}/v1/datasets/{dataset_id}/retrieve"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Validate top_k to ensure it's never null/None and always a valid positive integer
        try:
            top_k = int(top_k) if top_k is not None else 10
            if top_k <= 0:
                top_k = 10
        except (ValueError, TypeError):
            top_k = 10

        # Build request payload - Dify API expects nested structure under "retrieval_model"
        retrieval_model = {
            "search_method": search_method,
            "reranking_enable": does_rerank,
            "reranking_model": {
                "reranking_provider_name": reranking_provider_name,
                "reranking_model_name": reranking_model_name,
            },
            "top_k": top_k,
            "score_threshold_enabled": score_threshold_enabled,
        }

        # Add optional parameters
        if score_threshold is not None:
            retrieval_model["score_threshold"] = score_threshold
        if weights is not None:
            retrieval_model["weights"] = weights
        if metadata_filter is not None:
            retrieval_model["metadata_filtering_conditions"] = metadata_filter

        payload = {"query": query, "retrieval_model": retrieval_model}

        response = None
        try:
            print(f"DEBUG: Calling Dify API: {url}")
            print(f"DEBUG: Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=AssistantDefaults.DIFY_API_TIMEOUT.value,
            )
            response.raise_for_status()

            data = response.json()
            print(f"DEBUG: Dify API Response: {json.dumps(data, indent=2)}")

            # Return raw Dify response for compatibility with tests and integrations
            return data

        except requests.exceptions.HTTPError as e:
            # HTTP errors with status codes
            # Try to get status code from exception's response, or from outer scope response
            if hasattr(e, "response") and e.response:
                status_code = e.response.status_code
                reason = getattr(e.response, "reason", "Unknown Error")
                # Try to extract error content from response
                try:
                    error_content = e.response.json().get("error", str(e))
                except Exception as json_err:
                    # If JSON parsing fails, use response text
                    logger.debug(
                        f"Failed to parse error response as JSON: {json_err}",
                        exc_info=True,
                    )
                    error_content = getattr(e.response, "text", str(e))
            elif response:
                status_code = response.status_code
                reason = getattr(response, "reason", "Unknown Error")
                error_content = str(e)
            else:
                status_code = 500
                reason = "Unknown Error"
                error_content = str(e)

            return {
                "result": [],
                "error": True,
                "status_code": status_code,
                "reason": reason,
                "error_message": f"HTTP {status_code}: {str(e)}",
                "error_content": error_content,  # For backward compatibility
                "url": url,
                "debug_payload": {"payload": payload},
            }
        except requests.exceptions.Timeout as e:
            # Timeout errors
            return {
                "result": [],
                "error": True,
                "error_message": f"Request timed out: {str(e)}",
                "url": url,
                "debug_payload": {"payload": payload},
            }
        except requests.exceptions.ConnectionError as e:
            # Connection errors
            return {
                "result": [],
                "error": True,
                "error_message": f"Connection failed: {str(e)}",
                "url": url,
                "debug_payload": {"payload": payload},
            }
        except requests.exceptions.RequestException as e:
            # Other request exceptions
            return {
                "result": [],
                "error": True,
                "error_message": f"API request failed: {str(e)}",
                "url": url,
                "debug_payload": {"payload": payload},
            }
        except json.JSONDecodeError as e:
            # JSON parsing errors
            return {
                "result": [],
                "error": True,
                "error_message": f"Invalid JSON response: {str(e)}",
                "url": url,
                "debug_payload": {"payload": payload},
            }
        except Exception as e:
            # Catch-all for unexpected errors
            return {
                "result": [],
                "error": True,
                "error_message": f"Unexpected error: {str(e)}",
                "url": url,
                "debug_payload": {"payload": payload},
            }


class WorkingComponentFactory:
    """Factory for creating working service components"""

    @staticmethod
    def create_components(credentials: "Credentials") -> Dict[str, Any]:
        """Create all required service components using real API calls"""
        return {
            "retriever": KnowledgeBaseRetriever(
                credentials.retrieval_endpoint, credentials.retrieval_api_key
            ),
            "intention_extractor": WorkingIntentionExtractor(
                credentials.llm_api_url,
                credentials.llm_model,
                credentials.llm_api_token,
            ),
        }


class WorkingIntentionExtractor:
    """Working intention extractor that calls real LLM API"""

    def __init__(
        self, llm_api_url: str, llm_model: str, llm_api_token: Optional[str] = None
    ):
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.llm_api_token = llm_api_token

    def extract_intention(self, query: str) -> Dict[str, Any]:
        """Extract user intention using real LLM API"""
        try:
            headers = {
                "Content-Type": "application/json",
            }
            if self.llm_api_token:
                headers["Authorization"] = f"Bearer {self.llm_api_token}"

            payload = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts user intentions from queries. Return a JSON response with 'intention' and 'refined_query' fields.",
                    },
                    {
                        "role": "user",
                        "content": f"Extract the intention from this query: {query}",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 200,
            }

            response = requests.post(
                f"{self.llm_api_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Try to parse JSON response
            try:
                result = json.loads(content)
                return {
                    "success": True,
                    "intention": result.get(
                        "intention", f"User wants to find information about: {query}"
                    ),
                    "updated_query": result.get("refined_query", query),
                }
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return valid JSON
                return {
                    "success": True,
                    "intention": f"User wants to find information about: {query}",
                    "updated_query": query,
                }

        except Exception as e:
            # Fallback on error
            logger.debug(
                f"Intention extraction failed, using fallback: {e}", exc_info=True
            )
            return {
                "success": True,
                "intention": f"User wants to find information about: {query}",
                "updated_query": query,
            }


class WorkingDatasetProcessor:
    """Working dataset processor that uses real API calls"""

    def __init__(
        self,
        components: Dict[str, Any],
        config: ProcessingConfig,
        credentials: Credentials,
    ):
        self.components = components
        self.config = config
        self.credentials = credentials

    def process_datasets(
        self, dataset_pairs: List[Dict[str, str]], refined_query: str
    ) -> tuple:
        """Process datasets using real API calls"""
        all_candidates = []
        dataset_results = []

        for dataset_pair in dataset_pairs:
            dataset_id = dataset_pair["id"]

            # Use direct approach (simple search)
            direct_result = self._process_direct_approach(dataset_id, refined_query)
            if direct_result.get("candidates"):
                all_candidates.extend(direct_result["candidates"])

            # Use advanced approach (with reranking)
            advanced_result = self._process_advanced_approach(dataset_id, refined_query)
            if advanced_result.get("candidates"):
                all_candidates.extend(advanced_result["candidates"])

            dataset_results.append(
                {
                    "dataset_id": dataset_id,
                    "direct_results": direct_result,
                    "advanced_results": advanced_result,
                }
            )

        return dataset_results, all_candidates

    def _process_direct_approach(self, dataset_id: str, query: str) -> Dict[str, Any]:
        """Process using direct approach (simple search)"""
        retriever = self.components["retriever"]

        result = retriever.retrieve(
            dataset_id=dataset_id,
            query=query,
            search_method="hybrid_search",
            does_rerank=False,
            top_k=5,
        )

        candidates = []
        if result.get("result"):
            for item in result["result"]:
                candidates.append(
                    {
                        "content": item.get("content", ""),
                        "score": 1.0,  # Direct approach doesn't provide scores
                        "source": "dify_direct_search",
                        "metadata": {"document_name": item.get("document_name", "")},
                    }
                )

        return {"candidates": candidates, "total_found": len(candidates)}

    def _process_advanced_approach(self, dataset_id: str, query: str) -> Dict[str, Any]:
        """Process using advanced approach (with reranking)"""
        retriever = self.components["retriever"]

        # Check if reranking should be enabled based on credentials availability
        does_rerank = (
            self.credentials.is_reranking_available() if self.credentials else False
        )

        # Use DifyRetrieverDefaults for reranking parameters
        # These are used for Dify's built-in reranking feature
        result = retriever.retrieve(
            dataset_id=dataset_id,
            query=query,
            search_method="hybrid_search",
            does_rerank=does_rerank,
            top_k=5,
            reranking_provider_name=DifyRetrieverDefaults.RERANKING_PROVIDER_NAME.value,
            reranking_model_name=DifyRetrieverDefaults.RERANKING_MODEL_NAME.value,
        )

        candidates = []
        if result.get("result"):
            for item in result["result"]:
                candidates.append(
                    {
                        "content": item.get("content", ""),
                        "score": 0.8,  # Advanced approach with reranking
                        "source": "dify_advanced_search",
                        "metadata": {"document_name": item.get("document_name", "")},
                    }
                )

        return {"candidates": candidates, "total_found": len(candidates)}


class WorkingResultFormatter:
    """Working result formatter"""

    @staticmethod
    def format_final_answer(
        candidates: List[Dict[str, Any]], query: str, credentials: Credentials
    ) -> str:
        """Format final answer from candidates"""
        if not candidates:
            return "No relevant information found in the knowledge base."

        # Group by source
        direct_results = [
            c for c in candidates if c.get("source") == "dify_direct_search"
        ]
        advanced_results = [
            c for c in candidates if c.get("source") == "dify_advanced_search"
        ]

        answer_parts = []

        if direct_results:
            answer_parts.append("**Search Results:**")
            for i, result in enumerate(direct_results[:3], 1):
                content = result.get("content", "").strip()
                if content:
                    answer_parts.append(f"{i}. {content}")

        if advanced_results and len(answer_parts) < 3:
            answer_parts.append("\n**Additional Results:**")
            for i, result in enumerate(
                advanced_results[:2], 1
            ):  # Top 2 additional results
                content = result.get("content", "").strip()
                if content and content not in [
                    r.get("content", "") for r in direct_results
                ]:
                    answer_parts.append(f"{i}. {content}")

        return (
            "\n".join(answer_parts)
            if answer_parts
            else "No relevant information found."
        )
