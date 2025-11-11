import logging
import os
from typing import Any, Dict, Optional, Tuple

from kbbridge.config.constants import AssistantDefaults
from kbbridge.core.discovery.file_reranker import rerank_documents
from kbbridge.integrations import RetrieverRouter

from .models import Credentials, ProcessingConfig, WorkerDistribution

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Factory for creating service components"""

    @staticmethod
    def create_components(
        credentials: Credentials, dataset_id: str = ""
    ) -> Dict[str, Any]:
        """Create all required service components"""

        # Create retriever factory using routing system
        # This will automatically use the backend specified in RETRIEVER_BACKEND env var
        def retriever_factory(ds_id: str):
            """Factory function that creates retriever for a specific dataset"""
            if not ds_id:
                return None

            # Build configuration from credentials
            config = {
                "timeout": 30,
            }

            # Add backend-specific credentials based on environment
            backend_type = os.getenv("RETRIEVER_BACKEND", "dify").lower()

            if backend_type == "dify":
                config["endpoint"] = credentials.retrieval_endpoint
                config["api_key"] = credentials.retrieval_api_key
            elif backend_type == "opensearch":
                config["endpoint"] = os.getenv("OPENSEARCH_ENDPOINT")
                config["auth"] = os.getenv("OPENSEARCH_AUTH")
            elif backend_type == "n8n":
                config["webhook_url"] = os.getenv("N8N_WEBHOOK_URL")
                config["api_key"] = os.getenv("N8N_API_KEY")

            try:
                return RetrieverRouter.create_retriever(
                    dataset_id=ds_id, backend_type=backend_type, **config
                )
            except ValueError:
                # Fallback to Dify if specified backend is not available
                if backend_type != "dify":
                    return RetrieverRouter.create_retriever(
                        dataset_id=ds_id,
                        backend_type="dify",
                        endpoint=credentials.retrieval_endpoint,
                        api_key=credentials.retrieval_api_key,
                        timeout=30,
                    )
                raise

        def file_discover_factory(ds_id: str):
            # Local import to avoid importing optional dependencies at module import time
            from kbbridge.core.discovery.file_discover import FileDiscover

            r = retriever_factory(ds_id)

            def file_rerank_fn(query: str, documents, all_docs, **kw):
                # Use surrogate document texts for more accurate reranking
                return rerank_documents(
                    query=query,
                    documents=documents,
                    all_docs=all_docs,
                    relevance_score_threshold=kw.get("relevance_score_threshold", 0.0),
                    rerank_url=kw.get("rerank_url"),
                    model=kw.get("model"),
                )

            frf = None
            if getattr(credentials, "rerank_url", None) and getattr(
                credentials, "rerank_model", None
            ):
                frf = file_rerank_fn

            return FileDiscover(
                retriever=r,
                chunk_rerank_fn=None,
                file_rerank_fn=frf,
                top_chunks_per_file=2,
                use_surrogates=True,
            )

        # Local imports inside return to avoid import-time heavy deps (e.g., dspy)
        from kbbridge.core.query.intention_extractor import UserIntentionExtractor
        from kbbridge.core.synthesis.answer_extractor import OrganizationAnswerExtractor

        return {
            "retriever_factory": retriever_factory,
            "file_discover_factory": file_discover_factory,
            "answer_extractor": OrganizationAnswerExtractor(
                credentials.llm_api_url,
                credentials.llm_model,
                llm_api_token=credentials.llm_api_token,
                llm_temperature=credentials.llm_temperature,
                llm_timeout=credentials.llm_timeout,
                max_tokens=AssistantDefaults.LLM_MAX_TOKENS.value,
            ),
            "intention_extractor": UserIntentionExtractor(
                credentials.llm_api_url,
                credentials.llm_model,
                llm_api_token=credentials.llm_api_token,
                llm_temperature=credentials.llm_temperature,
                llm_timeout=credentials.llm_timeout,
                max_tokens=AssistantDefaults.LLM_MAX_TOKENS.value,
            ),
        }


class WorkerDistributor:
    """Handles worker distribution calculations"""

    @staticmethod
    def calculate_distribution(
        max_workers: int, dataset_count: int, avg_files_per_dataset: int = 5
    ) -> WorkerDistribution:
        """Calculate optimal worker distribution across different parallelism levels"""
        if dataset_count == 1:
            # Single dataset: focus workers on file processing
            return WorkerDistribution(
                dataset_workers=1,
                approach_workers=min(2, max_workers),
                file_workers=max_workers,
            )
        elif dataset_count <= max_workers:
            # Few datasets: one worker per dataset
            return WorkerDistribution(
                dataset_workers=dataset_count,
                approach_workers=min(2, max(1, max_workers // dataset_count)),
                file_workers=max(1, max_workers // dataset_count),
            )
        else:
            # Many datasets: balance between dataset and file processing
            dataset_workers = min(max_workers, dataset_count)
            remaining_workers = max(1, max_workers // dataset_workers)
            return WorkerDistribution(
                dataset_workers=dataset_workers,
                approach_workers=min(2, remaining_workers),
                file_workers=remaining_workers,
            )


class ParameterValidator:
    """Validates and sanitizes input parameters"""

    @staticmethod
    def validate_config(tool_parameters: Dict[str, Any]) -> ProcessingConfig:
        """Validate and create processing configuration"""
        # Get required parameters
        dataset_id = tool_parameters["dataset_id"]
        query = tool_parameters["query"]
        verbose = tool_parameters.get("verbose", False)

        # Sanitize query
        query = query.strip() if isinstance(query, str) else query

        # Get optional parameters with defaults
        score_threshold = ParameterValidator._validate_score_threshold(
            tool_parameters.get(
                "score_threshold", AssistantDefaults.SCORE_THRESHOLD.value
            )
        )
        top_k = ParameterValidator._validate_top_k(
            tool_parameters.get("top_k", AssistantDefaults.TOP_K.value)
        )
        max_workers = ParameterValidator._validate_max_workers(
            tool_parameters.get("max_workers", AssistantDefaults.MAX_WORKERS.value)
        )

        # Content booster parameters
        use_content_booster = tool_parameters.get("use_content_booster", True)
        max_boost_keywords = ParameterValidator._validate_max_boost_keywords(
            tool_parameters.get(
                "max_boost_keywords", AssistantDefaults.MAX_BOOST_KEYWORDS.value
            )
        )

        # Debug logging
        if verbose:
            logger.debug(
                f"use_content_booster: {use_content_booster}, max_boost_keywords: {max_boost_keywords}"
            )

        return ProcessingConfig(
            dataset_id=dataset_id,
            query=query,
            verbose=verbose,
            score_threshold=score_threshold,
            top_k=top_k,
            max_workers=max_workers,
            use_content_booster=use_content_booster,
            max_boost_keywords=max_boost_keywords,
        )

    @staticmethod
    def _validate_score_threshold(score_threshold: Any) -> Optional[float]:
        """Validate score threshold parameter"""
        if score_threshold is None:
            return None

        if isinstance(score_threshold, str) and score_threshold.strip() == "":
            return None

        try:
            threshold = float(score_threshold)
            if 0 <= threshold <= 1:
                return threshold
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid score_threshold value '{score_threshold}': {e}")

        return None

    @staticmethod
    def _validate_top_k(top_k: Any) -> int:
        """Validate top_k parameter"""
        if (
            top_k is None
            or top_k == ""
            or (isinstance(top_k, str) and top_k.strip() == "")
        ):
            return AssistantDefaults.TOP_K.value

        try:
            k = int(top_k)
            if k > 0:
                return k
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid top_k value '{top_k}': {e}")

        return AssistantDefaults.TOP_K.value

    @staticmethod
    def _validate_max_workers(max_workers: Any) -> int:
        """Validate max_workers parameter"""
        if (
            max_workers is None
            or max_workers == ""
            or (isinstance(max_workers, str) and max_workers.strip() == "")
        ):
            return AssistantDefaults.MAX_WORKERS.value

        try:
            workers = int(max_workers)
            if 1 <= workers <= 5:
                return workers
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid max_workers value '{max_workers}': {e}")

        return AssistantDefaults.MAX_WORKERS.value

    @staticmethod
    def _validate_max_boost_keywords(max_boost_keywords: Any) -> int:
        """Validate max_boost_keywords parameter"""
        if (
            max_boost_keywords is None
            or max_boost_keywords == ""
            or (
                isinstance(max_boost_keywords, str) and max_boost_keywords.strip() == ""
            )
        ):
            return 5  # default value

        try:
            keywords = int(max_boost_keywords)
            if 1 <= keywords <= 100:  # increased range to allow higher values like 50
                return keywords
        except (ValueError, TypeError) as e:
            logger.debug(
                f"Invalid max_boost_keywords value '{max_boost_keywords}': {e}"
            )

        return 5  # default value


class CredentialParser:
    """Parses and validates credentials"""

    @staticmethod
    def parse_credentials(
        runtime_credentials: Dict[str, Any]
    ) -> Tuple[Optional[Credentials], Optional[str]]:
        """Parse credentials from runtime, returns (credentials, error_message)"""
        retrieval_endpoint = runtime_credentials.get("retrieval_endpoint")
        retrieval_api_key = runtime_credentials.get("retrieval_api_key")
        llm_api_url = runtime_credentials.get("llm_api_url")
        llm_model = runtime_credentials.get("llm_model")

        # Check required fields
        missing = []
        if not retrieval_endpoint:
            missing.append("retrieval_endpoint")
        if not retrieval_api_key:
            missing.append("retrieval_api_key")
        if not llm_api_url:
            missing.append("llm_api_url")
        if not llm_model:
            missing.append("llm_model")

        if missing:
            return None, f"Missing required credentials: {', '.join(missing)}"

        # Parse optional timeout
        llm_timeout = None
        llm_timeout_raw = runtime_credentials.get("llm_timeout")
        if llm_timeout_raw is not None:
            try:
                llm_timeout = int(llm_timeout_raw)
                if llm_timeout <= 0:
                    llm_timeout = None
            except (ValueError, TypeError) as e:
                logger.debug(f"Invalid llm_timeout value '{llm_timeout_raw}': {e}")
                llm_timeout = None

        credentials = Credentials(
            retrieval_endpoint=retrieval_endpoint,
            retrieval_api_key=retrieval_api_key,
            llm_api_url=llm_api_url,
            llm_model=llm_model,
            llm_api_token=runtime_credentials.get("llm_api_token"),
            llm_temperature=AssistantDefaults.LLM_TEMPERATURE.value,
            llm_timeout=llm_timeout,
            rerank_url=runtime_credentials.get("rerank_url"),
            rerank_model=runtime_credentials.get("rerank_model"),
        )

        return credentials, None
