import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from kbbridge.config.constants import (
    AssistantDefaults,
    RetrieverDefaults,
    RetrieverSearchMethod,
)
from kbbridge.core.query.constants import KeywordGeneratorDefaults
from kbbridge.core.query.keyword_generator import ContentBoostKeywordGenerator
from kbbridge.core.synthesis.answer_extractor import OrganizationAnswerExtractor
from kbbridge.core.synthesis.constants import ResponseMessages
from kbbridge.core.utils.profiling_utils import profile_stage
from kbbridge.integrations import Retriever
from kbbridge.utils.formatting import format_search_results
from kbbridge.utils.kb_utils import build_context_from_segments, format_debug_details

from .models import Credentials, DatasetResult, ProcessingConfig
from .services import WorkerDistributor

logger = logging.getLogger(__name__)


class _LegacyFileSearcherAdapter:
    """
    Adapter to wrap legacy FileSearcher interface.

    Translates resource_id to dataset_id for backward compatibility with old interfaces.
    This isolates backend-specific parameter naming (dataset_id) to the adapter layer.
    """

    def __init__(self, legacy_searcher):
        """Initialize adapter with legacy FileSearcher instance."""
        self._legacy_searcher = legacy_searcher

    def search_files(self, query: str, resource_id: str, **kwargs):
        """
        Search files using legacy interface.

        Translates resource_id to dataset_id for the underlying legacy searcher.
        """
        # Translate generic resource_id to backend-specific dataset_id
        return self._legacy_searcher.search_files(
            query=query, dataset_id=resource_id, **kwargs
        )


class FileSearchStrategy:
    """Handles file search operations"""

    def __init__(
        self,
        discover_factory_or_searcher,
        credentials: Credentials = None,
        verbose: bool = False,
    ):
        # Backward-compat: support old signature with FileSearcher instance
        if hasattr(discover_factory_or_searcher, "search_files"):
            # Wrap old FileSearcher interface with adapter to handle resource_id -> dataset_id translation
            self._compat_file_searcher = _LegacyFileSearcherAdapter(
                discover_factory_or_searcher
            )
            self.file_searcher = discover_factory_or_searcher
            self.discover_factory = None
            self.credentials = credentials or Credentials(
                "", "", "", ""
            )  # minimal placeholder
        else:
            self.discover_factory = discover_factory_or_searcher
            self.credentials = credentials
        self.verbose = verbose

    def parallel_search(
        self,
        query: str,
        resource_id: str,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute parallel file search for a resource"""
        search_profiling = {}
        debug_info = []

        with profile_stage(
            f"resource_{resource_id}.standalone_file_search",
            search_profiling,
            self.verbose,
        ):
            try:
                start_time = time.perf_counter()

                if getattr(self, "_compat_file_searcher", None) is not None:
                    # Adapter handles resource_id -> dataset_id translation
                    search_result = self._compat_file_searcher.search_files(
                        query=query,
                        resource_id=resource_id,
                        max_keywords=AssistantDefaults.MAX_KEYWORDS.value,
                        top_k_per_keyword=AssistantDefaults.TOP_K_PER_KEYWORD.value,
                        max_workers=max_workers or AssistantDefaults.MAX_WORKERS.value,
                        verbose=self.verbose,
                    )
                else:
                    discover = self.discover_factory(resource_id)
                    metadata_filter = discover.retriever.build_metadata_filter()
                    files = discover(
                        query=query,
                        search_method=RetrieverSearchMethod.SEMANTIC_SEARCH.value,
                        top_k_recall=AssistantDefaults.TOP_K_PER_KEYWORD.value,
                        top_k_return=AssistantDefaults.MAX_FILES.value,
                        do_chunk_rerank=False,
                        do_file_rerank=(
                            self.credentials.is_reranking_available()
                            if self.credentials
                            else False
                        ),
                        metadata_filter=metadata_filter,
                        rerank_url=self.credentials.rerank_url
                        if self.credentials
                        else None,
                        rerank_model=self.credentials.rerank_model
                        if self.credentials
                        else None,
                        relevance_score_threshold=AssistantDefaults.RELEVANCE_SCORE_THRESHOLD.value,
                    )

                    # Log top-3 discovered files for quick verification
                    if files:
                        topn = min(3, len(files))
                        logger.info("Top discovered files (first %d):", topn)
                        for i, f in enumerate(files[:topn], 1):
                            name = getattr(f, "file_name", "") or getattr(
                                f, "title", ""
                            )
                            try:
                                name = unquote(name)
                            except Exception as e:
                                logger.debug(
                                    f"Failed to unquote filename '{name}': {e}"
                                )
                            score = getattr(f, "score", None)
                            if score is not None:
                                logger.info(
                                    "  %d) %s (score=%.4f)", i, name, float(score)
                                )
                            else:
                                logger.info("  %d) %s", i, name)

                    distinct = [getattr(f, "file_name", "") for f in files]
                    search_result = {
                        "success": True,
                        "results": [],
                        "distinct_files": distinct,
                        "steps": [],
                    }

                end_time = time.perf_counter()
                search_duration = end_time - start_time

                return self._format_search_result(
                    search_result, search_duration, debug_info, search_profiling
                )

            except Exception as e:
                return self._format_search_error(str(e), debug_info, search_profiling)

    def _format_search_result(
        self,
        search_result: Dict[str, Any],
        duration: float,
        debug_info: List[str],
        profiling: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format search results into standardized response"""
        if search_result.get("success"):
            file_names = search_result.get("distinct_files", [])

            debug_msg = (
                f"Found {len(file_names)} files in {round(duration * 1000, 1)}ms"
            )
            if "keywords_used" in search_result:
                debug_msg += (
                    f" using keywords: {', '.join(search_result['keywords_used'])}"
                )
            debug_info.append(debug_msg)

            return {
                "success": True,
                "file_names": file_names,
                "search_duration_ms": round(duration * 1000, 1),
                "keywords_used": search_result.get("keywords_used", []),
                "debug_info": debug_info,
                "profiling": profiling if self.verbose else {},
                "search_result_details": search_result,
            }
        else:
            error_msg = f"Search failed in {round(duration * 1000, 1)}ms: {search_result.get('message', 'Unknown error')}"
            debug_info.append(error_msg)

            return {
                "success": False,
                "file_names": [],
                "error": error_msg,
                "search_duration_ms": round(duration * 1000, 1),
                "debug_info": debug_info,
                "profiling": profiling if self.verbose else {},
                "search_result_details": search_result,
            }

    def _format_search_error(
        self, error: str, debug_info: List[str], profiling: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format search error into standardized response"""
        error_msg = f"File search error: {error}"
        if self.verbose:
            debug_info.append(error_msg)

        return {
            "success": False,
            "file_names": [],
            "error": error_msg,
            "debug_info": debug_info,
            "profiling": profiling if self.verbose else {},
        }


class DirectApproachProcessor:
    """Processes queries using the direct approach"""

    def __init__(
        self,
        retriever: Retriever,
        answer_extractor: OrganizationAnswerExtractor,
        verbose: bool = False,
        custom_instructions: Optional[str] = None,
        credentials: Optional[Credentials] = None,
    ):
        self.retriever = retriever
        self.answer_extractor = answer_extractor
        self.verbose = verbose
        self.custom_instructions = custom_instructions
        self.credentials = credentials

    def process(
        self,
        query: str,
        resource_id: str,
        score_threshold: Optional[float],
        top_k: int,
        document_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute direct approach: query -> retrieval -> answer"""
        debug_info = []
        logger.info(f"Starting direct approach processing for resource {resource_id}")
        logger.debug(
            f"Query: '{query}', top_k: {top_k}, score_threshold: {score_threshold}"
        )

        # Build metadata filter
        metadata_filter = self.retriever.build_metadata_filter(
            document_name=(document_name or "")
        )
        logger.debug(f"Built metadata filter: {metadata_filter}")

        if self.verbose and metadata_filter:
            debug_info.append(f"Metadata filter: {json.dumps(metadata_filter)}")

        # Retrieve segments
        logger.info(f"Retrieving segments for resource {resource_id}")
        retrieval_result = self._retrieve_segments(
            resource_id, query, metadata_filter, score_threshold, top_k
        )

        if retrieval_result.get("error"):
            logger.error(
                f"Retrieval failed for resource {resource_id}: {retrieval_result.get('error')}"
            )
            return self._format_retrieval_error(retrieval_result, debug_info)

        # Format and extract answer
        segments = self._format_segments(retrieval_result)
        logger.info(f"Retrieved {len(segments)} segments for resource {resource_id}")

        if not segments:
            logger.warning(f"No segments found for resource {resource_id}")
            return {
                "success": True,
                "answer": ResponseMessages.NO_ANSWER,
                "debug_info": debug_info if self.verbose else [],
            }

        logger.info(f"Extracting answer from {len(segments)} segments")
        return self._extract_answer(segments, query, debug_info)

    def _retrieve_segments(
        self,
        resource_id: str,
        query: str,
        metadata_filter: Optional[Dict],
        score_threshold: Optional[float],
        top_k: int,
    ) -> Dict[str, Any]:
        """Retrieve segments from knowledge base"""
        # Check if reranking should be enabled based on credentials
        does_rerank = (
            self.credentials.is_reranking_available() if self.credentials else False
        )

        # Support both working retriever interface (retrieve) and integrations retriever (call)
        # Reranking config is handled internally by the adapter based on backend type
        if hasattr(self.retriever, "retrieve"):
            return self.retriever.retrieve(
                dataset_id=resource_id,  # Backward compatibility
                query=query,
                search_method=RetrieverSearchMethod.HYBRID_SEARCH.value,
                does_rerank=does_rerank,
                top_k=top_k,
                score_threshold_enabled=score_threshold is not None,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
                weights=RetrieverDefaults.WEIGHTS.value,
            )
        else:
            # Integrations retriever (e.g., DifyRetriever) uses call()
            return self.retriever.call(
                query=query,
                method=RetrieverSearchMethod.HYBRID_SEARCH.value,
                top_k=top_k,
                does_rerank=does_rerank,
                score_threshold_enabled=score_threshold is not None,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
                weights=RetrieverDefaults.WEIGHTS.value,
            )

    def _format_segments(
        self, retrieval_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Format retrieval results into segments"""
        formatted_results = format_search_results([retrieval_result])
        return formatted_results.get("result", [])

    def _extract_answer(
        self, segments: List[Dict[str, Any]], query: str, debug_info: List[str]
    ) -> Dict[str, Any]:
        """Extract answer from segments"""
        logger.info(f"Building context from {len(segments)} segments")
        context = build_context_from_segments(segments, self.verbose)
        logger.debug(f"Context built: {len(context)} characters")

        if self.verbose:
            debug_info.append(f"Retrieved {len(segments)} segments")
            debug_info.append(f"Context length: {len(context)} characters")

        logger.info("Starting answer extraction")
        # Enhance query with custom instructions if provided
        enhanced_query = query
        if self.custom_instructions:
            enhanced_query = (
                f"{query}\n\nAdditional Context: {self.custom_instructions}"
            )
            if self.verbose:
                debug_info.append(
                    f"Using custom instructions: {self.custom_instructions}"
                )

        extraction_result = self.answer_extractor.extract(context, enhanced_query)

        if not extraction_result.get("success"):
            logger.error(f"Answer extraction failed: {extraction_result}")
            return {
                "success": False,
                "error": "Answer extraction failed",
                "details": extraction_result,
            }

        answer = extraction_result.get("answer", "")

        # Enhanced logging with metrics
        logger.info(f"Answer extraction successful:")
        logger.info(f"   EXTRACTION METRICS:")
        logger.info(
            f"      - Input context: {len(context):,} chars from {len(segments)} segments"
        )
        logger.info(f"      - Output answer: {len(answer):,} chars")

        # Count terms/items (rough estimate)
        output_lines = answer.split("\n")
        numbered_items = sum(
            1
            for line in output_lines
            if line.strip()
            and (
                line.strip()[0:2].rstrip(".").isdigit()
                or line.strip().startswith("•")
                or line.strip().startswith("-")
            )
        )
        logger.info(f"   Estimated items in extracted answer: ~{numbered_items}")

        return {
            "success": True,
            "answer": answer,
            "segments": segments if self.verbose else [],
            # Always include top source files for downstream citation, even when not verbose
            "source_files": list(
                {s.get("document_name", "") for s in segments if s.get("document_name")}
            )[: AssistantDefaults.MAX_SOURCE_FILES_TO_SHOW.value],
            "debug_info": debug_info if self.verbose else [],
        }

    def _format_retrieval_error(
        self, retrieval_result: Dict[str, Any], debug_info: List[str]
    ) -> Dict[str, Any]:
        """Format retrieval error response"""
        if self.verbose:
            debug_info.append(f"Direct retrieval failed: {retrieval_result}")
            if "debug_payload" in retrieval_result:
                debug_info.append("Debug payload sent to Dify API:")
                debug_info.extend(
                    format_debug_details(
                        [
                            f"{k}: {v}"
                            for k, v in retrieval_result["debug_payload"].items()
                        ]
                    )
                )

        return {
            "success": False,
            "error": "Knowledge base retrieval failed",
            "details": retrieval_result,
        }


class AdvancedApproachProcessor:
    """Processes queries using the advanced approach with file-level processing"""

    def __init__(
        self,
        retriever: Retriever,
        answer_extractor: OrganizationAnswerExtractor,
        verbose: bool = False,
        custom_instructions: Optional[str] = None,
        adaptive_top_k_enabled: bool = AssistantDefaults.ADAPTIVE_TOP_K_ENABLED.value,
        total_segment_budget: int = AssistantDefaults.TOTAL_SEGMENT_BUDGET.value,
        credentials: Optional[Credentials] = None,
    ):
        self.retriever = retriever
        self.answer_extractor = answer_extractor
        self.verbose = verbose
        self.custom_instructions = custom_instructions
        self.adaptive_top_k_enabled = adaptive_top_k_enabled
        self.total_segment_budget = total_segment_budget
        self.credentials = credentials

    def process(
        self,
        query: str,
        resource_id: str,
        top_k: int,
        file_search_result: Dict[str, Any],
        max_workers: Optional[int] = None,
        use_content_booster: bool = True,
        max_boost_keywords: int = AssistantDefaults.MAX_BOOST_KEYWORDS.value,
        llm_api_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute advanced approach with pre-searched files"""
        debug_info = []
        approach_profiling = {}

        # Validate file search results
        if not file_search_result.get("success"):
            return self._format_search_failure(debug_info, approach_profiling)

        file_names = file_search_result.get("file_names", [])
        if not file_names:
            return self._format_no_files_found(debug_info, approach_profiling)

        # File searcher already enforces MAX_FILES limit with reranking
        logger.info(f"📁 Processing {len(file_names)} files from file search")

        if self.verbose:
            debug_info.append(f"Processing {len(file_names)} files")
            if use_content_booster and llm_api_url and llm_model:
                debug_info.append(
                    f"Content booster enabled with max {max_boost_keywords} keywords per file"
                )
            else:
                if not use_content_booster:
                    debug_info.append("Content booster disabled by user")
                elif not (llm_api_url and llm_model):
                    debug_info.append("Content booster disabled - LLM not configured")

        # Extract file search keywords for diversity guidance
        file_search_keywords = file_search_result.get("keywords_used", [])

        # Process files in parallel
        file_answers = self._process_files_parallel(
            file_names,
            query,
            resource_id,
            top_k,
            max_workers,
            approach_profiling,
            use_content_booster=use_content_booster,
            max_boost_keywords=max_boost_keywords,
            llm_api_url=llm_api_url,
            llm_model=llm_model,
            llm_api_token=llm_api_token,
            file_search_keywords=file_search_keywords,
        )

        return {
            "success": True,
            "file_answers": file_answers,
            "updated_query": query,
            "files_processed": len(file_names),
            "debug_info": debug_info if self.verbose else [],
            "profiling": approach_profiling if self.verbose else {},
        }

    def _process_files_parallel(
        self,
        file_names: List[str],
        query: str,
        resource_id: str,
        top_k: int,
        max_workers: Optional[int],
        profiling: Dict[str, Any],
        use_content_booster: bool = True,
        max_boost_keywords: int = AssistantDefaults.MAX_BOOST_KEYWORDS.value,
        llm_api_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_token: Optional[str] = None,
        file_search_keywords: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple files in parallel"""
        file_workers = min(
            max_workers or AssistantDefaults.MAX_WORKERS.value, len(file_names)
        )

        file_answers = []

        with profile_stage(
            f"resource_{resource_id}.advanced.parallel_file_processing",
            profiling,
            self.verbose,
        ):
            with ThreadPoolExecutor(max_workers=file_workers) as executor:
                future_to_meta = {
                    executor.submit(
                        self._process_single_file,
                        file_name,
                        query,
                        resource_id,
                        top_k,
                        profiling,
                        use_content_booster=use_content_booster,
                        max_boost_keywords=max_boost_keywords,
                        llm_api_url=llm_api_url,
                        llm_model=llm_model,
                        llm_api_token=llm_api_token,
                        file_search_keywords=file_search_keywords,
                    ): (idx, file_name)
                    for idx, file_name in enumerate(file_names)
                }

                collected = []
                for future in as_completed(future_to_meta):
                    idx, _ = future_to_meta[future]
                    file_result = future.result()
                    collected.append((idx, file_result))

                collected.sort(key=lambda x: x[0])
                file_answers = [res for _, res in collected]

        return file_answers

    def _process_single_file(
        self,
        file_name: str,
        query: str,
        resource_id: str,
        top_k: int,
        profiling: Dict[str, Any],
        use_content_booster: bool = True,
        max_boost_keywords: int = AssistantDefaults.MAX_BOOST_KEYWORDS.value,
        llm_api_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_token: Optional[str] = None,
        file_search_keywords: List[str] = None,
    ) -> Dict[str, Any]:
        """Process a single file"""
        file_profiling = {}

        try:
            # Generate boosted queries if content booster is enabled
            queries_to_process = [
                (query, "original")
            ]  # List of (query, query_type) tuples

            logger.info(f"Starting advanced processing for file: '{file_name}'")
            logger.info(f"   - Original query: '{query}'")
            logger.info(
                f"   - Content booster: {'ENABLED' if use_content_booster else 'DISABLED'}"
            )
            if file_search_keywords:
                logger.info(
                    f"   - File search keywords (for diversity): {file_search_keywords[:AssistantDefaults.MAX_FILE_SEARCH_KEYWORDS_TO_LOG.value]}"
                )

            if self.verbose:
                file_profiling["content_booster_enabled"] = use_content_booster
                file_profiling["llm_configured"] = bool(llm_api_url and llm_model)
                file_profiling["verbose_mode_info"] = {
                    "enabled": True,
                    "will_show": [
                        "keyword generation details",
                        "query processing steps",
                        "individual answers before combination",
                        "processing statistics",
                        "content booster summary",
                    ],
                }

            if use_content_booster and llm_api_url and llm_model:
                with profile_stage(
                    f"file_{file_name}.content_boosting", file_profiling, self.verbose
                ):
                    try:
                        if self.verbose:
                            file_profiling[
                                "boost_start"
                            ] = "Starting content boost generation"
                            file_profiling["max_boost_keywords"] = max_boost_keywords

                        keyword_generator = ContentBoostKeywordGenerator(
                            llm_api_url,
                            llm_model,
                            llm_api_token=llm_api_token,
                            llm_timeout=KeywordGeneratorDefaults.TIMEOUT_SECONDS.value,
                        )
                        boost_result = keyword_generator.generate(
                            query,
                            max_boost_keywords,
                            document_name=file_name,
                            custom_instructions=self.custom_instructions,  # Pass custom instructions for domain-specific guidance
                            file_search_keywords=file_search_keywords,  # Pass file search keywords for diversity guidance
                        )

                        if boost_result.get("success", False):
                            boosted_keywords = boost_result.get("keyword_sets", [])
                            # IMPORTANT: Enforce max_boost_keywords limit (LLM sometimes generates more)
                            boosted_keywords = boosted_keywords[:max_boost_keywords]
                            for keyword_set in boosted_keywords:
                                # Convert keyword list to query string
                                if isinstance(keyword_set, list):
                                    query_str = " ".join(keyword_set)
                                else:
                                    query_str = str(keyword_set)
                                queries_to_process.append((query_str, "boosted"))

                            # Always log content booster results (not just verbose mode)
                            logger.info(
                                f"   Content booster generated {len(queries_to_process)-1} additional queries"
                            )
                            logger.info(
                                f"      - Example boosted queries: {[q for q, t in queries_to_process if t == 'boosted'][:3]}"
                            )

                            if self.verbose:
                                file_profiling["boost_details"] = {
                                    "total_boosted_queries": len(queries_to_process)
                                    - 1,
                                    "example_boosted_queries": [
                                        q
                                        for q, t in queries_to_process
                                        if t == "boosted"
                                    ][:5],
                                }
                        else:
                            logger.info(
                                "   Content booster returned no keywords; continuing with original query only"
                            )
                            if self.verbose:
                                file_profiling["boost_details"] = {
                                    "total_boosted_queries": 0,
                                    "error": boost_result.get("error"),
                                }
                    except Exception as e:
                        logger.info(f"   Content booster failed: {e}")
                        if self.verbose:
                            file_profiling["boost_error"] = str(e)

            # List to collect answers from different queries for this file
            all_answers = []
            all_segments = []

            # Process each query (original + boosted ones)
            for query_text, query_type in queries_to_process:
                query_result = self._process_query_for_file(
                    file_name,
                    query_text,
                    query,
                    resource_id,
                    top_k,
                    all_segments,
                )

                if query_result.get("success") and query_result.get("answer"):
                    all_answers.append(query_result)

            # Combine answers from all queries (prioritize higher content coverage and clarity)
            if all_answers:
                final_answer = self._combine_file_answers(all_answers)

                # Enhanced logging with metrics
                if self.verbose:
                    total_input_chars = sum(
                        a.get("input_chars", 0) for a in all_answers
                    )
                    output_chars = len(final_answer)
                    if total_input_chars > 0:
                        reduction_pct = (
                            (total_input_chars - output_chars) / total_input_chars * 100
                        )
                        logger.info(
                            f"   Combined answer: {output_chars:,} chars (reduction: {reduction_pct:.1f}%)"
                        )
                    else:
                        logger.info(f"   Combined answer: {output_chars:,} chars")

                    # Count terms/items (rough estimate)
                    output_lines = final_answer.split("\n")
                    numbered_items = sum(
                        1
                        for line in output_lines
                        if line.strip()
                        and (
                            line.strip()[0:2].rstrip(".").isdigit()
                            or line.strip().startswith("•")
                            or line.strip().startswith("-")
                        )
                    )
                    logger.info(
                        f"   Estimated items in combined answer: ~{numbered_items}"
                    )

                response_data = {
                    "file_name": file_name,
                    "success": True,
                    "answer": final_answer,
                    "total_segments": len(all_segments),
                    "segments": all_segments,  # Include segments for analysis
                    "queries_processed": len(queries_to_process),
                    "successful_answers": len(all_answers),
                    "profiling": file_profiling if self.verbose else {},
                }

                # Add individual answers when verbose mode is enabled
                if self.verbose and len(all_answers) > 1:
                    response_data["individual_answers"] = all_answers
                    response_data["combined_answer"] = final_answer

                    # Add content booster summary
                    if use_content_booster and llm_api_url and llm_model:
                        response_data["content_booster_summary"] = {
                            "total_queries": len(queries_to_process),
                            "original_queries": len(
                                [q for q in queries_to_process if q[1] == "original"]
                            ),
                            "boosted_queries": len(
                                [q for q in queries_to_process if q[1] == "boosted"]
                            ),
                            "successful_answers": len(all_answers),
                            "answer_combination": "combined"
                            if len(all_answers) > 1
                            else "single",
                            "keyword_processing_details": file_profiling.get(
                                "boost_details", {}
                            ),
                            "query_processing_summary": file_profiling.get(
                                "query_processing_summary", {}
                            ),
                            "processing_stats": {
                                "total_segments_retrieved": len(all_segments),
                                "queries_with_segments": len(
                                    [
                                        a
                                        for a in all_answers
                                        if a.get("segments_count", 0) > 0
                                    ]
                                ),
                                "queries_without_segments": len(queries_to_process)
                                - len(
                                    [
                                        a
                                        for a in all_answers
                                        if a.get("segments_count", 0) > 0
                                    ]
                                ),
                                "answer_extraction_success_rate": f"{(len(all_answers) / len(queries_to_process)) * 100:.1f}%",
                            },
                        }

                return response_data
            else:
                return self._format_no_segments(file_name, file_profiling)

        except Exception as e:
            return self._format_file_exception(file_name, str(e), file_profiling)

    def _process_query_for_file(
        self,
        file_name: str,
        query_for_rerank: str,
        original_query: str,
        resource_id: str,
        top_k: int,
        all_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process a single query for a file and extract answer"""
        query_profiling = {}
        try:
            with profile_stage(
                f"file_{file_name}.query_processing",
                query_profiling,
                self.verbose,
            ):
                # Build metadata filter targeting this specific file
                metadata_filter = self.retriever.build_metadata_filter(
                    document_name=file_name
                )

                # Retrieve relevant segments
                # Support both working retriever interface (retrieve) and integrations retriever (call)
                # Reranking config is handled internally by the adapter
                if hasattr(self.retriever, "retrieve"):
                    retrieval_result = self.retriever.retrieve(
                        dataset_id=resource_id,  # Backward compatibility
                        query=query_for_rerank,
                        search_method=RetrieverSearchMethod.HYBRID_SEARCH.value,
                        does_rerank=False,
                        top_k=min(
                            top_k, AssistantDefaults.MAX_TOP_K_PER_FILE_QUERY.value
                        )
                        if self.adaptive_top_k_enabled
                        else top_k,
                        score_threshold_enabled=False,
                        metadata_filter=metadata_filter,
                        weights=RetrieverDefaults.WEIGHTS.value,
                    )
                else:
                    retrieval_result = self.retriever.call(
                        query=query_for_rerank,
                        method=RetrieverSearchMethod.HYBRID_SEARCH.value,
                        top_k=min(
                            top_k, AssistantDefaults.MAX_TOP_K_PER_FILE_QUERY.value
                        )
                        if self.adaptive_top_k_enabled
                        else top_k,
                        does_rerank=False,
                        score_threshold_enabled=False,
                        metadata_filter=metadata_filter,
                        weights=RetrieverDefaults.WEIGHTS.value,
                    )

                # Format and extract answer from retrieved segments
                segments = self._format_segments(retrieval_result)
                if self.verbose:
                    query_profiling["segments_count"] = len(segments)

                if not segments:
                    logger.info(
                        f"   No segments found for file '{file_name}' with query type '{'boosted' if query_for_rerank != original_query else 'original'}'"
                    )
                    return {"success": False, "error": "No segments found"}

                # Build context and extract answer
                context = build_context_from_segments(segments, self.verbose)
                extraction_result = self.answer_extractor.extract(
                    context, original_query
                )

                if not extraction_result.get("success"):
                    return {
                        "success": False,
                        "error": "Answer extraction failed",
                        "details": extraction_result,
                    }

                answer = extraction_result.get("answer", "")
                if self.verbose:
                    query_profiling["answer_length"] = len(answer)
                    query_profiling["segments_count"] = len(segments)
                    query_profiling["input_chars"] = len(context)

                # Return enhanced result with metrics
                return {
                    "success": True,
                    "answer": answer,
                    "segments_count": len(segments),
                    "profiling": query_profiling,
                    "input_chars": len(context),
                }
        except Exception as e:
            return {
                "success": False,
                "error": "Query processing failed",
                "details": str(e),
                "profiling": query_profiling,
            }

    def _combine_file_answers(self, answers: List[Dict[str, Any]]) -> str:
        """Combine answers from multiple queries intelligently"""
        if not answers:
            return ResponseMessages.NO_ANSWER

        # Prioritize answers with higher segment count and more content
        answers_sorted = sorted(
            answers,
            key=lambda x: (x.get("segments_count", 0), len(x.get("answer", ""))),
            reverse=True,
        )

        # If only one answer, return it
        if len(answers_sorted) == 1:
            return answers_sorted[0].get("answer", ResponseMessages.NO_ANSWER)

        # Combine top answers with clear separation and deduplication of similar lines
        combined_lines = []
        seen_lines = set()

        for answer in answers_sorted[
            : AssistantDefaults.MAX_TOP_ANSWERS_TO_COMBINE.value
        ]:
            for line in answer.get("answer", "").split("\n"):
                normalized = line.strip()
                if normalized and normalized not in seen_lines:
                    combined_lines.append(normalized)
                    seen_lines.add(normalized)

        return "\n".join(combined_lines)

    def _format_no_segments(
        self, file_name: str, profiling: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format response when no segments are found"""
        if self.verbose:
            profiling["segments_count"] = 0
        return {
            "file_name": file_name,
            "success": False,
            "answer": "",
            "total_segments": 0,
            "segments": [],
            "queries_processed": 0,
            "successful_answers": 0,
            "profiling": profiling if self.verbose else {},
        }

    def _format_file_exception(
        self, file_name: str, error: str, profiling: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format response when an exception occurs during file processing"""
        if self.verbose:
            profiling["error"] = error
        return {
            "file_name": file_name,
            "success": False,
            "error": error,
            "profiling": profiling if self.verbose else {},
        }

    def _format_search_failure(
        self, debug_info: List[str], profiling: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format response when file search fails"""
        if self.verbose:
            debug_info.append("File search failed")
        return {
            "success": False,
            "error": "File search failed",
            "debug_info": debug_info if self.verbose else [],
            "profiling": profiling if self.verbose else {},
        }

    def _format_no_files_found(
        self, debug_info: List[str], profiling: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format response when no files are found in file search"""
        if self.verbose:
            debug_info.append("No files found in file search")
        return {
            "success": True,
            "file_answers": [],
            "updated_query": "",
            "files_processed": 0,
            "debug_info": debug_info if self.verbose else [],
            "profiling": profiling if self.verbose else {},
        }


class DatasetProcessor:
    """Coordinates processing across multiple datasets"""

    def __init__(
        self,
        components: Dict[str, Any],
        config: ProcessingConfig,
        credentials: Credentials,
        profiling_data: Dict[str, Any] = None,
        custom_instructions: Optional[str] = None,
        focus_document_name: Optional[str] = None,
    ):
        self.components = components
        self.config = config
        self.credentials = credentials
        # Preserve custom instructions for per-dataset processors
        self.custom_instructions = custom_instructions
        self.focus_document_name = focus_document_name or ""
        self.file_search_strategy = FileSearchStrategy(
            components["file_discover_factory"],
            self.credentials,
            verbose=config.verbose,
        )
        self.direct_processor = DirectApproachProcessor(
            components.get("retriever"),
            components["answer_extractor"],
            verbose=config.verbose,
            custom_instructions=custom_instructions,
            credentials=credentials,
        )
        self.advanced_processor = AdvancedApproachProcessor(
            components.get("retriever"),
            components["answer_extractor"],
            verbose=config.verbose,
            custom_instructions=custom_instructions,
            adaptive_top_k_enabled=config.adaptive_top_k_enabled,
            total_segment_budget=config.total_segment_budget,
            credentials=credentials,
        )
        self.retriever_factory = components.get("retriever_factory")
        self.profiling_data = profiling_data or {}

    def process_datasets(
        self, dataset_pairs: List[Dict[str, str]], query: str
    ) -> Tuple[List[DatasetResult], List[Dict[str, Any]]]:
        """Process multiple datasets and return results and candidate answers"""
        results = []
        candidates = []

        # Calculate optimal worker distribution
        worker_dist = WorkerDistributor.calculate_distribution(
            self.config.max_workers, len(dataset_pairs)
        )

        if self.config.verbose:
            self.profiling_data["worker_distribution"] = worker_dist.__dict__

        # Verify resource contents (lightweight check)
        resources_with_files = []
        for pair in dataset_pairs:
            resource_id_value = pair.get("id")
            try:
                has_files = True
                if self.retriever_factory:
                    r = self.retriever_factory(resource_id_value)
                    if hasattr(r, "list_files"):
                        files = r.list_files(resource_id=resource_id_value, timeout=30)
                        has_files = len(files) > 0
            except Exception as e:
                # NEW BEHAVIOR: If file lister fails, proceed assuming resource may have files
                logger.warning(
                    f"File lister check failed for resource {resource_id_value}: {e}. Proceeding with processing."
                )
                has_files = True

            resources_with_files.append(
                {"id": resource_id_value, "has_files": has_files}
            )

        # Check if at least one resource has files
        if not any(d.get("has_files", False) for d in resources_with_files):
            raise ValueError("No resources with files found")

        # Process resources with available files
        for pair in resources_with_files:
            resource_id_value = pair["id"]
            has_files = pair.get("has_files", False)

            if not has_files:
                continue

            # Create per-resource components using factory pattern
            per_resource_components = dict(self.components)
            retriever_factory = self.components.get("retriever_factory")
            if retriever_factory:
                per_resource_components["retriever"] = retriever_factory(
                    resource_id_value
                )

            # IMPORTANT: Use per-resource processors so retriever is bound to the resource
            direct_processor = DirectApproachProcessor(
                per_resource_components.get("retriever"),
                self.components["answer_extractor"],
                verbose=self.config.verbose,
                custom_instructions=self.custom_instructions,
                credentials=self.credentials,
            )

            advanced_processor = AdvancedApproachProcessor(
                per_resource_components.get("retriever"),
                self.components["answer_extractor"],
                verbose=self.config.verbose,
                custom_instructions=self.custom_instructions,
                adaptive_top_k_enabled=self.config.adaptive_top_k_enabled,
                total_segment_budget=self.config.total_segment_budget,
                credentials=self.credentials,
            )

            # First stage: standalone file search (or pin to a specific file if requested)
            if self.focus_document_name:
                file_search_result = {
                    "success": True,
                    "file_names": [self.focus_document_name],
                    "keywords_used": [],
                }
            else:
                file_search_result = self.file_search_strategy.parallel_search(
                    query, resource_id_value, worker_dist.file_workers
                )

            # Second stage: direct approach
            direct_result = direct_processor.process(
                query,
                resource_id_value,
                self.config.score_threshold,
                self.config.top_k,
                document_name=(self.focus_document_name or None),
            )

            # Third stage: advanced approach using file search results
            advanced_result = advanced_processor.process(
                query,
                resource_id_value,
                top_k=self.config.top_k,
                file_search_result=file_search_result,
                max_workers=worker_dist.file_workers,
                use_content_booster=self.config.use_content_booster,
                max_boost_keywords=self.config.max_boost_keywords,
                llm_api_url=self.credentials.llm_api_url,
                llm_model=self.credentials.llm_model,
                llm_api_token=self.credentials.llm_api_token,
            )

            # Convert file-level answers to candidate answers
            file_answers = advanced_result.get("file_answers", [])
            for answer in file_answers:
                candidates.append(
                    {
                        "source": "advanced",
                        "resource_id": resource_id_value,
                        "file_name": answer.get("file_name", ""),
                        "answer": answer.get("answer", ""),
                        "success": answer.get("success", False),
                    }
                )

            # Add direct answer as fallback candidate to avoid empty candidate sets
            direct_answer_text = direct_result.get("answer", "")
            if (
                direct_answer_text
                and direct_answer_text.strip().upper() != ResponseMessages.NO_ANSWER
            ):
                direct_sources = [
                    name for name in (direct_result.get("source_files") or []) if name
                ]
                display_sources = []
                for name in direct_sources[
                    : AssistantDefaults.MAX_DISPLAY_SOURCES.value
                ]:
                    try:
                        display_sources.append(f"{resource_id_value}/{unquote(name)}")
                    except Exception:
                        display_sources.append(f"{resource_id_value}/{name}")
                display_source_str = (
                    "; ".join(display_sources) if display_sources else resource_id_value
                )
                file_name_hint = direct_sources[0] if direct_sources else ""
                candidates.append(
                    {
                        "source": "direct",
                        "resource_id": resource_id_value,
                        "file_name": file_name_hint,
                        "answer": direct_answer_text,
                        "success": True,
                        "display_source": display_source_str,
                    }
                )

            # Create standardized result
            result = DatasetResult(
                resource_id=resource_id_value,
                direct_result=direct_result,
                advanced_result=advanced_result,
                candidates=candidates,
                debug_info=[],
                profiling=self.profiling_data if self.config.verbose else {},
            )
            results.append(result)

        return results, candidates
