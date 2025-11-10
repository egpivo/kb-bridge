import json
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import dspy
from pydantic import BaseModel, Field

from .answer_reranker import AnswerReranker
from .constants import (
    AnswerExtractorDefaults,
    ResponseMessages,
    StructuredAnswerFormatterDefaults,
)

logger = logging.getLogger(__name__)


class SourceInfo(BaseModel):
    """Source information for citation"""

    source: str = Field(description="Source identifier (dataset_id/file_name)")
    relevance: str = Field(description="Relevance level: high, medium, or low")


class StructuredAnswerSignature(dspy.Signature):
    """Format multiple candidate answers into a comprehensive, deduplicated structured response.

    Your task is to combine multiple candidate answers into ONE unified answer that:
    1. REMOVES ALL DUPLICATES - each unique piece of information appears EXACTLY ONCE
    2. PRESERVES ALL UNIQUE ITEMS - include every distinct entity/term from all candidates
    3. MERGES similar definitions - combine different wordings of the same concept
    4. FORMATS consistently - use clear structure with inline citations

    ## CRITICAL DEDUPLICATION RULES

    For List/Definition Queries (queries asking for "TERMS", "DEFINITIONS", "LIST ALL"):
    1. **STRICT DEDUPLICATION**: If Term X appears in multiple candidates, include it ONLY ONCE
    2. **Merge Definitions**: If a term has multiple definitions, merge them into one comprehensive definition
    3. **Single Unified List**: Output ONE numbered list, not multiple separate lists
    4. **NO "---" Separators**: Never use "---" or start new sections
    5. **NO "Additionally"**: Never say "Additionally, here is..." - everything in ONE list
    6. **Preserve ALL Unique Items**: Include EVERY unique term that appears in ANY candidate

    ## Inline Citation Requirements (MANDATORY)
    - Each bullet/numbered item MUST end with an inline citation: " (Source: display_source)"
    - If multiple sources support the same item, list them separated by "; "
    - Example: "Term definition (Source: Dataset/File A; Dataset/File B)"

    ## Entity Preservation Rules
    - **Different names = DIFFERENT entities** - preserve both even if similar
    - Only remove EXACT duplicates (same name appearing multiple times)
    - When in doubt, treat as DIFFERENT and keep both
    - Goal: PRESERVATION over SUMMARIZATION

    ## Success Criteria
    - Input has N unique entities → Output should have N entities
    - Only deduplication = removing exact repeats
    - Completeness = preserve all unique items
    """

    query: str = dspy.InputField(desc="The original user query")
    candidates_json: str = dspy.InputField(
        desc="JSON array of candidate answers with metadata"
    )
    answer: str = dspy.OutputField(
        desc="Comprehensive answer with ALL unique information, NO duplicates, with inline citations on every item"
    )
    sources: List[SourceInfo] = dspy.OutputField(
        desc="List of source information with relevance levels"
    )
    total_sources: int = dspy.OutputField(desc="Total number of sources used")
    confidence: str = dspy.OutputField(desc="Confidence level: high, medium, or low")


class StructuredAnswerFormatter(dspy.Module):
    """
    Formats multiple candidate answers into a structured response using DSPy
    """

    def __init__(
        self,
        llm_api_url: str,
        llm_model: str,
        llm_api_token: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        rerank_url: Optional[str] = None,
        rerank_model: Optional[str] = None,
        use_cot: bool = False,
    ):
        """
        Initialize the structured answer formatter with DSPy

        Args:
            llm_api_url: LLM API service URL
            llm_model: LLM model name
            llm_api_token: Optional API token (falls back to default if None)
            llm_temperature: Optional temperature (falls back to default if None)
            llm_timeout: Optional timeout in seconds (falls back to default if None)
            max_tokens: Optional max tokens (falls back to default if None)
            rerank_url: Optional reranking service URL
            rerank_model: Optional reranking model name
            use_cot: Whether to use Chain of Thought reasoning (default: False)
        """
        super().__init__()

        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.llm_api_token = llm_api_token

        # Use provided values or fall back to defaults
        self.llm_temperature = (
            llm_temperature
            if llm_temperature is not None
            else AnswerExtractorDefaults.TEMPERATURE.value
        )
        self.llm_timeout = (
            llm_timeout
            if llm_timeout is not None
            else AnswerExtractorDefaults.TIMEOUT_SECONDS.value
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else AnswerExtractorDefaults.MAX_TOKENS.value
        )

        # Initialize reranker if credentials provided
        self.reranker = None
        if rerank_url and rerank_model:
            self.reranker = AnswerReranker(rerank_url, rerank_model)

        # Configure DSPy LM
        self._configure_dspy_lm()

        # Use Predict (faster) or ChainOfThought (reasoning)
        if use_cot:
            self.predictor = dspy.ChainOfThought(StructuredAnswerSignature)
        else:
            self.predictor = dspy.Predict(StructuredAnswerSignature)

    def _configure_dspy_lm(self):
        """Configure DSPy language model.

        Creates a local LM instance that will be used with dspy.settings.context()
        to provide thread-safe, per-call configuration.
        """
        lm_kwargs = {
            "model": self.llm_model,
            "api_base": self.llm_api_url,
            "temperature": self.llm_temperature,
            "max_tokens": self.max_tokens,
            "request_timeout": self.llm_timeout,
        }

        # Only pass api_key if it's not None
        if self.llm_api_token:
            lm_kwargs["api_key"] = self.llm_api_token

        # Create LM instance - will be used with context manager for thread safety
        self._lm = dspy.LM(**lm_kwargs)

    def forward(self, query: str, candidates_json: str) -> Dict[str, Any]:
        """DSPy Module forward method.

        Args:
            query: The original user query
            candidates_json: JSON string of candidate answers

        Returns:
            Dict with structured answer data
        """
        # Use context manager for thread-safe LM configuration
        with dspy.settings.context(lm=self._lm):
            result = self.predictor(query=query, candidates_json=candidates_json)

        # Convert DSPy result to dict
        return {
            "answer": result.answer,
            "sources": [
                {"source": s.source, "relevance": s.relevance} for s in result.sources
            ],
            "total_sources": result.total_sources,
            "confidence": result.confidence,
        }

    def format_structured_answer(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format multiple candidate answers into a structured response

        Args:
            query: The original user query
            candidates: List of candidate answers with file information

        Returns:
            Dict containing the structured result or error information
        """
        start_time = time.time()

        try:
            logger.info(f"Structured Answer Formatter starting:")
            logger.info(f"   - Query: '{query}'")
            logger.info(f"   - Total candidates received: {len(candidates)}")

            # Filter out unsuccessful candidates and trivial/N/A answers
            valid_candidates = []
            for c in candidates:
                if not c.get("success", False):
                    continue
                ans = c.get("answer", "")
                if not ans:
                    continue
                if ans.strip().upper() == ResponseMessages.NO_ANSWER:
                    continue
                valid_candidates.append(c)

            logger.info(
                f"   - Valid candidates after filtering: {len(valid_candidates)}"
            )

            # Log each candidate's answer length
            for idx, candidate in enumerate(valid_candidates, 1):
                answer_len = len(candidate.get("answer", ""))
                source = candidate.get("source", "unknown")
                file_name = candidate.get("file_name", "")
                try:
                    file_name = unquote(file_name)
                except Exception as e:
                    logger.debug(f"Failed to unquote file_name '{file_name}': {e}")
                logger.info(
                    f"   - Candidate {idx} [{source}/{file_name}]: {answer_len} chars"
                )

            if not valid_candidates:
                logger.warning("   No valid candidates found")
                return self._build_no_results_response(query)

            # Use reranking to intelligently select best candidates if we have too many
            rerank_threshold = StructuredAnswerFormatterDefaults.RERANK_THRESHOLD.value
            max_candidates = StructuredAnswerFormatterDefaults.MAX_CANDIDATES.value
            max_answer_length = (
                StructuredAnswerFormatterDefaults.MAX_CANDIDATE_ANSWER_LENGTH.value
            )

            if len(valid_candidates) > rerank_threshold and self.reranker:
                logger.info(
                    f"   Using reranking to select top {max_candidates} from {len(valid_candidates)} candidates"
                )
                try:
                    rerank_start = time.time()
                    rerank_result = self.reranker.rerank_answers(
                        query, valid_candidates
                    )
                    rerank_duration = time.time() - rerank_start

                    if rerank_result.get("detailed_results"):
                        # Extract top candidates based on rerank scores
                        sorted_results = rerank_result["detailed_results"][
                            :max_candidates
                        ]
                        valid_candidates = [
                            r["candidate_answer"] for r in sorted_results
                        ]

                        # Log reranking statistics
                        logger.info(f"   Reranking completed in {rerank_duration:.2f}s")
                        logger.info(
                            f"   Selected top {len(valid_candidates)} candidates by relevance"
                        )
                        if len(sorted_results) > 0:
                            top_score = sorted_results[0]["relevance_score"]
                            bottom_score = sorted_results[-1]["relevance_score"]
                            logger.info(
                                f"   Score range: {bottom_score:.3f} - {top_score:.3f}"
                            )
                    else:
                        logger.warning(
                            "   Reranking returned no results, using original order"
                        )
                        valid_candidates = valid_candidates[:max_candidates]

                except Exception as e:
                    logger.warning(
                        f"   Reranking failed: {str(e)}, falling back to limiting"
                    )
                    valid_candidates = valid_candidates[:max_candidates]
            elif len(valid_candidates) > max_candidates:
                logger.warning(
                    f"   Limiting candidates from {len(valid_candidates)} to {max_candidates} (reranking not available)"
                )
                valid_candidates = valid_candidates[:max_candidates]

            # Truncate long answers and track statistics
            truncated_count = 0
            total_chars_before = 0
            total_chars_after = 0

            for candidate in valid_candidates:
                answer = candidate.get("answer", "")
                total_chars_before += len(answer)

                if len(answer) > max_answer_length:
                    truncated_count += 1
                    # Truncate with ellipsis
                    candidate["answer"] = (
                        answer[:max_answer_length]
                        + "\n\n[... truncated for API size limits ...]"
                    )

                total_chars_after += len(candidate.get("answer", ""))

            if truncated_count > 0:
                reduction_pct = (
                    (
                        (total_chars_before - total_chars_after)
                        / total_chars_before
                        * 100
                    )
                    if total_chars_before > 0
                    else 0
                )
                logger.info(
                    f"   Truncated {truncated_count} candidate(s) to prevent 413 error"
                )
                logger.info(
                    f"   Payload size: {total_chars_before:,} -> {total_chars_after:,} chars ({reduction_pct:.1f}% reduction)"
                )

            # Add human-readable display_source for inline citations
            for c in valid_candidates:
                if not c.get("display_source"):
                    try:
                        ds_id = c.get("dataset_id", "")
                        fname = c.get("file_name", "")
                        if fname:
                            c["display_source"] = f"{ds_id}/{unquote(fname)}"
                        else:
                            c["display_source"] = ds_id
                    except Exception as e:
                        logger.debug(
                            f"Failed to unquote file_name '{fname}' for display_source: {e}"
                        )
                        c["display_source"] = c.get("dataset_id", "")

            # Format candidates for the prompt (without indentation to save space)
            candidates_json = json.dumps(valid_candidates)

            # Call DSPy Module
            logger.info(f"   Calling DSPy for structured formatting...")
            llm_start = time.time()

            try:
                structured_data = self(query=query, candidates_json=candidates_json)
                llm_duration = time.time() - llm_start
                logger.info(f"   DSPy call completed in {llm_duration:.2f}s")

                answer_text = structured_data.get("answer", "")

                # Calculate input and output metrics
                total_input_chars = sum(
                    len(c.get("answer", "")) for c in valid_candidates
                )
                output_chars = len(answer_text)
                reduction_pct = (
                    ((total_input_chars - output_chars) / total_input_chars * 100)
                    if total_input_chars > 0
                    else 0
                )

                # Count terms/items in INPUT candidates
                input_terms_total = 0
                for c in valid_candidates:
                    cand_text = c.get("answer", "")
                    cand_lines = cand_text.split("\n")
                    cand_items = sum(
                        1
                        for line in cand_lines
                        if line.strip()
                        and (
                            line.strip()[0:2].rstrip(".").isdigit()
                            or line.strip().startswith("•")
                            or line.strip().startswith("-")
                        )
                    )
                    input_terms_total += cand_items

                # Count terms/items in OUTPUT (rough estimate)
                output_lines = answer_text.split("\n")
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

                logger.info(f"   DSPy formatting successful:")
                logger.info(
                    f"      - Input: {total_input_chars:,} chars from {len(valid_candidates)} candidate(s)"
                )
                logger.info(
                    f"      - Output: {output_chars:,} chars ({reduction_pct:.1f}% reduction)"
                )
                logger.info(
                    f"      - Sources: {len(structured_data.get('sources', []))}, Confidence: {structured_data.get('confidence', ResponseMessages.NO_ANSWER)}"
                )

                total_duration = time.time() - start_time
                logger.info(f"   Total formatting time: {total_duration:.2f}s")

                return self._build_success_response(structured_data, query, candidates)

            except Exception as e:
                logger.error(f"   DSPy call failed: {str(e)}")
                return self._build_exception_response(e, query, candidates)

        except Exception as e:
            return self._build_exception_response(e, query, candidates)

    # _parse_structured_response removed - DSPy handles parsing automatically

    def _build_base_response(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build base response with common fields"""
        return {
            "query": query,
            "candidates_count": len(candidates),
            "model_used": self.llm_model,
            "tool_type": "structured_answer_formatter",
        }

    def _build_success_response(
        self,
        structured_data: Dict[str, Any],
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build successful response"""
        response = self._build_base_response(query, candidates)
        response.update(
            {
                "success": True,
                "structured_answer": structured_data,
                "answer": structured_data.get("answer", ""),
                "total_sources": structured_data.get("total_sources", 0),
                "confidence": structured_data.get("confidence", "medium"),
            }
        )
        return response

    # _build_error_response and _build_parse_error_response removed
    # DSPy handles API calls and parsing automatically

    def _build_no_results_response(self, query: str) -> Dict[str, Any]:
        """Build response when no valid candidates are found"""
        return {
            "success": False,
            "error": "No valid candidates found",
            "details": "All candidates were filtered out or contained no valid answers",
            "query": query,
            "candidates_count": 0,
            "model_used": self.llm_model,
            "tool_type": "structured_answer_formatter",
        }

    def _build_exception_response(
        self, exception: Exception, query: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build error response for exceptions"""
        response = self._build_base_response(query, candidates)
        response.update(
            {
                "success": False,
                "error": f"Structured answer formatting failed: {str(exception)}",
                "details": "An unexpected error occurred during processing",
            }
        )
        return response
