import json
import logging
from typing import Any, Dict, List, Optional

import dspy

from kbbridge.integrations.retriever_base import ChunkHit, FileHit

from .constants import ReflectionConstants
from .models import QualityScores, ReflectionResult

logger = logging.getLogger(__name__)


class QualityEval(dspy.Signature):
    """DSPy signature for evaluating answer quality."""

    query: str = dspy.InputField()
    answer: str = dspy.InputField()
    sources: str = dspy.InputField()

    completeness: float = dspy.OutputField(desc="0-1")
    accuracy: float = dspy.OutputField(desc="0-1")
    clarity: float = dspy.OutputField(desc="0-1")
    relevance: float = dspy.OutputField(desc="0-1")
    confidence: float = dspy.OutputField(desc="0-1")
    feedback: str = dspy.OutputField()
    suggestions: str = dspy.OutputField(desc="JSON array")
    missing: str = dspy.OutputField(desc="JSON array")


class Evaluator:
    """Evaluates answer quality using DSPy and LLM."""

    def __init__(
        self,
        lm: dspy.LM,
        threshold: float = ReflectionConstants.DEFAULT_QUALITY_THRESHOLD,
        examples: Optional[List[Any]] = None,
    ) -> None:
        self._lm = lm
        self.threshold = threshold
        self.examples = (
            examples[: ReflectionConstants.MAX_EXAMPLES_TO_USE] if examples else []
        )
        self.evaluator = dspy.ChainOfThought(QualityEval)

        if self.examples:
            logger.info(f"Evaluator initialized with {len(self.examples)} examples")

    async def evaluate(
        self, query: str, answer: str, sources: List[Dict[str, Any]], attempt: int = 1
    ) -> ReflectionResult:
        """Evaluate answer quality and return structured result."""
        try:
            sources_text = self._format_sources(sources)

            logger.info(
                f"Evaluating answer (attempt {attempt}): "
                f"query_length={len(query)}, answer_length={len(answer)}"
            )

            with dspy.settings.context(lm=self._lm):
                result = self.evaluator(
                    query=query,
                    answer=answer,
                    sources=sources_text,
                )

            scores = QualityScores(
                completeness=float(result.completeness),
                accuracy=float(result.accuracy),
                clarity=float(result.clarity),
                relevance=float(result.relevance),
                confidence=float(result.confidence),
            )

            overall = scores.calculate_overall()
            passed = overall >= self.threshold

            logger.info(
                f"Evaluation result (attempt {attempt}): "
                f"overall={overall:.3f}, passed={passed}"
            )

            return ReflectionResult(
                scores=scores,
                overall_score=overall,
                passed=passed,
                feedback=result.feedback,
                refinement_suggestions=self._parse_json(result.suggestions),
                missing_aspects=self._parse_json(result.missing),
                attempt=attempt,
                threshold=self.threshold,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)

            return ReflectionResult(
                scores=QualityScores(
                    completeness=ReflectionConstants.FALLBACK_SCORE,
                    accuracy=ReflectionConstants.FALLBACK_SCORE,
                    clarity=ReflectionConstants.FALLBACK_SCORE,
                    relevance=ReflectionConstants.FALLBACK_SCORE,
                    confidence=ReflectionConstants.FALLBACK_SCORE,
                ),
                overall_score=ReflectionConstants.FALLBACK_SCORE,
                passed=True,
                feedback=str(e),
                attempt=attempt,
                threshold=self.threshold,
            )

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format source list into readable text."""
        if not sources:
            return "No sources"

        formatted = []
        for i, source in enumerate(sources[:10], 1):
            title = source.get("title", "Unknown")
            content = source.get("content", "")[:200]
            formatted.append(f"{i}. {title}\n   {content}...")
        return "\n".join(formatted)

    def _parse_json(self, text: str) -> List[str]:
        """Parse JSON array from text."""
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            items = text.strip("[]\"'")
            return [item.strip("\"'") for item in items.split(",") if item]


def get_default_examples() -> List[Any]:
    """Return empty list. Examples should be generated dynamically."""
    return []


class FileDiscoveryRecallEvaluator:
    """Evaluates file discovery recall rate using ground truth or statistics."""

    @staticmethod
    def evaluate_recall(
        query: str,
        discovered_files: List[str],
        ground_truth_files: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate file discovery recall rate.

        Args:
            query: Search query
            discovered_files: List of discovered file names
            ground_truth_files: List of relevant file names (ground truth)

        Returns:
            Dictionary with recall, precision, F1, and other metrics
        """
        relevant_set = set(ground_truth_files)
        discovered_set = set(discovered_files)

        found = len(relevant_set & discovered_set)
        total_relevant = len(relevant_set)
        total_discovered = len(discovered_set)

        recall = found / total_relevant if total_relevant > 0 else 0.0
        precision = found / total_discovered if total_discovered > 0 else 0.0
        f1_score = (
            2 * (recall * precision) / (recall + precision)
            if (recall + precision) > 0
            else 0.0
        )

        missed_files = list(relevant_set - discovered_set)
        false_positives = list(discovered_set - relevant_set)

        return {
            "query": query,
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "found_relevant": found,
            "total_relevant": total_relevant,
            "total_discovered": total_discovered,
            "missed_files": missed_files,
            "false_positives": false_positives,
            "recall_status": FileDiscoveryRecallEvaluator._classify_recall(recall),
        }

    @staticmethod
    def _classify_recall(recall: float) -> str:
        """Classify recall level."""
        if recall >= 0.8:
            return "high"
        elif recall >= 0.5:
            return "medium"
        else:
            return "low"

    @staticmethod
    def evaluate_batch(
        test_cases: List[Dict[str, Any]],
        file_discover_fn: callable,
    ) -> Dict[str, Any]:
        """
        Batch evaluate recall for multiple queries.

        Args:
            test_cases: List of test cases, each with query and ground_truth_files
            file_discover_fn: File discovery function

        Returns:
            Batch evaluation results
        """
        results = []
        total_recall = 0.0
        total_precision = 0.0

        for test_case in test_cases:
            query = test_case["query"]
            ground_truth_files = test_case["ground_truth_files"]

            try:
                discover_result = file_discover_fn(query)
                if isinstance(discover_result, dict):
                    discovered_files = discover_result.get("distinct_files", [])
                else:
                    discovered_files = [f.file_name for f in discover_result]
            except Exception as e:
                logger.error(f"File discovery failed for query '{query}': {e}")
                discovered_files = []

            evaluation = FileDiscoveryRecallEvaluator.evaluate_recall(
                query, discovered_files, ground_truth_files
            )

            results.append(evaluation)
            total_recall += evaluation["recall"]
            total_precision += evaluation["precision"]

        n = len(results)
        avg_recall = total_recall / n if n > 0 else 0.0
        avg_precision = total_precision / n if n > 0 else 0.0

        high_recall_count = sum(1 for r in results if r["recall"] >= 0.8)
        medium_recall_count = sum(1 for r in results if 0.5 <= r["recall"] < 0.8)
        low_recall_count = sum(1 for r in results if r["recall"] < 0.5)

        return {
            "average_recall": avg_recall,
            "average_precision": avg_precision,
            "average_f1": (
                2 * (avg_recall * avg_precision) / (avg_recall + avg_precision)
                if (avg_recall + avg_precision) > 0
                else 0.0
            ),
            "total_queries": n,
            "recall_distribution": {
                "high": high_recall_count,
                "medium": medium_recall_count,
                "low": low_recall_count,
            },
            "per_query_results": results,
        }

    @staticmethod
    def evaluate_by_statistics(
        query: str,
        discovered_files: List[Any],  # List[FileHit]
        all_files_count: int,
    ) -> Dict[str, Any]:
        """
        Evaluate based on statistics (no ground truth required).

        Args:
            query: Search query
            discovered_files: List of discovered files (FileHit objects)
            all_files_count: Total number of files in knowledge base

        Returns:
            Statistical evaluation results
        """
        discovered_count = len(discovered_files)

        scores = [getattr(f, "score", 0.0) for f in discovered_files]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        if scores:
            score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        else:
            score_variance = 0.0

        coverage_ratio = (
            discovered_count / all_files_count if all_files_count > 0 else 0.0
        )

        potential_low_recall = (
            score_variance > 0.1 and coverage_ratio < 0.1 and avg_score < 0.5
        )

        return {
            "query": query,
            "coverage_ratio": coverage_ratio,
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "score_variance": score_variance,
            "potential_low_recall": potential_low_recall,
            "statistics": {
                "total_files": all_files_count,
                "discovered_count": discovered_count,
            },
        }


class FileDiscoveryQualityEval(dspy.Signature):
    """DSPy signature for evaluating file discovery quality."""

    query: str = dspy.InputField(desc="The search query")
    discovered_files_summary: str = dspy.InputField(
        desc="Summary of discovered files with scores and content previews"
    )
    chunks_summary: str = dspy.InputField(
        desc="Summary of retrieved chunks with scores and relevance"
    )
    all_files_count: int = dspy.InputField(
        desc="Total number of files in knowledge base"
    )

    completeness: float = dspy.OutputField(
        desc="File discovery completeness (0-1): Are all relevant files found?"
    )
    relevance: float = dspy.OutputField(
        desc="File relevance (0-1): Are discovered files relevant to query?"
    )
    coverage: float = dspy.OutputField(
        desc="Coverage (0-1): Do files cover all aspects of the query?"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence (0-1): Confidence in file discovery quality"
    )
    feedback: str = dspy.OutputField(desc="Feedback on file discovery quality")
    missing_aspects: str = dspy.OutputField(
        desc="JSON array: Aspects of query that might be missing from discovered files"
    )
    should_expand_search: bool = dspy.OutputField(
        desc="Should we expand search (increase top_k_recall, top_k_return)?"
    )
    estimated_recall: float = dspy.OutputField(
        desc="Estimated recall rate (0-1): Likely percentage of relevant files found"
    )


class FileDiscoveryQualityEvaluator:
    """
    Evaluates file discovery quality using LLM, based on Reflector pattern.

    Used before advanced/naive search to determine if file discovery is sufficient.
    Can judge based on chunks score + summary (semantic).
    """

    def __init__(
        self,
        llm_model: str,
        llm_api_url: str,
        api_key: str,
        temperature: float = 0.0,
        timeout: int = 30,
    ):
        """
        Initialize File Discovery Quality Evaluator.

        Args:
            llm_model: LLM model name
            llm_api_url: LLM API URL
            api_key: API key
            temperature: Temperature for LLM
            timeout: Request timeout
        """
        self.llm_model = llm_model
        self.llm_api_url = llm_api_url
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

        try:
            from kbbridge.core.reflection.config import setup

            self._lm = setup(
                llm_model,
                llm_api_url,
                api_key,
                temperature=temperature,
            )
            self.evaluator = dspy.ChainOfThought(FileDiscoveryQualityEval)
            self.use_dspy = True
            logger.info(
                f"File Discovery Quality Evaluator enabled: DSPy setup successful, model={llm_model}"
            )
        except Exception as e:
            logger.warning(
                f"File Discovery Quality Evaluator disabled: DSPy setup failed: {e}",
                exc_info=True,
            )
            self._lm = None
            self.evaluator = None
            self.use_dspy = False

    async def evaluate(
        self,
        query: str,
        discovered_files: List[FileHit],
        chunks: List[ChunkHit],
        all_files_count: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate file discovery quality.

        Args:
            query: Search query
            discovered_files: List of discovered files
            chunks: List of retrieved chunks
            all_files_count: Total number of files in knowledge base

        Returns:
            Evaluation result dictionary with completeness, relevance, coverage, etc.
        """
        if not self.evaluator or not self._lm:
            logger.info(
                "File discovery evaluation skipped: evaluator or LM not initialized"
            )
            return None

        try:
            files_summary = self._build_files_summary(discovered_files, chunks)
            chunks_summary = self._build_chunks_summary(chunks)

            logger.info(
                f"Evaluating file discovery quality: "
                f"query_length={len(query)}, files={len(discovered_files)}, "
                f"chunks={len(chunks)}, total_files={all_files_count}"
            )

            with dspy.settings.context(lm=self._lm):
                result = self.evaluator(
                    query=query,
                    discovered_files_summary=files_summary,
                    chunks_summary=chunks_summary,
                    all_files_count=all_files_count,
                )

            evaluation = {
                "completeness": float(result.completeness),
                "relevance": float(result.relevance),
                "coverage": float(result.coverage),
                "confidence": float(result.confidence),
                "estimated_recall": float(result.estimated_recall),
                "should_expand_search": bool(result.should_expand_search),
                "feedback": result.feedback,
                "missing_aspects": self._parse_json(result.missing_aspects),
            }

            logger.info(
                f"File discovery evaluation result: "
                f"completeness={evaluation['completeness']:.3f}, "
                f"relevance={evaluation['relevance']:.3f}, "
                f"estimated_recall={evaluation['estimated_recall']:.3f}, "
                f"should_expand={evaluation['should_expand_search']}"
            )

            return evaluation

        except Exception as e:
            logger.error(f"File discovery evaluation failed: {e}", exc_info=True)
            return None

    def _build_files_summary(self, files: List[FileHit], chunks: List[ChunkHit]) -> str:
        """Build file summary with file names, scores, and content previews."""
        if not files:
            return "No files discovered"

        per_file: Dict[str, List[ChunkHit]] = {}
        for chunk in chunks:
            file_name = chunk.document_name
            if file_name not in per_file:
                per_file[file_name] = []
            per_file[file_name].append(chunk)

        summary_parts = []
        for i, file in enumerate(files[:10], 1):
            file_name = getattr(file, "file_name", "") or getattr(file, "title", "")
            score = getattr(file, "score", 0.0)

            file_chunks = per_file.get(file_name, [])[:2]
            chunk_preview = " | ".join(c.content[:100] for c in file_chunks[:2])[:200]

            summary_parts.append(
                f"{i}. {file_name} (score: {score:.3f})\n"
                f"   Preview: {chunk_preview}..."
            )

        if len(files) > 10:
            summary_parts.append(f"\n... and {len(files) - 10} more files")

        return "\n".join(summary_parts)

    def _build_chunks_summary(self, chunks: List[ChunkHit]) -> str:
        """Build chunks summary with score distribution and content."""
        if not chunks:
            return "No chunks retrieved"

        scores = [getattr(c, "score", 0.0) for c in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        top_chunks = sorted(
            chunks, key=lambda c: getattr(c, "score", 0.0), reverse=True
        )[:5]
        top_previews = [
            f"  - Score {getattr(c, 'score', 0.0):.3f}: {c.content[:150]}..."
            for c in top_chunks
        ]

        summary = f"""Chunks Statistics:
- Total chunks: {len(chunks)}
- Score range: {min_score:.3f} - {max_score:.3f} (avg: {avg_score:.3f})

Top Chunks:
{chr(10).join(top_previews)}
"""

        return summary

    def _parse_json(self, text: str) -> List[str]:
        """Parse JSON array from text."""
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            items = text.strip("[]\"'")
            return [item.strip("\"'") for item in items.split(",") if item]

    def should_expand_search(
        self, evaluation: Dict[str, Any], threshold: float = 0.7
    ) -> bool:
        """
        Determine if search should be expanded.

        Args:
            evaluation: Evaluation result
            threshold: Quality threshold (0-1), expand if below this value

        Returns:
            Whether to expand search

        Note:
            Designed to evaluate only once to avoid closed loop.
            If search is expanded, use results directly without re-evaluation.
        """
        if not evaluation:
            return False

        overall_score = (
            evaluation.get("completeness", 1.0)
            + evaluation.get("relevance", 1.0)
            + evaluation.get("coverage", 1.0)
        ) / 3.0

        if overall_score < threshold:
            return True

        if evaluation.get("estimated_recall", 1.0) < threshold:
            return True

        if evaluation.get("should_expand_search", False):
            return True

        return False
