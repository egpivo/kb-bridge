from typing import Any, Dict, List
from urllib.parse import unquote

from kbbridge.core.synthesis.answer_formatter import StructuredAnswerFormatter
from kbbridge.core.synthesis.answer_reranker import AnswerReranker
from kbbridge.core.synthesis.constants import ResponseMessages

from .models import Credentials


class ResultFormatter:
    """Formats results for output"""

    @staticmethod
    def format_final_answer(
        candidates: List[Dict[str, Any]], query: str, credentials: Credentials
    ) -> str:
        """Format the best answer from candidates"""
        if not candidates:
            return ResponseMessages.NO_ANSWER_WITH_CONTEXT

        if len(candidates) == 1:
            return ResultFormatter._format_single_candidate(candidates[0])

        # Multiple candidates - try reranking if available
        if credentials.rerank_url and credentials.rerank_model:
            reranker = AnswerReranker(credentials.rerank_url, credentials.rerank_model)
            rerank_result = reranker.rerank_answers(query, candidates)
            if rerank_result.get("final_result"):
                return rerank_result["final_result"]

        # For multiple candidates, combine them intelligently
        return ResultFormatter._combine_candidates(candidates, query)

    @staticmethod
    def _combine_candidates(candidates: List[Dict[str, Any]], query: str) -> str:
        """Combine multiple candidates into a comprehensive answer"""
        if not candidates:
            return ResponseMessages.NO_ANSWER_WITH_CONTEXT

        # Prefer successful, non-empty candidates (excluding trivial N/A)
        def is_nontrivial(c: Dict[str, Any]) -> bool:
            ans = (c.get("answer", "") or "").strip()
            return bool(ans) and ans.upper() != ResponseMessages.NO_ANSWER

        successful_candidates = [
            c for c in candidates if c.get("success", False) and is_nontrivial(c)
        ]

        if len(successful_candidates) > 1:
            # Combine all successful answers
            all_answers = [c["answer"] for c in successful_candidates]
            combined = "\n\n---\n\n".join(all_answers)
            return combined

        if len(successful_candidates) == 1:
            return ResultFormatter._format_single_candidate(successful_candidates[0])

        # No successful candidates: pick first non-empty answer to preserve order
        nonempty = [c for c in candidates if is_nontrivial(c)]
        if nonempty:
            return ResultFormatter._format_single_candidate(nonempty[0])

        # Still nothing usable
        return ResponseMessages.NO_ANSWER_WITH_CONTEXT

    @staticmethod
    def _format_single_candidate(candidate: Dict[str, Any]) -> str:
        """Format a single candidate answer"""
        if candidate["source"] == "direct":
            return candidate["answer"]
        else:
            resource_id = candidate.get("resource_id") or candidate.get(
                "dataset_id", ""
            )
            file_name = candidate.get("file_name", "")
            answer = candidate["answer"]

            if file_name:
                try:
                    display_name = unquote(file_name)
                except Exception:
                    display_name = file_name
                return f"**{resource_id}/{display_name}**: {answer}"
            else:
                return f"**{resource_id}**: {answer}"

    @staticmethod
    def format_structured_answer(
        candidates: List[Dict[str, Any]], query: str, credentials: Credentials
    ) -> Dict[str, Any]:
        """Format candidates into a structured answer using LLM"""
        if not candidates:
            return {
                "success": False,
                "error": "No candidates found",
                "details": "No valid candidates were generated from the search",
            }

        # Create structured answer formatter with reranking support
        formatter = StructuredAnswerFormatter(
            llm_api_url=credentials.llm_api_url,
            llm_model=credentials.llm_model,
            llm_api_token=credentials.llm_api_token,
            max_tokens=12800,
            rerank_url=credentials.rerank_url,
            rerank_model=credentials.rerank_model,
        )

        # Format the structured answer
        result = formatter.format_structured_answer(query, candidates)
        return result
