import json
from typing import TYPE_CHECKING, Dict, List, Union

import requests

from .constants import ResponseMessages

if TYPE_CHECKING:
    from kbbridge.core.orchestration.models import CandidateAnswer


class AnswerReranker:
    """
    Handles reranking of candidate answers using external reranking services.

    This class is backend-agnostic and works with any RAG system that produces
    candidate answers with standard metadata (resource_id, file_name).
    """

    def __init__(self, rerank_url: str, rerank_model: str):
        """
        Initialize the answer reranker.

        Args:
            rerank_url: URL of the reranking service
            rerank_model: Model to use for reranking
        """
        self.rerank_url = rerank_url
        self.rerank_model = rerank_model

    def rerank_answers(
        self,
        query: str,
        candidate_answers: List[Union["CandidateAnswer", dict]],
        timeout: int = 30,
    ) -> Dict:
        """
        Rerank candidate answers to find the most relevant one.

        Args:
            query: The user query
            candidate_answers: List of CandidateAnswer objects or dicts
            timeout: Request timeout in seconds

        Returns:
            Dict with final_result and detailed_results
        """
        try:
            # 1. Normalize inputs
            candidates = self._normalize_candidates(candidate_answers)

            # 2. Extract valid answers
            valid_candidates = [c for c in candidates if self._is_valid_answer(c)]
            if not valid_candidates:
                return {"final_result": "", "detailed_results": []}

            # 3. Call reranking service
            rerank_results = self._call_reranking_service(
                query, [c.answer for c in valid_candidates], timeout
            )

            # 4. Map rerank scores back to candidates
            ranked_candidates = self._map_scores_to_candidates(
                rerank_results, valid_candidates
            )

            # 5. Format best result
            if ranked_candidates:
                best = ranked_candidates[0]["candidate_answer"]
                final_result = self._format_candidate(best)
                return {
                    "final_result": final_result,
                    "detailed_results": ranked_candidates,
                }

            return {"final_result": "", "detailed_results": []}

        except Exception as e:
            return {"final_result": "", "detailed_results": [], "rerank_error": str(e)}

    def _normalize_candidates(
        self, candidates: List[Union["CandidateAnswer", dict]]
    ) -> List["CandidateAnswer"]:
        """Convert all candidates to CandidateAnswer objects."""
        from kbbridge.core.orchestration.models import CandidateAnswer

        return [
            CandidateAnswer.from_dict(c) if isinstance(c, dict) else c
            for c in candidates
        ]

    def _is_valid_answer(self, candidate: "CandidateAnswer") -> bool:
        """Check if candidate has a valid answer."""
        return (
            candidate.success
            and candidate.answer
            and candidate.answer != ResponseMessages.NO_ANSWER
        )

    def _call_reranking_service(
        self, query: str, documents: List[str], timeout: int
    ) -> Dict:
        """Call external reranking service."""
        payload = {
            "query": query,
            "documents": documents,
            "return_documents": False,
            "model": self.rerank_model,
        }
        response = requests.post(
            self.rerank_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False),
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def _map_scores_to_candidates(
        self, rerank_results: Dict, candidates: List["CandidateAnswer"]
    ) -> List[Dict]:
        """Map reranking scores back to original candidates."""
        scored = []
        for result in rerank_results.get("results", []):
            idx = result.get("index")
            if idx is not None and 0 <= idx < len(candidates):
                scored.append(
                    {
                        "index": idx,
                        "relevance_score": result.get("relevance_score"),
                        "candidate_answer": candidates[idx],
                        "document": candidates[idx].answer,
                    }
                )
        # Sort by score (desc), then index (asc) for stability
        return sorted(scored, key=lambda x: (-x["relevance_score"], x["index"]))

    def _format_candidate(self, candidate: "CandidateAnswer") -> str:
        """Format candidate answer with source citation."""
        if candidate.source == "direct":
            return candidate.answer

        # Advanced answer with citation
        dataset = candidate.resource_id or "Unknown dataset"
        if candidate.file_name:
            return f"**{dataset}/{candidate.file_name}**: {candidate.answer}"
        else:
            return f"**{dataset}**: {candidate.answer}"
