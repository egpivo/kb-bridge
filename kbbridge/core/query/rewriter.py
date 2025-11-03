import logging
import os
from typing import Optional

import dspy

from .models import RewriteResult, RewriteStrategy

logger = logging.getLogger(__name__)


class QueryAnalysisSignature(dspy.Signature):
    """You are an expert search query analyzer. Analyze queries to determine the best rewrite strategy.

    Available strategies:
    1. EXPANSION - Add synonyms, related terms, and context to broaden search
    2. RELAXATION - Simplify query, remove constraints, use broader terms
    3. NO_CHANGE - Keep original query as-is

    Analysis criteria:
    - Query length and complexity
    - Specificity vs generality
    - Domain-specific terms
    - Search intent (finding specific documents vs exploring topics)"""

    query: str = dspy.InputField(desc="The search query to analyze")
    context: str = dspy.InputField(
        desc="Context about the search domain (e.g., 'General document search')"
    )
    strategy: str = dspy.OutputField(
        desc="Recommended strategy: EXPANSION, RELAXATION, or NO_CHANGE"
    )
    reason: str = dspy.OutputField(
        desc="Brief explanation of why this strategy was chosen"
    )
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    query_analysis: str = dspy.OutputField(
        desc="Brief analysis of the query characteristics"
    )


class QueryExpansionSignature(dspy.Signature):
    """You are an expert at expanding search queries for better document retrieval.

    Task: Expand queries to improve document retrieval by:
    1. Adding relevant synonyms and related terms
    2. Including domain-specific terminology
    3. Adding context that might appear in target documents
    4. Keeping the core meaning intact

    Guidelines:
    - Focus on terms that would appear in legal/clinical documents
    - Include both formal and informal terminology
    - Add related concepts that might be in the same documents
    - Keep the expanded query concise and readable
    - Maximum 1-2 sentences, under 200 characters
    - Avoid overly verbose explanations"""

    query: str = dspy.InputField(desc="Original search query to expand")
    analysis: str = dspy.InputField(desc="Analysis of the query characteristics")
    expanded_query: str = dspy.OutputField(
        desc="Expanded query with synonyms and related terms"
    )


class QueryRelaxationSignature(dspy.Signature):
    """You are an expert at relaxing search queries for broader document retrieval.

    Task: Relax queries to improve document retrieval by:
    1. Removing overly specific constraints
    2. Using broader, more general terms
    3. Simplifying complex phrases
    4. Focusing on core concepts

    Guidelines:
    - Remove specific dates, names, or overly precise terms
    - Use more general legal/clinical terminology
    - Simplify complex sentence structures
    - Keep the main intent clear
    - Maximum 1 sentence, under 150 characters
    - Focus on key terms only"""

    query: str = dspy.InputField(desc="Original search query to relax")
    analysis: str = dspy.InputField(desc="Analysis of the query characteristics")
    relaxed_query: str = dspy.OutputField(
        desc="Relaxed query with broader, simpler terms"
    )


class LLMQueryRewriter(dspy.Module):
    def __init__(
        self,
        llm_api_url: str,
        llm_model: str,
        llm_api_token: Optional[str] = None,
        llm_temperature: float = 0.3,
        llm_timeout: int = 30,
        max_tokens: int = 1000,
        use_cot: bool = False,
    ):
        """Initialize the LLM query rewriter"""
        super().__init__()

        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.llm_api_token = llm_api_token or os.getenv("LLM_API_TOKEN")
        self.llm_temperature = llm_temperature
        self.llm_timeout = llm_timeout
        self.max_tokens = max_tokens

        # Configure DSPy LM
        self._configure_dspy_lm()

        # Initialize predictors
        predictor_class = dspy.ChainOfThought if use_cot else dspy.Predict
        self.analyzer = predictor_class(QueryAnalysisSignature)
        self.expander = predictor_class(QueryExpansionSignature)
        self.relaxer = predictor_class(QueryRelaxationSignature)

    def _configure_dspy_lm(self):
        """Configure DSPy language model with instance-specific settings"""
        if not self.llm_api_token:
            raise ValueError(
                "LLM API token is required (should be provided by config/service layer)"
            )

        self._lm = dspy.LM(
            model=f"openai/{self.llm_model}",
            api_base=self.llm_api_url,
            api_key=self.llm_api_token,
            temperature=self.llm_temperature,
            max_tokens=self.max_tokens,
            timeout=self.llm_timeout,
        )

    def rewrite_query(self, query: str, context: Optional[str] = None) -> RewriteResult:
        """
        Rewrite query using LLM-based strategy selection

        Args:
            query: Original search query
            context: Optional context about the search domain

        Returns:
            RewriteResult with strategy and rewritten query
        """
        try:
            logger.info(f"Rewriting query: '{query[:50]}...'")

            # Analyze query and determine strategy using DSPy
            with dspy.settings.context(lm=self._lm):
                analysis = self.analyzer(
                    query=query, context=context or "General document search"
                )

            # Parse strategy
            strategy_str = (
                analysis.strategy.upper().strip()
                if hasattr(analysis, "strategy")
                else "NO_CHANGE"
            )
            if "EXPANSION" in strategy_str:
                strategy = RewriteStrategy.EXPANSION
            elif "RELAXATION" in strategy_str:
                strategy = RewriteStrategy.RELAXATION
            else:
                strategy = RewriteStrategy.NO_CHANGE

            confidence = (
                float(analysis.confidence) if hasattr(analysis, "confidence") else 0.5
            )
            reason = analysis.reason if hasattr(analysis, "reason") else "Analysis"
            query_analysis = (
                analysis.query_analysis if hasattr(analysis, "query_analysis") else ""
            )

            # If no change needed, return original
            if strategy == RewriteStrategy.NO_CHANGE:
                return RewriteResult(
                    strategy=RewriteStrategy.NO_CHANGE,
                    rewritten_query=query,
                    confidence=confidence,
                    reason=reason,
                    metadata={"query_analysis": query_analysis},
                )

            with dspy.settings.context(lm=self._lm):
                if strategy == RewriteStrategy.EXPANSION:
                    result = self.expander(query=query, analysis=query_analysis)
                    rewritten = (
                        result.expanded_query
                        if hasattr(result, "expanded_query")
                        else query
                    )
                else:
                    result = self.relaxer(query=query, analysis=query_analysis)
                    rewritten = (
                        result.relaxed_query
                        if hasattr(result, "relaxed_query")
                        else query
                    )

            rewrite_result = RewriteResult(
                strategy=strategy,
                rewritten_query=rewritten.strip(),
                confidence=confidence,
                reason=reason,
                metadata={
                    "original_query": query,
                    "query_analysis": query_analysis,
                    "dspy_based": True,
                },
            )

            logger.info(
                f"Query rewritten using {rewrite_result.strategy.value} strategy"
            )
            logger.info(f"   Original: '{query[:50]}...'")
            logger.info(f"   Rewritten: '{rewrite_result.rewritten_query[:50]}...'")

            return rewrite_result

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return RewriteResult(
                strategy=RewriteStrategy.NO_CHANGE,
                rewritten_query=query,
                confidence=0.0,
                reason=f"Rewrite failed: {str(e)}",
                metadata={"error": str(e)},
            )
