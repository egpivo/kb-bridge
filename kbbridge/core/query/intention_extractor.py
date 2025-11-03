import json
from typing import Any, Dict, List, Optional

import dspy

from .constants import IntentionExtractorDefaults


class UserIntentionSignature(dspy.Signature):
    """Analyze user query intent and document relevance.

    NEVER decompose comprehensive list queries ("all/every/complete/list").
    Only decompose complex multi-aspect queries (comparisons, pros/cons)."""

    user_query: str = dspy.InputField(desc="The user's query to analyze")
    doc_names_json: str = dspy.InputField(
        desc="JSON string of available document names"
    )
    intent: str = dspy.OutputField(desc="Primary goal description")
    information_type: str = dspy.OutputField(desc="Type of information needed")
    relevant_documents: List[str] = dspy.OutputField(
        desc="List of relevant document names"
    )
    query_complexity: str = dspy.OutputField(
        desc="Query complexity: simple, moderate, or complex"
    )
    suggested_approach: str = dspy.OutputField(desc="Recommended search strategy")
    should_decompose: bool = dspy.OutputField(
        desc="Whether the query should be decomposed into sub-queries"
    )
    sub_queries: List[str] = dspy.OutputField(
        desc="Focused sub-queries (empty list if should_decompose is false)"
    )
    updated_query: str = dspy.OutputField(
        desc="Cleaned query with filename references removed"
    )


class UserIntentionExtractor(dspy.Module):
    """
    Query cleaning service that removes filename references from user queries
    """

    def __init__(
        self,
        llm_api_url: str,
        llm_model: str,
        llm_api_token: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        use_cot: bool = False,
    ):
        """
        Initialize the user intention extractor

        Args:
            llm_api_url: LLM API service URL
            llm_model: LLM model name
            llm_api_token: Optional API token (falls back to default if None)
            llm_temperature: Optional temperature (falls back to default if None)
            llm_timeout: Optional timeout in seconds (falls back to default if None)
            max_tokens: Optional max tokens (falls back to default if None)
            use_cot: Whether to use Chain of Thought reasoning (default: False)
        """
        super().__init__()

        self.llm_api_url = llm_api_url
        self.llm_model = llm_model

        # Use provided values or fall back to defaults
        self.llm_api_token = llm_api_token
        self.llm_temperature = (
            llm_temperature
            if llm_temperature is not None
            else IntentionExtractorDefaults.TEMPERATURE.value
        )
        self.llm_timeout = (
            llm_timeout
            if llm_timeout is not None
            else IntentionExtractorDefaults.TIMEOUT_SECONDS.value
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else IntentionExtractorDefaults.MAX_TOKENS.value
        )

        # Configure DSPy LM
        self._configure_dspy_lm()

        # Initialize predictor with appropriate signature
        if use_cot:
            self.predictor = dspy.ChainOfThought(UserIntentionSignature)
        else:
            self.predictor = dspy.Predict(UserIntentionSignature)

    def _configure_dspy_lm(self):
        """Configure DSPy language model with instance-specific settings"""
        if not self.llm_api_token:
            raise ValueError(
                "LLM API token is required (should be provided by config/service layer)"
            )

        # Create a local LM instance (not global configuration)
        self._lm = dspy.LM(
            model=f"openai/{self.llm_model}",
            api_base=self.llm_api_url,
            api_key=self.llm_api_token,
            temperature=self.llm_temperature,
            max_tokens=self.max_tokens,
            timeout=self.llm_timeout,
        )

    def forward(self, user_query: str, doc_names_json: str) -> dspy.Prediction:
        """
        DSPy forward method for user intention extraction

        Args:
            user_query: The user's query to analyze
            doc_names_json: JSON string of available document names

        Returns:
            DSPy Prediction with intention analysis fields
        """
        # Use context manager for thread-safe LM configuration
        with dspy.settings.context(lm=self._lm):
            return self.predictor(user_query=user_query, doc_names_json=doc_names_json)

    def extract_intention(
        self, user_query: str, doc_names: List[str]
    ) -> Dict[str, Any]:
        """
        Clean user query by removing filename references

        Args:
            user_query: The original user query
            doc_names: List of document names to remove from the query

        Returns:
            Dict containing the result or error information
        """
        try:
            # Format document names as JSON array string
            doc_names_json = json.dumps(doc_names)

            # Call DSPy forward method
            result = self.forward(user_query=user_query, doc_names_json=doc_names_json)

            # Extract fields from DSPy result
            updated_query = (
                result.updated_query if hasattr(result, "updated_query") else user_query
            )
            should_decompose = (
                result.should_decompose
                if hasattr(result, "should_decompose")
                else False
            )
            sub_queries = result.sub_queries if hasattr(result, "sub_queries") else []

            # Return successful response
            response = self._build_success_response(
                updated_query, user_query, doc_names, should_decompose, sub_queries
            )
            response["debug_details"] = ["DSPy processing successful"]
            return response

        except Exception as e:
            return self._build_exception_response(str(e), user_query, doc_names)

    def _build_base_response(
        self, user_query: str, doc_names: List[str]
    ) -> Dict[str, Any]:
        """Build base response with common fields"""
        return {
            "original_query": user_query,
            "doc_names": doc_names,
            "model_used": self.llm_model,
            "tool_type": "user_intention_extractor",
        }

    def _build_success_response(
        self,
        updated_query: str,
        user_query: str,
        doc_names: List[str],
        should_decompose: bool = False,
        sub_queries: List[str] = None,
    ) -> Dict[str, Any]:
        """Build successful response"""
        if sub_queries is None:
            sub_queries = []
        response = self._build_base_response(user_query, doc_names)
        response.update(
            {
                "success": True,
                "updated_query": updated_query.strip(),
                "should_decompose": should_decompose,
                "sub_queries": sub_queries,
            }
        )
        return response

    def _build_exception_response(
        self, exception_msg: str, user_query: str, doc_names: List[str]
    ) -> Dict[str, Any]:
        """Build error response from unexpected exception"""
        response = self._build_base_response(user_query, doc_names)
        response.update(
            {
                "success": False,
                "error": "unknown_error",
                "message": f"Error extracting user intention: {exception_msg}",
                "debug_details": [f"Exception occurred: {exception_msg}"],
            }
        )
        return response
