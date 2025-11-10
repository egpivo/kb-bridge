from typing import Any, Dict, Optional

import dspy

from .constants import AnswerExtractorDefaults, ResponseMessages


class AnswerExtractionSignature(dspy.Signature):
    """Extract answers from context. Return exactly "N/A" if no relevant information found.

    Modes:
    - File Print: Print file content verbatim when user asks to "show" or "display" a file
    - Table: Extract tables as JSON + summary
    - Comprehensive List: For "all/every/complete" queries, extract ALL items exhaustively
    - Standard Q&A: Answer based on context

    For comprehensive queries: Scan ALL context, extract EVERY item, verify completeness."""

    context: str = dspy.InputField(
        desc="The contextual information (documents, policies, etc.)"
    )
    user_query: str = dspy.InputField(desc="The user's question or request")
    answer: str = dspy.OutputField(
        desc=f"Comprehensive answer based on the context, or '{ResponseMessages.NO_ANSWER}' if no relevant information is found"
    )


class OrganizationAnswerExtractor(dspy.Module):
    """
    Organization AI assistant for document processing with specialized modes
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
        Initialize the answer extractor

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

        # Configure DSPy LM
        self._configure_dspy_lm()

        # Initialize predictor with appropriate signature
        if use_cot:
            self.predictor = dspy.ChainOfThought(AnswerExtractionSignature)
        else:
            self.predictor = dspy.Predict(AnswerExtractionSignature)

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

    def forward(self, context: str, user_query: str) -> dspy.Prediction:
        """
        DSPy forward method for answer extraction

        Args:
            context: The contextual information
            user_query: The user's question

        Returns:
            DSPy Prediction with answer field
        """
        # Use context manager for thread-safe LM configuration
        with dspy.settings.context(lm=self._lm):
            return self.predictor(context=context, user_query=user_query)

    def extract(self, context: str, user_query: str) -> Dict[str, Any]:
        """
        Extract answer using the Organization AI assistant

        Args:
            context: The contextual information (documents, policies, etc.)
            user_query: The user's question or request

        Returns:
            Dict containing the result or error information
        """
        try:
            # Call DSPy forward method
            result = self.forward(context=context, user_query=user_query)

            # Extract answer from DSPy result
            answer = result.answer if hasattr(result, "answer") else ""

            # Return successful response
            return self._build_success_response(answer, user_query, context)

        except Exception as e:
            return self._build_exception_response(e, user_query, context)

    def _build_base_response(self, user_query: str, context: str) -> Dict[str, Any]:
        """Build base response with common fields"""
        return {
            "user_query": user_query,
            "context_length": len(context),
            "model_used": self.llm_model,
            "tool_type": "answer_extractor",
        }

    def _build_success_response(
        self, content: str, user_query: str, context: str
    ) -> Dict[str, Any]:
        """Build successful response"""
        response = self._build_base_response(user_query, context)
        response.update(
            {
                "success": True,
                "answer": content.strip(),
            }
        )
        return response

    def _build_exception_response(
        self, exception: Exception, user_query: str, context: str
    ) -> Dict[str, Any]:
        """Build error response from unexpected exception"""
        response = self._build_base_response(user_query, context)
        response.update(
            {
                "success": False,
                "error": "unknown_error",
                "message": f"Error extracting answer: {str(exception)}",
            }
        )
        return response
