import re
from typing import Any, Dict, Optional

import dspy

from kbbridge.config.constants import ParagraphLocatorDefaults


class ParagraphLocationSignature(dspy.Signature):
    """Locate the key paragraph and specific term within a document that best addresses the user's query.

    Your task is to identify the most relevant section (anchor_section) and specific term (anchor_term)
    within the document content that directly relates to the query.

    Instructions:
    1. Read the entire document carefully
    2. Identify the paragraph or section most relevant to the query
    3. Extract a distinctive phrase or sentence from that section (anchor_section)
    4. Identify the most specific term within that section (anchor_term)
    5. Format your response as: "anchor_section — anchor_term"
    6. If no relevant content is found, return an empty string

    The anchor_section should be a distinctive excerpt (10-50 words) that can be uniquely found in the document.
    The anchor_term should be a key phrase (1-5 words) within that section that directly relates to the query.
    """

    document_content: str = dspy.InputField(
        desc="The full document text to search within"
    )
    query: str = dspy.InputField(desc="The user's query to locate in the document")
    location: str = dspy.OutputField(
        desc="The location formatted as 'anchor_section — anchor_term', or empty string if not found"
    )


class ParagraphLocator(dspy.Module):
    """
    Locates key paragraphs and terms within documents using LLM analysis
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
        Initialize the paragraph locator with DSPy

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
        self.llm_api_token = llm_api_token

        # Use provided values or fall back to defaults
        self.llm_temperature = (
            llm_temperature
            if llm_temperature is not None
            else ParagraphLocatorDefaults.TEMPERATURE.value
        )
        self.llm_timeout = (
            llm_timeout
            if llm_timeout is not None
            else ParagraphLocatorDefaults.TIMEOUT_SECONDS.value
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else ParagraphLocatorDefaults.MAX_TOKENS.value
        )

        # Configure DSPy LM (creates self._lm)
        self._configure_dspy_lm()

        # Use Predict (faster) or ChainOfThought (reasoning)
        if use_cot:
            self.predictor = dspy.ChainOfThought(ParagraphLocationSignature)
        else:
            self.predictor = dspy.Predict(ParagraphLocationSignature)

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

        # Only pass api_key if it's not None (some endpoints don't need auth)
        if self.llm_api_token:
            lm_kwargs["api_key"] = self.llm_api_token

        # Create LM instance - will be used with context manager for thread safety
        self._lm = dspy.LM(**lm_kwargs)

    def forward(self, document_content: str, query: str) -> str:
        """DSPy Module forward method.

        Args:
            document_content: The full document text to search
            query: The user's query to locate

        Returns:
            Location string formatted as "anchor_section — anchor_term"
        """
        # Use context manager for thread-safe LM configuration
        with dspy.settings.context(lm=self._lm):
            result = self.predictor(document_content=document_content, query=query)

        return result.location

    def _parse_location_response(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract anchor section and term

        Returns:
            Dict with 'success', 'anchor_section', 'anchor_term', and optionally 'error'/'raw_response'
        """
        try:
            # Clean up the response - remove any markdown symbols, newlines, extra spaces
            cleaned_content = content.strip()
            cleaned_content = re.sub(
                r"```[^`]*```", "", cleaned_content
            )  # Remove code blocks
            cleaned_content = re.sub(
                r"[`\n\r]", "", cleaned_content
            )  # Remove backticks and newlines
            cleaned_content = cleaned_content.strip()

            # Check if empty result (no match found)
            if (
                not cleaned_content
                or cleaned_content == '""'
                or cleaned_content == "''"
            ):
                return {
                    "success": True,
                    "anchor_section": "",
                    "anchor_term": "",
                    "found_match": False,
                }

            # Try to split on the em dash (—) or regular dash (-)
            if "—" in cleaned_content:
                parts = cleaned_content.split("—", 1)
            elif "-" in cleaned_content:
                parts = cleaned_content.split("-", 1)
            else:
                # If no separator found, treat the whole thing as anchor_section
                return {
                    "success": True,
                    "anchor_section": cleaned_content,
                    "anchor_term": "",
                    "found_match": True,
                }

            if len(parts) == 2:
                anchor_section = parts[0].strip()
                anchor_term = parts[1].strip()

                return {
                    "success": True,
                    "anchor_section": anchor_section,
                    "anchor_term": anchor_term,
                    "found_match": True,
                }
            else:
                # Fallback: treat the whole content as anchor_section
                return {
                    "success": True,
                    "anchor_section": cleaned_content,
                    "anchor_term": "",
                    "found_match": True,
                }

        except Exception as e:
            return {
                "success": False,
                "error": "parse_error",
                "raw_response": content,
                "message": f"Error parsing location response: {str(e)}",
            }

    def locate(self, document_content: str, query: str) -> Dict[str, Any]:
        """
        Locate the key paragraph and term for a query within document content

        Args:
            document_content: The full document text to search
            query: The user's query to locate

        Returns:
            Dict containing the result or error information
        """
        try:
            # Call DSPy module (uses forward method)
            location = self(document_content=document_content, query=query)

            # Parse the response
            parse_result = self._parse_location_response(location)
            if parse_result["success"]:
                return self._build_success_response(parse_result, query)
            else:
                return self._build_parse_error_response(parse_result, query)

        except Exception as e:
            return self._build_exception_response(e, query)

    def _build_base_response(self, query: str) -> Dict[str, Any]:
        """Build base response with common fields"""
        return {
            "query": query,
            "model_used": self.llm_model,
            "tool_type": "paragraph_locator",
        }

    def _build_success_response(self, parse_result: Dict, query: str) -> Dict[str, Any]:
        """Build successful response"""
        response = self._build_base_response(query)
        response.update(
            {
                "success": True,
                "anchor_section": parse_result["anchor_section"],
                "anchor_term": parse_result["anchor_term"],
                "found_match": parse_result["found_match"],
            }
        )

        # Include formatted result for convenience
        if parse_result["found_match"] and parse_result["anchor_section"]:
            if parse_result["anchor_term"]:
                response[
                    "formatted_result"
                ] = f"{parse_result['anchor_section']}—{parse_result['anchor_term']}"
            else:
                response["formatted_result"] = parse_result["anchor_section"]
        else:
            response["formatted_result"] = ""

        return response

    def _build_parse_error_response(
        self, parse_result: Dict, query: str
    ) -> Dict[str, Any]:
        """Build error response from parsing failure"""
        response = self._build_base_response(query)
        response.update(
            {
                "success": False,
                "error": parse_result["error"],
                "message": parse_result.get(
                    "message", "Failed to parse location response"
                ),
                "raw_response": parse_result["raw_response"],
            }
        )
        return response

    def _build_exception_response(
        self, exception: Exception, query: str
    ) -> Dict[str, Any]:
        """Build error response from unexpected exception"""
        response = self._build_base_response(query)
        response.update(
            {
                "success": False,
                "error": "unknown_error",
                "message": f"Error locating paragraph: {str(exception)}",
            }
        )
        return response
