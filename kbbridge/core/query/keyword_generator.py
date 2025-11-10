import json
import logging
from typing import Any, Dict, List, Optional

import dspy

from kbbridge.core.utils.json_utils import parse_json_from_markdown

from .constants import KeywordGeneratorDefaults

logger = logging.getLogger(__name__)


class FileSearchKeywordSignature(dspy.Signature):
    """Generate diverse keyword sets for file discovery and document search.

    This signature is optimized for finding FILES by name and content, focusing on:
    - File name matching and document titles
    - Document type detection (agreement, policy, procedure, etc.)
    - Entity recognition (organizations, companies, specific names)
    - Domain-specific terminology and abbreviations
    """

    query: str = dspy.InputField(
        desc="User's search query for finding relevant files and documents"
    )
    max_sets: int = dspy.InputField(
        desc="Maximum number of distinct keyword sets to generate"
    )

    keyword_sets: List[List[str]] = dspy.OutputField(
        desc=(
            "Array of keyword arrays. Each inner array is a distinct search strategy. "
            "Focus on terms likely to appear in file names, document titles, and content. "
            "Include entity names, document types, domain terminology, and abbreviations. "
            'Return as JSON: [["kw1", "kw2"], ["kw3", "kw4"], ...]'
        )
    )


class ContentBoostKeywordSignature(dspy.Signature):
    """Generate complementary keyword sets for content boosting within a specific document.

    This signature generates keywords that:
    - Target content WITHIN an already-identified document
    - Complement file-search keywords (avoid duplication)
    - Stay aligned with user's query intent
    - Vary by category/type while maintaining focus
    """

    query: str = dspy.InputField(
        desc="User's search query for content within the document"
    )
    document_name: str = dspy.InputField(
        desc="Name of the specific document being searched"
    )
    max_sets: int = dspy.InputField(
        desc="Maximum number of distinct keyword sets to generate"
    )
    file_search_keywords: str = dspy.InputField(
        desc=(
            "Comma-separated keywords already used by file search. "
            "Generate complementary keywords that avoid exact repetition "
            "but stay aligned with the query intent."
        ),
        default="",
    )
    custom_instructions: str = dspy.InputField(
        desc="Optional domain-specific guidance for keyword generation",
        default="",
    )

    keyword_sets: List[List[str]] = dspy.OutputField(
        desc=(
            "Array of keyword arrays representing complementary search strategies. "
            "Stay aligned with user intent while varying categories/types. "
            "Avoid repeating file_search_keywords exactly. "
            'Return as JSON: [["kw1", "kw2"], ["kw3", "kw4"], ...]'
        )
    )


class KeywordGenerator(dspy.Module):
    """DSPy-based keyword generator using Signatures for structured generation."""

    def __init__(
        self,
        llm_api_url: str,
        llm_model: str,
        llm_api_token: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        prompt_type: str = "file_search",
        use_cot: bool = False,
    ):
        """Initialize the keyword generator.

        Args:
            llm_api_url: LLM API service URL
            llm_model: LLM model name
            llm_api_token: Optional API token
            llm_temperature: Temperature (default from constants)
            llm_timeout: Timeout in seconds (default from constants)
            max_tokens: Max tokens (default from constants)
            prompt_type: "file_search" or "content_boosting"
            use_cot: Whether to use ChainOfThought (default: False for speed)
        """
        super().__init__()

        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.llm_api_token = llm_api_token
        self.prompt_type = prompt_type

        # Use provided values or fall back to defaults
        self.llm_temperature = (
            llm_temperature
            if llm_temperature is not None
            else KeywordGeneratorDefaults.TEMPERATURE.value
        )
        self.llm_timeout = (
            llm_timeout
            if llm_timeout is not None
            else KeywordGeneratorDefaults.TIMEOUT_SECONDS.value
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else KeywordGeneratorDefaults.MAX_TOKENS.value
        )

        # Select signature and predictor based on prompt type
        if prompt_type == "file_search":
            signature = FileSearchKeywordSignature
        elif prompt_type == "content_boosting":
            signature = ContentBoostKeywordSignature
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        # Configure DSPy LM (creates self._lm)
        self._configure_dspy_lm()

        # Use Predict (faster) or ChainOfThought (reasoning)
        if use_cot:
            self.predictor = dspy.ChainOfThought(signature)
        else:
            self.predictor = dspy.Predict(signature)

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

    def forward(
        self,
        query: str,
        max_sets: int,
        document_name: str = "",
        file_search_keywords: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
    ) -> List[List[str]]:
        """Generate keyword sets using DSPy (DSPy Module forward method).

        Args:
            query: User's search query
            max_sets: Maximum number of keyword sets
            document_name: Document name (for content_boosting)
            file_search_keywords: Keywords already used by file search
            custom_instructions: Domain-specific guidance

        Returns:
            List of keyword sets (list of string lists)
        """
        # Use context manager for thread-safe LM configuration
        with dspy.settings.context(lm=self._lm):
            # Build inputs based on prompt type
            if self.prompt_type == "file_search":
                result = self.predictor(query=query, max_sets=max_sets)
            else:  # content_boosting
                # Format file_search_keywords as comma-separated string
                fsk_str = ", ".join(file_search_keywords or [])
                result = self.predictor(
                    query=query,
                    max_sets=max_sets,
                    document_name=document_name or "",
                    file_search_keywords=fsk_str,
                    custom_instructions=custom_instructions or "",
                )

        # Extract keyword_sets from result
        keyword_sets = getattr(result, "keyword_sets", None)

        # If keyword_sets is a string (LLM returned JSON), parse it
        if isinstance(keyword_sets, str):
            keyword_sets = self._parse_keyword_sets(keyword_sets)

        # Validate and return
        if not isinstance(keyword_sets, list):
            raise ValueError(f"Expected list of keyword sets, got {type(keyword_sets)}")

        return keyword_sets

    def generate(
        self,
        query: str,
        max_sets: int,
        document_name: Optional[str] = None,
        file_search_keywords: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate keyword sets (convenience method with dict response).

        Args:
            query: User's search query
            max_sets: Maximum number of keyword sets
            document_name: Document name (for content_boosting)
            file_search_keywords: Keywords already used by file search
            custom_instructions: Domain-specific guidance

        Returns:
            Dict with success status, keyword_sets, and metadata
        """
        try:
            keyword_sets = self(
                query=query,
                max_sets=max_sets,
                document_name=document_name or "",
                file_search_keywords=file_search_keywords,
                custom_instructions=custom_instructions,
            )

            return self._build_success_response(
                query, max_sets, keyword_sets, document_name
            )

        except Exception as e:
            logger.exception("Keyword generation failed: %s", e)
            return self._build_error_response(
                query, max_sets, "generation_error", str(e), document_name
            )

    @staticmethod
    def _parse_keyword_sets(content: str) -> List[List[str]]:
        """Parse keyword sets from LLM response (handles JSON or markdown)."""
        try:
            # Try direct JSON parsing
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try markdown extraction
        try:
            result = parse_json_from_markdown(content)
            return result["result"]
        except (ValueError, json.JSONDecodeError):
            pass

        # Last resort: return empty list
        logger.warning("Failed to parse keyword sets from: %s", content[:200])
        return []

    def _build_success_response(
        self,
        query: str,
        max_sets: int,
        keyword_sets: List[List[str]],
        document_name: Optional[str],
    ) -> Dict[str, Any]:
        """Build successful response."""
        response = {
            "success": True,
            "query": query,
            "max_sets": max_sets,
            "keyword_sets": keyword_sets,
            "total_sets": len(keyword_sets),
            "model_used": self.llm_model,
            "prompt_type": self.prompt_type,
        }
        if document_name:
            response["document_name"] = document_name
        return response

    def _build_error_response(
        self,
        query: str,
        max_sets: int,
        error: str,
        message: str,
        document_name: Optional[str],
    ) -> Dict[str, Any]:
        """Build error response."""
        response = {
            "success": False,
            "query": query,
            "max_sets": max_sets,
            "error": error,
            "message": message,
            "model_used": self.llm_model,
            "prompt_type": self.prompt_type,
        }
        if document_name:
            response["document_name"] = document_name
        return response


def create_keyword_generator(
    prompt_type: str,
    llm_api_url: str,
    llm_model: str,
    llm_api_token: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    llm_timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_cot: bool = False,
    **kwargs: Any,
) -> KeywordGenerator:
    """Factory function to create a keyword generator.

    Args:
        prompt_type: "file_search" or "content_boosting"
        llm_api_url: LLM API service URL
        llm_model: LLM model name
        llm_api_token: Optional API token
        llm_temperature: Optional temperature
        llm_timeout: Optional timeout
        max_tokens: Optional max tokens
        use_cot: Whether to use ChainOfThought
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        KeywordGenerator instance
    """
    return KeywordGenerator(
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        llm_api_token=llm_api_token,
        llm_temperature=llm_temperature,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
        prompt_type=prompt_type,
        use_cot=use_cot,
    )


def generate_file_search_keywords(
    query: str,
    max_sets: int,
    llm_api_url: str,
    llm_model: str,
    **generator_kwargs: Any,
) -> Dict[str, Any]:
    """Generate keyword sets for file discovery/search."""
    generator = KeywordGenerator(
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        prompt_type="file_search",
        **{
            k: v
            for k, v in generator_kwargs.items()
            if k
            in [
                "llm_api_token",
                "llm_temperature",
                "llm_timeout",
                "max_tokens",
                "use_cot",
            ]
        },
    )
    return generator.generate(query, max_sets)


def generate_content_boosting_keywords(
    query: str,
    document_name: str,
    max_sets: int,
    llm_api_url: str,
    llm_model: str,
    **generator_kwargs: Any,
) -> Dict[str, Any]:
    """Generate keyword sets for content boosting/query rephrasing."""
    generator = KeywordGenerator(
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        prompt_type="content_boosting",
        **{
            k: v
            for k, v in generator_kwargs.items()
            if k
            in [
                "llm_api_token",
                "llm_temperature",
                "llm_timeout",
                "max_tokens",
                "use_cot",
            ]
        },
    )
    return generator.generate(query, max_sets, document_name=document_name)


def generate_keywords(
    query: str,
    max_sets: int,
    llm_api_url: str,
    llm_model: str,
    prompt_type: str = "file_search",
    document_name: Optional[str] = None,
    llm_api_token: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    llm_timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
    file_search_keywords: Optional[List[str]] = None,
    custom_instructions: Optional[str] = None,
    **generate_kwargs: Any,
) -> Dict[str, Any]:
    """Unified keyword generation function (backward compatibility).

    Args:
        query: User's search query
        max_sets: Maximum number of keyword sets
        llm_api_url: LLM API URL
        llm_model: LLM model name
        prompt_type: "file_search" or "content_boosting"
        document_name: Document name (for content_boosting)
        llm_api_token: Optional API token
        llm_temperature: Optional temperature
        llm_timeout: Optional timeout
        max_tokens: Optional max tokens
        file_search_keywords: Keywords already used by file search
        custom_instructions: Domain-specific guidance
        **generate_kwargs: Additional generation arguments (ignored)

    Returns:
        Dict with success status, keyword_sets, and metadata
    """
    generator = create_keyword_generator(
        prompt_type=prompt_type,
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        llm_api_token=llm_api_token,
        llm_temperature=llm_temperature,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
    )

    return generator.generate(
        query=query,
        max_sets=max_sets,
        document_name=document_name,
        file_search_keywords=file_search_keywords,
        custom_instructions=custom_instructions,
    )


def FileSearchKeywordGenerator(
    llm_api_url: str,
    llm_model: str,
    llm_api_token: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    llm_timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_cot: bool = False,
) -> KeywordGenerator:
    """Create a KeywordGenerator configured for file search.

    This is a factory function that returns a KeywordGenerator with
    prompt_type='file_search' pre-configured.

    Args:
        llm_api_url: LLM API service URL
        llm_model: LLM model name
        llm_api_token: Optional API token
        llm_temperature: Optional temperature
        llm_timeout: Optional timeout
        max_tokens: Optional max tokens
        use_cot: Whether to use ChainOfThought

    Returns:
        KeywordGenerator configured for file search
    """
    return KeywordGenerator(
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        llm_api_token=llm_api_token,
        llm_temperature=llm_temperature,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
        prompt_type="file_search",
        use_cot=use_cot,
    )


def ContentBoostKeywordGenerator(
    llm_api_url: str,
    llm_model: str,
    llm_api_token: Optional[str] = None,
    llm_temperature: Optional[float] = None,
    llm_timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_cot: bool = False,
) -> KeywordGenerator:
    """Create a KeywordGenerator configured for content boosting.

    This is a factory function that returns a KeywordGenerator with
    prompt_type='content_boosting' pre-configured.

    Args:
        llm_api_url: LLM API service URL
        llm_model: LLM model name
        llm_api_token: Optional API token
        llm_temperature: Optional temperature
        llm_timeout: Optional timeout
        max_tokens: Optional max tokens
        use_cot: Whether to use ChainOfThought

    Returns:
        KeywordGenerator configured for content boosting
    """
    return KeywordGenerator(
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        llm_api_token=llm_api_token,
        llm_temperature=llm_temperature,
        llm_timeout=llm_timeout,
        max_tokens=max_tokens,
        prompt_type="content_boosting",
        use_cot=use_cot,
    )
