import logging
from typing import Any

import dspy

logger = logging.getLogger(__name__)


def setup(
    llm_model: str, llm_api_url: str, api_key: str, temperature: float = 0.0
) -> dspy.LM:
    """Configure DSPy with LLM settings for reflection."""
    lm = dspy.LM(
        model=llm_model,
        api_base=llm_api_url,
        api_key=api_key,
        temperature=temperature,
    )
    logger.info(f"DSPy configured: model={llm_model}")
    return lm


def get_lm() -> Any:
    """Get configured DSPy language model."""
    return dspy.settings.lm
