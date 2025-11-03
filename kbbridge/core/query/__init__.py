"""
Query processing module

This module contains components for query analysis, rewriting, keyword generation,
and intention extraction.
"""

from .constants import IntentionExtractorDefaults, KeywordGeneratorDefaults

__all__ = [
    # Constants
    "IntentionExtractorDefaults",
    "KeywordGeneratorDefaults",
    # Keeping __all__ minimal to avoid importing heavy deps at import time
]

# Provide a lightweight 'rewriter' attribute to allow tests to patch
# LLMQueryRewriter without importing the heavy module at import time.
try:  # pragma: no cover - shim for tests
    import types as _types

    class _LLMQRStub:
        def __init__(self, *args, **kwargs):
            pass

    rewriter = _types.SimpleNamespace(LLMQueryRewriter=_LLMQRStub)
except Exception:
    pass
