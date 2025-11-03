"""Orchestration: lightweight exports for models and core services.

This package intentionally avoids importing heavy modules (e.g., DSPy-based
implementations) at import time so tests can patch symbols like
`kbbridge.core.orchestration.ComponentFactory` without requiring optional
dependencies.
"""

from kbbridge.core.utils.profiling_utils import profile_stage  # noqa: F401

from .models import *  # noqa: F401,F403
from .services import (  # noqa: F401
    ComponentFactory,
    CredentialParser,
    ParameterValidator,
    WorkerDistributor,
)

__all__ = [
    # Models (wildcard via models.__all__ if present)
    # Core services
    "ComponentFactory",
    "ParameterValidator",
    "WorkerDistributor",
    "CredentialParser",
    # Utilities
    "profile_stage",
]

# Expose a lightweight 'utils' attribute for patching in tests without importing
# heavy DSPy-dependent modules at import time.
try:  # pragma: no cover - shim for tests
    import types as _types

    class _ResultFormatterStub:
        @staticmethod
        def format_final_answer(candidates, query, credentials):
            return ""

        @staticmethod
        def format_structured_answer(query, candidates):
            return {"success": False}

    utils = _types.SimpleNamespace(ResultFormatter=_ResultFormatterStub)
except Exception:  # pragma: no cover
    pass

# Provide placeholders for helper functions so tests can patch
def _rewrite_query(*args, **kwargs):  # pragma: no cover - replaced by tests
    raise NotImplementedError


def _extract_intention(*args, **kwargs):  # pragma: no cover - replaced by tests
    raise NotImplementedError


# Lightweight stub to allow tests to patch kbbridge.core.orchestration.DatasetProcessor
# without importing the heavy pipeline module at import-time.
class DatasetProcessor:  # pragma: no cover - replaced by tests via patching
    pass
