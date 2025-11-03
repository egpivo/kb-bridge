"""
Discovery module

This module contains components for discovering and retrieving relevant files
and documents from knowledge bases.
"""

from .file_reranker import rerank_documents, rerank_files_by_names

__all__ = [
    "rerank_documents",
    "rerank_files_by_names",
]

# Lightweight shim to allow tests to patch
# "kbbridge.core.discovery.file_discover.FileDiscover" without importing
# the heavy DSPy-dependent module at import time.
try:  # pragma: no cover - for test-time patching
    import types as _types

    file_discover = _types.SimpleNamespace(FileDiscover=object)
except Exception:
    pass
