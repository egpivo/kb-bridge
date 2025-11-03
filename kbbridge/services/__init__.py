"""
KB Bridge Services Package

High-level service modules for knowledge base interaction.

Note: We intentionally avoid re-exporting callables with the same names as
their modules (e.g., file_discover_service) to prevent import/patching
collisions in tests that target module paths such as
"kbbridge.services.file_discover_service.RetrievalCredentials".
Import services from their modules instead, e.g.:

from kbbridge.services.retriever_service import retriever_service
"""

__all__ = []
