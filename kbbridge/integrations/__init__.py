"""
QA Hub Integrations Package

Provides backend adapters for different retrieval systems (Dify, OpenSearch, etc.)
"""

from .backend_adapter import BackendAdapter, BackendAdapterFactory
from .credentials import RetrievalCredentials
from .dify.dify_backend_adapter import DifyBackendAdapter, create_dify_adapter
from .dify.dify_credentials import DifyCredentials, validate_dify_credentials
from .dify.dify_retriever import DifyRetriever
from .retriever_base import ChunkHit, FileHit, Retriever
from .retriever_router import RetrieverRouter, create_retriever_from_env, make_retriever

__all__ = [
    "BackendAdapter",
    "BackendAdapterFactory",
    "RetrievalCredentials",
    "Retriever",
    "ChunkHit",
    "FileHit",
    "RetrieverRouter",
    "make_retriever",
    "create_retriever_from_env",
    "DifyRetriever",
    "DifyCredentials",
    "DifyBackendAdapter",
    "validate_dify_credentials",
    "create_dify_adapter",
]
