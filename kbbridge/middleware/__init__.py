"""
Middleware Module for FastMCP Server

This module provides true middleware functionality for request/response processing,
authentication, logging, and other cross-cutting concerns.
"""

# Import classes and functions
from ._auth_core import AuthMiddleware
from .credential_manager import MCPConfigHelper
from .error_middleware import ErrorMiddleware, error_middleware
from .tool_decorators import mcp_tool_with_auth, optional_auth, require_auth

__all__ = [
    "AuthMiddleware",
    "ErrorMiddleware",
    "MCPConfigHelper",
    "error_middleware",
    "mcp_tool_with_auth",
    "require_auth",
    "optional_auth",
]
