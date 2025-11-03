import asyncio
from functools import wraps
from typing import Callable

from kbbridge.middleware._auth_core import auth_middleware
from kbbridge.middleware.error_middleware import error_middleware


def mcp_tool_with_auth(require_auth: bool = True):
    """
    Decorator for MCP tools that integrates with middleware

    Args:
        require_auth: Whether authentication is required for this tool
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            # Handle async functions
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    # Handle authentication
                    if require_auth:
                        credentials = auth_middleware.get_available_credentials()
                        if not credentials:
                            error_msg = auth_middleware.create_auth_error_response(
                                "Missing required credentials"
                            )
                            return error_msg

                        # Validate credentials
                        validation = auth_middleware.validate_credentials(credentials)
                        if not validation["valid"]:
                            error_msg = auth_middleware.create_auth_error_response(
                                "Invalid credentials", validation["errors"]
                            )
                            return error_msg

                        # Set current credentials in global context
                        from kbbridge.server import set_current_credentials

                        set_current_credentials(credentials)
                    else:
                        # Optional auth - set if available
                        credentials = auth_middleware.get_available_credentials()
                        if credentials:
                            from kbbridge.server import set_current_credentials

                            set_current_credentials(credentials)

                    # Execute the async tool
                    result = await func(*args, **kwargs)
                    return result

                except Exception as e:
                    # Handle errors
                    error_result = error_middleware.handle_error(e, func.__name__)
                    return error_result

            return async_wrapper
        else:
            # Handle sync functions
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    # Handle authentication
                    if require_auth:
                        credentials = auth_middleware.get_available_credentials()
                        if not credentials:
                            error_msg = auth_middleware.create_auth_error_response(
                                "Missing required credentials"
                            )
                            return error_msg

                        # Validate credentials
                        validation = auth_middleware.validate_credentials(credentials)
                        if not validation["valid"]:
                            error_msg = auth_middleware.create_auth_error_response(
                                "Invalid credentials", validation["errors"]
                            )
                            return error_msg

                        # Set current credentials in global context
                        from kbbridge.server import set_current_credentials

                        set_current_credentials(credentials)
                    else:
                        # Optional auth - set if available
                        credentials = auth_middleware.get_available_credentials()
                        if credentials:
                            from kbbridge.server import set_current_credentials

                            set_current_credentials(credentials)

                    # Execute the sync tool
                    result = func(*args, **kwargs)
                    return result

                except Exception as e:
                    # Handle errors
                    error_result = error_middleware.handle_error(e, func.__name__)
                    return error_result

            return sync_wrapper

    return decorator


def require_auth(func: Callable) -> Callable:
    """Decorator for tools that require authentication"""
    return mcp_tool_with_auth(require_auth=True)(func)


def optional_auth(func: Callable) -> Callable:
    """Decorator for tools that can work with or without authentication"""
    return mcp_tool_with_auth(require_auth=False)(func)
