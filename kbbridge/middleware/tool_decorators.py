import asyncio
from functools import wraps
from typing import Callable

from kbbridge.config.config import Credentials
from kbbridge.middleware._auth_core import auth_middleware, set_current_credentials
from kbbridge.middleware.error_middleware import error_middleware


def mcp_tool_with_auth(require_auth: bool = True):
    """Decorator for MCP tools that integrates with middleware."""

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    # Attempt to extract session config from ctx and set session creds
                    ctx = (
                        kwargs.get("ctx")
                        if "ctx" in kwargs
                        else next(
                            (a for a in args if hasattr(a, "session_config")), None
                        )
                    )
                    if ctx is not None and hasattr(ctx, "session_config"):
                        sc = getattr(ctx, "session_config", None)
                        if sc is not None:
                            # Support both Pydantic model and dict-like access
                            def _get(attr):
                                return (
                                    getattr(sc, attr, None)
                                    if hasattr(sc, attr)
                                    else (
                                        sc.get(attr) if isinstance(sc, dict) else None
                                    )
                                )

                            re = _get("retrieval_endpoint")
                            rk = _get("retrieval_api_key")
                            llm_url = _get("llm_api_url")
                            llm_model = _get("llm_model")
                            llm_token = _get("llm_api_token")
                            rr_url = _get("rerank_url")
                            rr_model = _get("rerank_model")
                            if re and rk:
                                auth_middleware.set_session_credentials(
                                    Credentials(
                                        retrieval_endpoint=re,
                                        retrieval_api_key=rk,
                                        llm_api_url=llm_url,
                                        llm_model=llm_model,
                                        llm_api_token=llm_token,
                                        rerank_url=rr_url,
                                        rerank_model=rr_model,
                                    )
                                )

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

                        set_current_credentials(credentials)
                    else:
                        credentials = auth_middleware.get_available_credentials()
                        if credentials:
                            set_current_credentials(credentials)

                    result = await func(*args, **kwargs)
                    return result

                except Exception as e:
                    error_result = error_middleware.handle_error(e, func.__name__)
                    return error_result

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    # Attempt to extract session config from ctx and set session creds
                    ctx = (
                        kwargs.get("ctx")
                        if "ctx" in kwargs
                        else next(
                            (a for a in args if hasattr(a, "session_config")), None
                        )
                    )
                    if ctx is not None and hasattr(ctx, "session_config"):
                        sc = getattr(ctx, "session_config", None)
                        if sc is not None:

                            def _get(attr):
                                return (
                                    getattr(sc, attr, None)
                                    if hasattr(sc, attr)
                                    else (
                                        sc.get(attr) if isinstance(sc, dict) else None
                                    )
                                )

                            re = _get("retrieval_endpoint")
                            rk = _get("retrieval_api_key")
                            llm_url = _get("llm_api_url")
                            llm_model = _get("llm_model")
                            llm_token = _get("llm_api_token")
                            rr_url = _get("rerank_url")
                            rr_model = _get("rerank_model")
                            if re and rk:
                                auth_middleware.set_session_credentials(
                                    Credentials(
                                        retrieval_endpoint=re,
                                        retrieval_api_key=rk,
                                        llm_api_url=llm_url,
                                        llm_model=llm_model,
                                        llm_api_token=llm_token,
                                        rerank_url=rr_url,
                                        rerank_model=rr_model,
                                    )
                                )

                    if require_auth:
                        credentials = auth_middleware.get_available_credentials()
                        if not credentials:
                            error_msg = auth_middleware.create_auth_error_response(
                                "Missing required credentials"
                            )
                            return error_msg

                        validation = auth_middleware.validate_credentials(credentials)
                        if not validation["valid"]:
                            error_msg = auth_middleware.create_auth_error_response(
                                "Invalid credentials", validation["errors"]
                            )
                            return error_msg

                        set_current_credentials(credentials)
                    else:
                        credentials = auth_middleware.get_available_credentials()
                        if credentials:
                            set_current_credentials(credentials)

                    result = func(*args, **kwargs)
                    return result

                except Exception as e:
                    error_result = error_middleware.handle_error(e, func.__name__)
                    return error_result

            return sync_wrapper

    return decorator


def require_auth(func: Callable) -> Callable:
    """Decorator for tools that require authentication."""
    return mcp_tool_with_auth(require_auth=True)(func)


def optional_auth(func: Callable) -> Callable:
    """Decorator for tools that can work with or without authentication."""
    return mcp_tool_with_auth(require_auth=False)(func)
