import argparse
import asyncio
import json
import logging
import os
from typing import Optional

from fastmcp import Context, FastMCP
from pydantic import BaseModel

from kbbridge.config.config import Config
from kbbridge.config.constants import AssistantDefaults, RetrieverDefaults
from kbbridge.config.env_loader import get_env_int, load_env_file, print_env_status
from kbbridge.middleware import MCPConfigHelper, require_auth
from kbbridge.middleware._auth_core import get_current_credentials
from kbbridge.prompts import mcp as prompts_mcp
from kbbridge.services.assistant_service import assistant_service
from kbbridge.services.file_discover_service import file_discover_service
from kbbridge.services.file_lister_service import file_lister_service
from kbbridge.services.keyword_generator_service import keyword_generator_service
from kbbridge.services.retriever_service import retriever_service


# Configure logging based on environment
def get_log_level():
    """Get log level from environment variable."""
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(env_level, logging.INFO)


# Load environment variables from .env file
load_env_file()

# Configure logging
log_level = get_log_level()

# Setup handlers for both console and file output
handlers = [logging.StreamHandler()]  # Console output

# Add file handler if enabled (useful for STDIO transport where console is captured)
if os.getenv("LOG_TO_FILE", "false").lower() == "true":
    log_file = os.getenv("LOG_FILE", "kbbridge_server.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handlers.append(file_handler)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)
logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")
if os.getenv("LOG_TO_FILE") == "true":
    logger.info(f"File logging enabled: {os.getenv('LOG_FILE', 'qa_hub_server.log')}")

print_env_status()

# Initialize FastMCP
mcp = FastMCP("kb-mcp-server")

# Mount prompts
mcp.mount(prompts_mcp)

# Initialize credential manager and load default credentials
config_helper = MCPConfigHelper()
config_helper.apply_to_environment()

# Set environment variables to disable multiprocessing globally
# Only override if not already set in .env file
if "MAX_WORKERS" not in os.environ:
    os.environ["MAX_WORKERS"] = "1"
if "USE_CONTENT_BOOSTER" not in os.environ:
    os.environ["USE_CONTENT_BOOSTER"] = "false"


class SessionConfig(BaseModel):
    """Session configuration passed per user/session."""

    retrieval_endpoint: Optional[str] = None
    retrieval_api_key: Optional[str] = None
    llm_api_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_token: Optional[str] = None
    rerank_url: Optional[str] = None
    rerank_model: Optional[str] = None


# MCP Tools
@mcp.tool(name="assistant")
@require_auth
async def assistant(
    resource_id: str,
    query: str,
    ctx: Context,
    custom_instructions: Optional[str] = None,
    document_name: str = "",
    enable_query_rewriting: bool = False,
) -> str:
    """Search and extract answers from knowledge bases."""
    await ctx.info(f"Executing assistant for query: {query}")
    if custom_instructions:
        await ctx.info(f"Using custom instructions: {custom_instructions}")
    if enable_query_rewriting:
        await ctx.info("Query rewriting enabled (LLM-based expansion/relaxation)")

    timeout_seconds = AssistantDefaults.OVERALL_REQUEST_TIMEOUT.value

    try:
        credentials = get_current_credentials()
        if not credentials:
            await ctx.error("No credentials available")
            return "Error: No credentials available"

        # Note: dify_endpoint is a backward-compat property → retrieval_endpoint
        await ctx.info(f"Using retrieval endpoint: {credentials.retrieval_endpoint}")
        await ctx.info(f"Request timeout set to: {timeout_seconds} seconds")

        await ctx.info("Calling assistant_service...")

        try:
            result = await asyncio.wait_for(
                assistant_service(
                    resource_id=resource_id,
                    query=query,
                    ctx=ctx,
                    custom_instructions=custom_instructions,
                    document_name=document_name,
                    enable_query_rewriting=enable_query_rewriting,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            error_msg = (
                f"Request timed out after {timeout_seconds} seconds. "
                f"This query is taking longer than expected. "
                f"Try: 1) Simplifying your query, 2) Reducing the dataset scope, "
                f"3) Setting OVERALL_REQUEST_TIMEOUT env var to a higher value (e.g., 600 for 10 minutes), "
                f"or 4) Reducing document scope (set document_name)."
            )
            await ctx.error(error_msg)
            return json.dumps(
                {
                    "error": "Request timeout",
                    "status": "timeout",
                    "message": error_msg,
                    "timeout_seconds": timeout_seconds,
                    "suggestions": [
                        "Simplify your query",
                        "Reduce dataset scope",
                        f"Increase OVERALL_REQUEST_TIMEOUT (currently {timeout_seconds}s)",
                        "Disable content booster",
                        "Reduce max_workers parameter",
                    ],
                }
            )

        await ctx.info("assistant_service completed successfully")
        return json.dumps(result)

    except Exception as e:
        await ctx.error(f"KB assistant execution failed: {e}")
        return json.dumps(
            {"error": "Tool execution failed", "status": "error", "message": str(e)}
        )


@mcp.tool()
@require_auth
async def file_discover(
    query: str,
    resource_id: str,
    ctx: Context,
    top_k_recall: int = 100,
    top_k_return: int = 20,
    do_file_rerank: bool = True,
    relevance_score_threshold: float = 0.0,
    backend_type: Optional[str] = None,
) -> str:
    """Discover files using backend retriever and DSPy."""
    await ctx.info(f"Executing file_discover for query: {query}")
    try:
        credentials = get_current_credentials()
        if not credentials:
            await ctx.error("No credentials available")
            return "Error: No credentials available"

        # Normalize reranking flag based on credentials availability
        if do_file_rerank and not credentials.is_reranking_available():
            do_file_rerank = False
            await ctx.info(
                "File reranking disabled: RERANK_URL or RERANK_MODEL not configured"
            )

        result = file_discover_service(
            query=query,
            resource_id=resource_id,
            top_k_recall=top_k_recall,
            top_k_return=top_k_return,
            do_file_rerank=do_file_rerank,
            relevance_score_threshold=relevance_score_threshold,
            backend_type=backend_type,
            retrieval_endpoint=credentials.retrieval_endpoint,
            retrieval_api_key=credentials.retrieval_api_key,
            rerank_url=credentials.rerank_url,
            rerank_model=credentials.rerank_model,
        )
        return json.dumps(result)
    except Exception as e:
        await ctx.error(f"File discover failed: {e}")
        return json.dumps(
            {"error": "Tool execution failed", "status": "error", "message": str(e)}
        )


@mcp.tool()
@require_auth
async def file_lister(
    resource_id: str,
    ctx: Context,
    timeout: int = 30,
    limit: Optional[int] = None,
    offset: int = 0,
    backend_type: Optional[str] = None,
) -> str:
    """List files in knowledge base resource with pagination support."""
    await ctx.info(
        f"Executing file_lister for resource: {resource_id} (limit: {limit}, offset: {offset})"
    )

    try:
        credentials = get_current_credentials()
        if not credentials:
            await ctx.error("No credentials available")
            return "Error: No credentials available"

        result = file_lister_service(
            resource_id=resource_id,
            timeout=timeout,
            backend_type=backend_type,
            retrieval_endpoint=credentials.retrieval_endpoint,
            retrieval_api_key=credentials.retrieval_api_key,
        )

        return json.dumps(result)

    except Exception as e:
        await ctx.error(f"File lister execution failed: {e}")
        return json.dumps(
            {"error": "Tool execution failed", "status": "error", "message": str(e)}
        )


@mcp.tool()
@require_auth
async def keyword_generator(
    query: str,
    ctx: Context,
    max_sets: int = 5,
) -> str:
    """Generate keyword sets for search."""
    await ctx.info(f"Executing keyword_generator for query: {query}")

    try:
        credentials = get_current_credentials()
        if not credentials:
            await ctx.error("No credentials available")
            return "Error: No credentials available"

        result = keyword_generator_service.fn(
            query=query,
            max_sets=max_sets,
            retrieval_endpoint=credentials.retrieval_endpoint,
            retrieval_api_key=credentials.retrieval_api_key,
            llm_api_url=credentials.llm_api_url,
            llm_model=credentials.llm_model,
            llm_api_token=credentials.llm_api_token,
            rerank_url=credentials.rerank_url,
            rerank_model=credentials.rerank_model,
        )

        return json.dumps(result)

    except Exception as e:
        await ctx.error(f"Keyword generator execution failed: {e}")
        return json.dumps(
            {"error": "Tool execution failed", "status": "error", "message": str(e)}
        )


@mcp.tool()
@require_auth
async def retriever(
    resource_id: str,
    query: str,
    ctx: Context,
    search_method: str = RetrieverDefaults.SEARCH_METHOD.value,
    does_rerank: bool = RetrieverDefaults.DOES_RERANK.value,
    top_k: int = RetrieverDefaults.TOP_K.value,
    score_threshold: float = RetrieverDefaults.SCORE_THRESHOLD.value,
    weights: float = RetrieverDefaults.WEIGHTS.value,
    document_name: str = "",
    verbose: bool = AssistantDefaults.VERBOSE.value,
    backend_type: Optional[str] = None,
) -> str:
    """Retrieve information from knowledge base."""
    await ctx.info(f"Executing retriever for query: {query}")

    try:
        credentials = get_current_credentials()
        if not credentials:
            await ctx.error("No credentials available")
            return "Error: No credentials available"

        # Normalize reranking flag based on credentials availability
        if does_rerank and not credentials.is_reranking_available():
            does_rerank = False
            await ctx.info(
                "Reranking disabled: RERANK_URL or RERANK_MODEL not configured"
            )

        result = retriever_service(
            resource_id=resource_id,
            query=query,
            method=search_method,
            does_rerank=does_rerank,
            top_k=top_k,
            score_threshold=score_threshold,
            backend_type=backend_type,
            weights=weights,
            document_name=document_name,
            verbose=verbose,
            retrieval_endpoint=credentials.retrieval_endpoint,
            retrieval_api_key=credentials.retrieval_api_key,
        )

        return json.dumps(result)

    except Exception as e:
        await ctx.error(f"Retriever execution failed: {e}")
        return json.dumps(
            {"error": "Tool execution failed", "status": "error", "message": str(e)}
        )


@mcp.tool()
@require_auth
async def file_count(resource_id: str, ctx: Context) -> str:
    """Get file count in knowledge base resource."""
    await ctx.info(f"Executing file_count for resource: {resource_id}")

    try:
        credentials = get_current_credentials()
        if not credentials:
            await ctx.error("No credentials available")
            return "Error: No credentials available"

        result = file_lister_service(
            resource_id=resource_id,
            timeout=30,
            retrieval_endpoint=credentials.retrieval_endpoint,
            retrieval_api_key=credentials.retrieval_api_key,
        )

        if "error" in result:
            return json.dumps(result)

        files = result.get("files", [])
        return json.dumps(
            {"has_files": len(files) > 0, "file_count": len(files), "files": files}
        )

    except Exception as e:
        await ctx.error(f"File count execution failed: {e}")
        return json.dumps(
            {"error": "Tool execution failed", "status": "error", "message": str(e)}
        )


def create_server() -> FastMCP:
    """Create and return the FastMCP server instance."""
    return mcp


async def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="Knowledge Base MCP Server - Working Version"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=5210, help="Port to bind to (default: 5210)"
    )
    parser.add_argument(
        "--transport", default="streamable-http", help="Transport method"
    )

    args = parser.parse_args()

    logger.info("Starting Knowledge Base MCP Server (Working Version)")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"URL: http://{args.host}:{args.port}")
    logger.info(f"MCP Endpoint: http://{args.host}:{args.port}/mcp")
    logger.info("Authentication: Header-based credential support")
    logger.info("Fixes: Single-threaded processing, no multiprocessing")

    # Show credential configuration status
    validation = config_helper.validate_credentials()
    env_creds_present = Config.get_default_credentials() is not None
    if validation["valid"]:
        logger.info("Credentials: All required credentials configured (config file)")
    elif env_creds_present:
        logger.info("Credentials: Using environment variables (.env or process env)")
    else:
        logger.info(f"Warning: Credentials: Missing {validation['missing_required']}")
        logger.info("Configure via .env file or environment variables")

    # Check reranking configuration
    default_creds = Config.get_default_credentials()
    if default_creds:
        if default_creds.is_reranking_available():
            logger.info("Reranking: ENABLED (RERANK_URL and RERANK_MODEL configured)")
        else:
            logger.warning(
                "Reranking: DISABLED (RERANK_URL or RERANK_MODEL not configured)"
            )
            logger.warning(
                "  Reranking will be automatically disabled in all retrieval operations"
            )
    else:
        logger.warning("Reranking: DISABLED (no default credentials available)")
        logger.warning(
            "  Reranking will be automatically disabled in all retrieval operations"
        )

    logger.info("=" * 60)

    # Configure uvicorn with extended timeouts for long-running requests
    request_timeout = AssistantDefaults.OVERALL_REQUEST_TIMEOUT.value
    uvicorn_config = {
        "timeout_keep_alive": get_env_int(
            "UVICORN_TIMEOUT_KEEP_ALIVE", int(request_timeout * 2.1)
        ),
        "timeout_graceful_shutdown": get_env_int(
            "UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", 30
        ),
        "limit_concurrency": get_env_int("UVICORN_LIMIT_CONCURRENCY", 100),
        "limit_max_requests": get_env_int("UVICORN_LIMIT_MAX_REQUESTS", 0) or None,
        "timeout_notify": get_env_int("UVICORN_TIMEOUT_NOTIFY", 60),
        "h11_max_incomplete_event_size": get_env_int(
            "UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE", 32768
        ),
    }

    logger.info("Uvicorn Configuration:")
    logger.info(f"   timeout_keep_alive: {uvicorn_config['timeout_keep_alive']}s")
    logger.info(
        f"   timeout_graceful_shutdown: {uvicorn_config['timeout_graceful_shutdown']}s"
    )
    logger.info("=" * 60)

    await mcp.run_http_async(
        host=args.host,
        port=args.port,
        transport=args.transport,
        path="/mcp",
        log_level="info",
        uvicorn_config=uvicorn_config,
    )


if __name__ == "__main__":
    asyncio.run(main())
