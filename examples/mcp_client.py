"""
MCP Client Helper for HTTP-based servers.

Provides a simple client wrapper around FastMCP Client for calling MCP tools.
"""

import json
from typing import Dict, Any, Optional

try:
    from fastmcp import Client as FastMCPClient
except ImportError:
    raise ImportError(
        "FastMCP is required. Install it with: pip install fastmcp\n"
        "FastMCP provides the best support for MCP servers."
    )


class ClientSession:
    """
    ClientSession-compatible interface for HTTP MCP servers using FastMCP.
    
    Usage:
        async with ClientSession("http://localhost:5566/mcp") as session:
            result = await session.call_tool("assistant", {"query": "..."})
            response_data = json.loads(result.content[0].text)
    """
    
    def __init__(self, url: str):
        """
        Initialize session.
        
        Args:
            url: MCP server URL (e.g., "http://localhost:5566/mcp")
        """
        self.url = url
        self.client = FastMCPClient(url)
        self._closed = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
        self._closed = True
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """
        Call an MCP tool.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
        
        Returns:
            Result object with content attribute (compatible with MCP ClientSession)
        """
        # Use FastMCP Client directly
        result = await self.client.call_tool(tool_name, arguments)
        
        # Convert to ClientSession-like format
        class Result:
            def __init__(self, data):
                self.data = data
                # FastMCP returns result directly, wrap it in content format
                if isinstance(data, dict):
                    content_obj = type('Content', (), {'text': json.dumps(data)})()
                    self.content = [content_obj]
                elif hasattr(data, 'content'):
                    self.content = data.content
                else:
                    content_obj = type('Content', (), {'text': json.dumps(data)})()
                    self.content = [content_obj]
        
        return Result(result)
