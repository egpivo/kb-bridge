"""
MCP Client Helper for HTTP-based servers.

Provides a simple client for calling MCP tools over HTTP.
"""

import json
import httpx
from typing import Dict, Any, Optional


class MCPClient:
    """Simple HTTP client for MCP servers."""
    
    def __init__(self, base_url: str):
        """
        Initialize MCP client.
        
        Args:
            base_url: Base URL of MCP server (e.g., "http://localhost:5566/mcp")
        """
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
        
        Returns:
            Tool response as dict
        """
        # MCP protocol uses JSON-RPC format
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self.client.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract result from JSON-RPC response
        if "result" in result:
            # If result has content (MCP format), extract it
            if "content" in result["result"]:
                content_text = result["result"]["content"][0].get("text", "")
                try:
                    return json.loads(content_text)
                except json.JSONDecodeError:
                    return {"text": content_text}
            return result["result"]
        elif "error" in result:
            raise Exception(f"MCP Error: {result['error']}")
        return result
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Backward compatibility: ClientSession-like interface
class ClientSession:
    """
    ClientSession-compatible interface for HTTP MCP servers.
    
    Usage:
        async with ClientSession("http://localhost:5566/mcp") as session:
            result = await session.call_tool("assistant", {"query": "..."})
            response_data = json.loads(result.content[0].text)
    """
    
    def __init__(self, url: str):
        """
        Initialize session.
        
        Args:
            url: MCP server URL
        """
        self.client = MCPClient(url)
        self._closed = False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """
        Call an MCP tool.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
        
        Returns:
            Result object with content attribute (compatible with MCP ClientSession)
        """
        result = await self.client.call_tool(tool_name, arguments)
        
        # Convert to ClientSession-like format
        # The original MCP ClientSession returns result.content[0].text as JSON string
        class Result:
            def __init__(self, data):
                self.data = data
                # Create content attribute similar to MCP ClientSession
                # content[0].text should be a JSON string
                content_obj = type('Content', (), {'text': json.dumps(data)})()
                self.content = [content_obj]
        
        return Result(result)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if not self._closed:
            await self.client.close()
            self._closed = True
