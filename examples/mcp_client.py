import json
from typing import Dict, Any

try:
    from fastmcp import Client as FastMCPClient
except ImportError:
    raise ImportError("FastMCP is required. Install it with: pip install fastmcp")


class ClientSession(FastMCPClient):
    """Simple wrapper - just override call_tool to ensure content[0].text format."""
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call tool - use FastMCP result directly if compatible, otherwise wrap."""
        result = await super().call_tool(tool_name, arguments)
        # If FastMCP already returns content[0].text structure, use it directly
        if (hasattr(result, 'content') and 
            isinstance(result.content, list) and 
            len(result.content) > 0 and 
            hasattr(result.content[0], 'text')):
            return result
        # Otherwise wrap it
        content = type('Content', (), {'text': json.dumps(result)})()
        return type('Result', (), {'content': [content]})()
