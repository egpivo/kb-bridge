try:
    from fastmcp import Client
except ImportError:
    raise ImportError("FastMCP is required. Install it with: pip install fastmcp")

# Alias for backward compatibility
ClientSession = Client
