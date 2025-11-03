import json
import logging

logger = logging.getLogger(__name__)


class ErrorMiddleware:
    """
    Error middleware that handles exceptions and formats error responses
    """

    def handle_error(self, error: Exception, tool_name: str) -> str:
        """Handle an exception and return formatted error response"""
        logger.error(f"Error in {tool_name}: {error}")

        # Create standardized error response
        error_response = {
            "error": "Tool execution failed",
            "status": "error",
            "tool": tool_name,
            "message": str(error),
            "type": type(error).__name__,
        }

        return json.dumps(error_response)

    def handle_auth_error(self, message: str, errors: list = None) -> str:
        """Handle authentication errors"""
        return json.dumps(
            {
                "error": "Authentication failed",
                "status": "error",
                "message": message,
                "errors": errors or [],
            }
        )

    def handle_validation_error(self, message: str, field: str = None) -> str:
        """Handle validation errors"""
        error_response = {
            "error": "Validation failed",
            "status": "error",
            "message": message,
        }

        if field:
            error_response["field"] = field

        return json.dumps(error_response)


# Global middleware instance
error_middleware = ErrorMiddleware()
