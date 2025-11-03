#!/bin/bash
# Stop MCP server

set -e

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
else
    echo "Error: utils.sh not found" >&2
    exit 1
fi

# Stop any running MCP server regardless of Python interpreter
print_info "Stopping MCP server..."

# Try to stop processes started with CLI flags first
if pgrep -f "kbbridge/server.py.*--host.*--port.*--transport" > /dev/null; then
    pkill -f "kbbridge/server.py.*--host.*--port.*--transport" || true
fi

# Fallback: stop any remaining kbbridge/server.py processes
if pgrep -f "kbbridge/server.py" > /dev/null; then
    pkill -f "kbbridge/server.py" || true
else
    print_warning "No server process found"
fi

print_success "Server stopped"
exit "${SUCCESS_EXITCODE}"
