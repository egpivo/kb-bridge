#!/bin/bash
# Test MCP server connection

set -e

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
else
    echo "Error: utils.sh not found" >&2
    exit 1
fi

SERVER_PORT="9900"
VENV="venv"
PYTHON="$VENV/bin/python"
SERVER_PROCESS="$PYTHON kbbridge/server.py"

print_info "Testing MCP server connection..."

if pgrep -f "$SERVER_PROCESS" > /dev/null; then
    print_success "Server is running"
    print_info "Testing connection..."

    response=$(curl -X POST "http://localhost:$SERVER_PORT/mcp" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}' \
        -s)

    if echo "$response" | jq . > /dev/null 2>&1; then
        print_success "Connection successful!"
        print_info "Response:"
        echo "$response" | jq .
        exit "${SUCCESS_EXITCODE}"
    else
        print_warning "Server responded but with non-JSON content:"
        echo "$response"
        exit "${ERROR_EXITCODE}"
    fi
else
    print_error "Server is not running. Start it with 'make start'"
    exit "${ERROR_EXITCODE}"
fi
