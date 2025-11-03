#!/bin/bash
# Start MCP server with logging

set -e

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
else
    echo "Error: utils.sh not found" >&2
    exit 1
fi

# Configuration
VENV="venv"
PYTHONPATH_VAR="$(pwd):."
SERVER_HOST="0.0.0.0"
SERVER_PORT="5210"
SERVER_TRANSPORT="streamable-http"
LOG_FILE="kbbridge_server.log"

# Determine Python interpreter
if [ -f "$VENV/bin/python" ]; then
    PYTHON="$VENV/bin/python"
    print_info "Using virtual environment Python"
else
    PYTHON="python"
    print_info "Using system Python ($(which python))"
fi

SERVER_CMD="$PYTHON kbbridge/server.py --host $SERVER_HOST --port $SERVER_PORT --transport $SERVER_TRANSPORT"
SERVER_PROCESS="$PYTHON kbbridge/server.py"

print_info "Starting MCP server with logging..."

if pgrep -f "kbbridge/server.py.*--host.*--port.*--transport" > /dev/null; then
    print_warning "Server is already running. Use 'make restart' to restart."
    exit "${ERROR_EXITCODE}"
else
    PYTHONPATH="$PYTHONPATH_VAR" $SERVER_CMD 2>&1 | tee "$LOG_FILE" &
    print_success "Server started. Logs are being written to $LOG_FILE"
    print_info "Use 'make tail-logs' to follow logs in real-time"
    exit "${SUCCESS_EXITCODE}"
fi
