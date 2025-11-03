#!/bin/bash
# Check MCP server status

set -e

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
else
    echo "Error: utils.sh not found" >&2
    exit 1
fi

print_info "MCP Server Status:"

if pgrep -f "kbbridge/server.py" > /dev/null; then
    print_success "Server is RUNNING"
    print_gray "Matching processes:"
    pgrep -fa "kbbridge/server.py"
    echo ""
    print_info "Log files:"
    for logfile in mcp_server*.log kbbridge_server*.log; do
        if [ -f "$logfile" ]; then
            echo "  - $logfile: $(wc -l < "$logfile") lines"
        fi
    done
    echo ""
    print_info "Current LOG_LEVEL: $(echo $LOG_LEVEL || echo 'INFO (default)')"
else
    print_warning "Server is NOT running"
    exit "${ERROR_EXITCODE}"
fi

exit "${SUCCESS_EXITCODE}"
