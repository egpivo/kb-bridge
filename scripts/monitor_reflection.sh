#!/bin/bash
# Monitor reflection logs in real-time

set -e

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
else
    echo "Error: utils.sh not found" >&2
    exit 1
fi

LOG_FILE="qa_hub_server.log"

print_info "Monitoring Reflection Logs"
echo "=============================="
echo ""
print_info "Watching: $LOG_FILE"
print_info "Press Ctrl+C to stop"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    print_error "Log file not found: $LOG_FILE"
    echo ""
    print_info "Make sure server is running and logging:"
    echo "  make start 2>&1 | tee qa_hub_server.log"
    exit "${ERROR_EXITCODE}"
fi

# Show last 5 lines first
print_info "Last 5 log entries:"
echo "-------------------"
tail -5 "$LOG_FILE"
echo ""
print_info "Waiting for new logs..."
echo "-------------------"

# Follow and filter for reflection (case-insensitive)
tail -f "$LOG_FILE" | grep -i --line-buffered -E "reflection|reflect|evaluat|qa_hub\.core\.reflection|dspy|score:|query|dataset|reflector" --color=always
