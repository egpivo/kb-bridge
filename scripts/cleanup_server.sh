#!/bin/bash
# Cleanup script to stop any running KB assistant servers

set -e

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
else
    echo "Error: utils.sh not found" >&2
    exit 1
fi

print_info "Cleaning up KB Assistant servers..."

# Find and kill any running kbbridge server processes
PIDS=$(ps aux | grep "kbbridge/server.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    print_info "No KB assistant servers running"
else
    print_warning "Found running servers: $PIDS"
    for PID in $PIDS; do
        print_gray "Killing process $PID"
        kill $PID
    done
    print_success "Cleanup completed"
fi

# Also clean up any tee processes that might be logging
TEE_PIDS=$(ps aux | grep "tee kbbridge_server.log" | grep -v grep | awk '{print $2}')
if [ ! -z "$TEE_PIDS" ]; then
    print_info "Cleaning up logging processes: $TEE_PIDS"
    for PID in $TEE_PIDS; do
        kill $PID 2>/dev/null
    done
fi

print_success "Server cleanup complete!"
exit "${SUCCESS_EXITCODE}"
