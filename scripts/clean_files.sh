#!/bin/bash
# Clean all temporary and cache files

set -e

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
else
    echo "Error: utils.sh not found" >&2
    exit 1
fi

print_info "Cleaning all temporary and cache files..."

print_info "Removing log files..."
rm -f mcp_server*.log

print_info "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "*.pyd" -delete 2>/dev/null || true

print_info "Removing pytest cache..."
rm -rf .pytest_cache/ 2>/dev/null || true

print_info "Removing coverage reports..."
rm -rf htmlcov/ 2>/dev/null || true
rm -f .coverage 2>/dev/null || true

print_info "Removing build artifacts..."
rm -rf build/ 2>/dev/null || true
rm -rf dist/ 2>/dev/null || true
rm -rf *.egg-info/ 2>/dev/null || true

print_info "Removing temporary files..."
rm -f .DS_Store 2>/dev/null || true
rm -f Thumbs.db 2>/dev/null || true

print_success "All temporary and cache files cleaned"
exit "${SUCCESS_EXITCODE}"
