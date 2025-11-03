#!/bin/bash
# Utility functions for kbbridge scripts

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source color map
if [[ -f "${SCRIPT_DIR}/color_map.sh" ]]; then
    source "${SCRIPT_DIR}/color_map.sh"
else
    echo "Error: color_map.sh not found" >&2
    exit 1
fi

# Source exit codes
if [[ -f "${SCRIPT_DIR}/exit_code.sh" ]]; then
    source "${SCRIPT_DIR}/exit_code.sh"
else
    echo "Error: exit_code.sh not found" >&2
    exit 1
fi

# Colored output functions
print_info() {
    echo -e "${FG_BLUE}[INFO]${FG_RESET} $1"
}

print_success() {
    echo -e "${FG_GREEN}[SUCCESS]${FG_RESET} $1"
}

print_warning() {
    echo -e "${FG_YELLOW}[WARNING]${FG_RESET} $1"
}

print_error() {
    echo -e "${FG_RED}[ERROR]${FG_RESET} $1" >&2
}

print_gray() {
    echo -e "${FG_GRAY}$1${FG_RESET}"
}
