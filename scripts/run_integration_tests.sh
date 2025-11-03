#!/bin/bash

# Integration Test Runner Script
# This script runs all Dify integration tests with proper environment setup

set -e  # Exit on any error

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/utils.sh" ]]; then
    source "${SCRIPT_DIR}/utils.sh"
else
    echo "Error: utils.sh not found" >&2
    exit 1
fi

# Alias print_status to print_info for compatibility
print_status() {
    print_info "$1"
}

# Function to check if environment variables are set
check_env_vars() {
    print_status "Checking environment variables..."

    local missing_vars=()

    if [ -z "$RETRIEVAL_ENDPOINT" ]; then
        missing_vars+=("RETRIEVAL_ENDPOINT")
    fi

    if [ -z "$RETRIEVAL_API_KEY" ]; then
        missing_vars+=("RETRIEVAL_API_KEY")
    fi

    if [ -z "$LLM_API_URL" ]; then
        missing_vars+=("LLM_API_URL")
    fi

    if [ -z "$LLM_MODEL" ]; then
        missing_vars+=("LLM_MODEL")
    fi

    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_warning "Some environment variables are not set: ${missing_vars[*]}"
        print_status "Using default values from Makefile..."
    else
        print_success "All environment variables are set"
    fi
}

# Function to run a specific test suite
run_test_suite() {
    local test_name="$1"
    local test_path="$2"
    local description="$3"

    print_status "Running $description..."
    echo "=========================================="

    if make "$test_name"; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

# Function to run all integration tests
run_all_integration_tests() {
    print_status "Starting comprehensive Dify integration test suite..."
    echo "=========================================="

    local failed_tests=()
    local total_tests=0
    local passed_tests=0

    # Test suites to run
    declare -a test_suites=(
        "test-dify:Dify Core Tests:Core Dify functionality tests"
        "test-evaluation:Dify Evaluation Tests:LLM evaluator integration tests"
        "test-qa-evaluation:Dify QA Evaluation Tests:QA test cases with real data"
        "test-dify-integration:Dify Integration Tests:All Dify integration tests"
    )

    for test_suite in "${test_suites[@]}"; do
        IFS=':' read -r make_target test_name description <<< "$test_suite"
        total_tests=$((total_tests + 1))

        if run_test_suite "$make_target" "$test_name" "$description"; then
            passed_tests=$((passed_tests + 1))
        else
            failed_tests+=("$test_name")
        fi

        echo ""
    done

    # Summary
    echo "=========================================="
    print_status "Integration Test Summary:"
    echo "  Total test suites: $total_tests"
    echo "  Passed: $passed_tests"
    echo "  Failed: $((total_tests - passed_tests))"

    if [ ${#failed_tests[@]} -ne 0 ]; then
        print_error "Failed test suites: ${failed_tests[*]}"
        return 1
    else
        print_success "All integration tests passed!"
        return 0
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all              Run all integration tests (default)"
    echo "  --dify             Run only Dify core tests"
    echo "  --evaluation       Run only evaluation tests"
    echo "  --qa-evaluation    Run only QA evaluation tests"
    echo "  --integration      Run all Dify integration tests"
    echo "  --help             Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DIFY_ENDPOINT              Dify API endpoint"
    echo "  DIFY_API_KEY               Dify dataset API key"
    echo "  DIFY_WORKFLOW_API_KEY      Dify workflow API key"
    echo "  LLM_API_URL                LLM API URL"
    echo "  LLM_MODEL                  LLM model name"
    echo "  LLM_API_TOKEN              LLM API token (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 --all                   # Run all integration tests"
    echo "  $0 --evaluation            # Run only evaluation tests"
    echo "  DIFY_ENDPOINT=https://... $0 --qa-evaluation  # Override endpoint"
}

# Main execution
main() {
    # Change to project root directory
    cd "$(dirname "$0")/.."

    print_status "Dify Integration Test Runner"
    print_status "Project: $(pwd)"
    print_status "Timestamp: $(date)"
    echo ""

    # Check environment variables
    check_env_vars
    echo ""

    # Parse command line arguments
    case "${1:---all}" in
        --all)
            run_all_integration_tests
            ;;
        --dify)
            run_test_suite "test-dify" "Dify Core Tests" "Core Dify functionality tests"
            ;;
        --evaluation)
            run_test_suite "test-evaluation" "Dify Evaluation Tests" "LLM evaluator integration tests"
            ;;
        --qa-evaluation)
            run_test_suite "test-qa-evaluation" "Dify QA Evaluation Tests" "QA test cases with real data"
            ;;
        --integration)
            run_test_suite "test-dify-integration" "Dify Integration Tests" "All Dify integration tests"
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit "${ERROR_EXITCODE}"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
