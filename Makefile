# MCP Knowledge Base Assistant - Makefile

# Configuration
VENV := venv
PYTHON := $(shell command -v python3 || command -v python || echo "python")
PYTHONPATH_VAR := $(PWD):.
SERVER_HOST := 0.0.0.0
SERVER_PORT := 5210
SERVER_TRANSPORT := streamable-http
SERVER_CMD := $(PYTHON) kbbridge/server.py --host $(SERVER_HOST) --port $(SERVER_PORT) --transport $(SERVER_TRANSPORT)

LOG_FILE := kbbridge_server.log
PYTEST_ARGS := -v --tb=short
PYTEST_COV_ARGS := --cov=kbbridge --cov-report=html --cov-report=term-missing

-include .env
.EXPORT_ALL_VARIABLES:

.PHONY: help start stop restart status cleanup logs tail-logs tail-reflection clean-logs clean clean-all \
	test coverage lint format venv install dev-setup activate version version-patch version-minor version-major version-set \
	start-dev start-prod start-test dev-start debug-env

help:
	@echo "MCP Knowledge Base Assistant - Available Commands:"
	@echo ""
	@echo "Server:  start, stop, restart, status, cleanup"
	@echo "Logs:    logs, tail-logs, tail-reflection, clean-logs"
	@echo "Tests:   test, coverage"
	@echo "Dev:     venv, install, dev-setup, lint, format"
	@echo "Version: version, version-patch, version-minor, version-major, version-set"
	@echo "Modes:   start-dev, start-prod, start-test, dev-start"
	@echo ""
	@echo "Run 'make <command>' for details"

# Server Management
start:
	@./scripts/start_server.sh

stop:
	@./scripts/stop_server.sh

restart: stop
	@sleep 1
	@$(MAKE) start

cleanup:
	@./scripts/cleanup_server.sh

status:
	@./scripts/check_status.sh

# Log Monitoring
logs:
	@if [ -f $(LOG_FILE) ]; then \
		cat $(LOG_FILE); \
	else \
		echo "Warning: No log file found. Start server with 'make start'"; \
	fi

tail-logs:
	@if [ -f $(LOG_FILE) ]; then \
		tail -f $(LOG_FILE) | grep -i --line-buffered -E "reflection|reflect|evaluat|kbbridge\.core\.reflection|info|warning|error|reflector"; \
	else \
		echo "Warning: No log file found. Run 'make start' first"; \
	fi

tail-reflection:
	@if [ -f $(LOG_FILE) ]; then \
		tail -f $(LOG_FILE) | grep -i --line-buffered -E "reflection|reflect|evaluat|kbbridge\.core\.reflection|dspy|score:|reflector"; \
	else \
		echo "Warning: No log file. Run 'LOG_LEVEL=DEBUG make start 2>&1 | tee $(LOG_FILE)'"; \
	fi

clean-logs:
	@rm -f mcp_server*.log kbbridge_server*.log && echo "Log files cleaned"

clean:
	@./scripts/clean_files.sh

clean-all: stop clean

# Testing
test:
	@echo "Running tests"
	@$(PYTHON) -c "import pytest" >/dev/null 2>&1 || (echo "Error: pytest not installed. Run 'make install' to install dev dependencies" && exit 1); \
	COV_ARGS=""; $(PYTHON) -c "import pytest_cov" >/dev/null 2>&1 && COV_ARGS="$(PYTEST_COV_ARGS)"; \
	PYTHONPATH=$(PYTHONPATH_VAR) $(PYTHON) -m pytest tests/ \
		--ignore=tests/dify \
		-m "not slow and not integration" \
		$$COV_ARGS $(PYTEST_ARGS); \
	echo ""; \
	echo "Coverage: htmlcov/index.html (run 'make coverage' to open)"

coverage:
	@if [ -f htmlcov/index.html ]; then \
		command -v open >/dev/null 2>&1 && open htmlcov/index.html || \
		command -v xdg-open >/dev/null 2>&1 && xdg-open htmlcov/index.html || \
		echo "Please open htmlcov/index.html manually"; \
	else \
		echo "Error: Run 'make test' first to generate coverage report"; exit 1; \
	fi

# Environment-Specific Starts
start-dev:
	@if pgrep -f "kbbridge/server.py.*--host.*--port.*--transport" > /dev/null; then \
		echo "Warning: Server already running. Use 'make restart'"; \
	else \
		echo "Starting server (DEBUG mode)..."; \
		LOG_LEVEL=DEBUG PYTHONPATH=$(PYTHONPATH_VAR) $(SERVER_CMD) 2>&1 | tee mcp_server_dev.log & \
		echo "Server started. Logs: mcp_server_dev.log"; \
	fi

start-prod:
	@if pgrep -f "kbbridge/server.py.*--host.*--port.*--transport" > /dev/null; then \
		echo "Warning: Server already running. Use 'make restart'"; \
	else \
		echo "Starting server (WARNING mode)..."; \
		LOG_LEVEL=WARNING PYTHONPATH=$(PYTHONPATH_VAR) $(SERVER_CMD) 2>&1 | tee mcp_server_prod.log & \
		echo "Server started. Logs: mcp_server_prod.log"; \
	fi

start-test:
	@if pgrep -f "kbbridge/server.py.*--host.*--port.*--transport" > /dev/null; then \
		echo "Warning: Server already running. Use 'make restart'"; \
	else \
		echo "Starting server (ERROR mode)..."; \
		LOG_LEVEL=ERROR PYTHONPATH=$(PYTHONPATH_VAR) $(SERVER_CMD) 2>&1 | tee mcp_server_test.log & \
		echo "Server started. Logs: mcp_server_test.log"; \
	fi

dev-start:
	@LOG_LEVEL=DEBUG PYTHONPATH=$(PYTHONPATH_VAR) $(SERVER_CMD)

# Code Quality
lint:
	@$(PYTHON) -m flake8 kbbridge/ tests/ --max-line-length=120 --ignore=E203,W503,E501,F541 --exclude=__pycache__,*.pyc,.git,venv

format:
	@$(PYTHON) -m black kbbridge/ tests/ --line-length=100

# Setup
venv:
	@python3 -m venv $(VENV) && echo "Virtual environment created"

install: venv
	@$(PYTHON) -m pip install -e ".[dev]"

activate: venv
	@echo "Run: source $(VENV)/bin/activate"

dev-setup: venv install
	@echo "Development setup complete! Run 'make test' to verify"

# Version Management
version:
	@$(PYTHON) -c "import kbbridge; print(kbbridge.__version__)"

version-patch:
	@bump2version patch && echo "Patch version bumped"

version-minor:
	@bump2version minor && echo "Minor version bumped"

version-major:
	@bump2version major && echo "Major version bumped"

version-set:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: Specify VERSION (e.g., make version-set VERSION=1.2.3)"; exit 1; \
	fi
	@bump2version --new-version $(VERSION) patch && echo "Version set to $(VERSION)"

# Utilities
debug-env:
	@echo "=== Environment Variables ==="
	@env | grep -E "DIFY|LLM|RETRIEVAL" || echo "No DIFY/LLM/RETRIEVAL vars found"
