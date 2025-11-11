# KB-Bridge

[![Tests](https://github.com/egpivo/kb-bridge/workflows/Test/badge.svg)](https://github.com/egpivo/kb-bridge/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/egpivo/kb-bridge/main.svg)](https://app.codecov.io/github/egpivo/kb-bridge?branch=main)
<a href="https://pypi.org/project/kbbridge/"><img src="https://img.shields.io/pypi/v/kbbridge.svg?logo=pypi&label=PyPI&logoColor=silver"/></a>


A Model Context Protocol (MCP) server for intelligent knowledge base search and retrieval with support for multiple backend providers.

## Installation

```bash
pip install kbbridge
```

## Quick Start

### Configuration

Create a `.env` file with your retrieval backend credentials:

```bash
# Required - Retrieval Backend Configuration
RETRIEVAL_ENDPOINT=https://api.dify.ai/v1  # Example: Dify endpoint
RETRIEVAL_API_KEY=your-retrieval-api-key
LLM_API_URL=https://your-llm-service.com/v1
LLM_MODEL=gpt-4o
LLM_API_TOKEN=your-token-here

# Optional
RERANK_URL=https://your-rerank-api.com
RERANK_MODEL=your-rerank-model
```

**Supported Backends:**

| Backend | Status | Notes |
|---------|--------|-------|
| Dify | Supported | Currently available |
| Others | Planned | Additional backends coming soon |

See `env.example` for all available configuration options.

### Running the Server

```bash
# Start server
python -m kbbridge.server --host 0.0.0.0 --port 5210

# Or using Makefile (if available)
make start
```

Server runs on `http://0.0.0.0:5210` with MCP endpoint at `http://0.0.0.0:5210/mcp`.

### Deployment Options

#### Option 1: Docker (Local Development / Simple Deployments)

For local development or simple single-container deployments:

```bash
# Build the image
docker build -t kbbridge:latest .

# Run with environment variables
docker run -d \
  --name kbbridge \
  -p 5210:5210 \
  --env-file .env \
  kbbridge:latest
```

For production deployments, use container orchestration platforms like Kubernetes with your preferred deployment method.

## Features

- **Backend Integration**: Extensible architecture supporting multiple retrieval backends
- **Multiple Search Methods**: Hybrid, semantic, keyword, and full-text search
- **Quality Reflection**: Automatic answer quality evaluation and refinement
- **Custom Instructions**: Domain-specific query guidance

## Available Tools

- **`assistant`**: Intelligent search and answer extraction from knowledge bases
- **`file_discover`**: Discover relevant files using retriever + optional reranking
- **`file_lister`**: List files in knowledge base datasets
- **`keyword_generator`**: Generate search keywords using LLM
- **`retriever`**: Retrieve information using various search methods
- **`file_count`**: Get file count in knowledge base dataset

## Usage Examples

### Basic Query

```python
import asyncio
from mcp import ClientSession

async def main():
    async with ClientSession("http://localhost:5210/mcp") as session:
        result = await session.call_tool("assistant", {
            "resource_id": "resource-id",
            "query": "What are the safety protocols?"
        })
        print(result.content[0].text)

asyncio.run(main())
```

### With Custom Instructions

```python
await session.call_tool("assistant", {
    "resource_id": "hr_dataset",
    "query": "What is the maternity leave policy?",
    "custom_instructions": "Focus on HR compliance and legal requirements."
})
```

### With Quality Reflection

```python
await session.call_tool("assistant", {
    "resource_id": "resource-id",
    "query": "What are the safety protocols?",
    "reflection_mode": "standard",  # "off", "standard", or "comprehensive"
    "reflection_threshold": 0.75,
    "max_reflection_iterations": 2
})
```

## Reflection Modes

- **`off`**: No reflection (fastest)
- **`standard`** (default): Answer quality evaluation only
- **`comprehensive`**: Search coverage + answer quality evaluation

Reflection evaluates answers on:
- **Completeness** (30%): Does the answer fully address the query?
- **Accuracy** (30%): Are sources relevant and correctly cited?
- **Relevance** (20%): Does the answer stay on topic?
- **Clarity** (10%): Is the answer clear and well-structured?
- **Confidence** (10%): Quality of supporting sources?

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black kbbridge/ tests/

# Lint code
ruff check kbbridge/ tests/
```

## License

Apache-2.0
