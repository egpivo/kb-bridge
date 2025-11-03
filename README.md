# KBBridge

A Model Context Protocol (MCP) server for intelligent knowledge base search and retrieval with Dify integration.

## Installation

```bash
pip install kbbridge
```

## Quick Start

### Configuration

Create a `.env` file with your Dify credentials:

```bash
# Required - Dify Configuration
RETRIEVAL_ENDPOINT=https://api.dify.ai/v1
RETRIEVAL_API_KEY=your-dify-api-key
LLM_API_URL=https://your-llm-service.com/v1
LLM_MODEL=gpt-4o
LLM_API_TOKEN=your-token-here

# Optional
RERANK_URL=https://your-rerank-api.com
RERANK_MODEL=your-rerank-model
```

**Note:** Currently supports Dify as the retrieval backend.

See `env.example` for all available configuration options.

### Running the Server

```bash
# Start server
python -m kbbridge.server --host 0.0.0.0 --port 5210

# Or using Makefile (if available)
make start
```

Server runs on `http://0.0.0.0:5210` with MCP endpoint at `http://0.0.0.0:5210/mcp`.

## Features

- **Dify Integration**: Native support for Dify knowledge base retrieval
- **Multiple Search Methods**: Hybrid, semantic, keyword, and full-text search via Dify
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
            "dataset_info": json.dumps([{"id": "dataset_id", "name": "Dataset"}]),
            "query": "What are the safety protocols?"
        })
        print(result.content[0].text)

asyncio.run(main())
```

### With Custom Instructions

```python
await session.call_tool("assistant", {
    "dataset_info": json.dumps([{"id": "hr_dataset", "name": "HR Policies"}]),
    "query": "What is the maternity leave policy?",
    "custom_instructions": "Focus on HR compliance and legal requirements."
})
```

### With Quality Reflection

```python
await session.call_tool("assistant", {
    "dataset_info": json.dumps([{"id": "dataset_id", "name": "Dataset"}]),
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
