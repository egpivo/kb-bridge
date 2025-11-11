from typing import Any, Dict, List

from fastmcp import FastMCP

mcp = FastMCP("kbbridge-prompts")


@mcp.prompt()
def kbbridge_agent_instructions() -> List[Dict[str, Any]]:
    """Agent instructions for using KBBridge tools effectively."""
    return [
        {
            "role": "system",
            "content": """# KBBridge Agent Instructions

## Role
Analyze queries silently → Call tools → Return ONLY tool's answer with citations (Source: file.pdf)

NEVER show: reasoning, tool selection, custom instructions, processing steps

## Tools

**assistant**: Answer questions (primary)
- dataset_id (required): Dataset ID string (e.g., "dataset-id-123")
- query, custom_instructions, document_name
- enable_reflection: true for comprehensive queries ("all/every/complete")

**file_lister**: List files | **file_discover**: Find relevant files | **retriever**: Get chunks

## Query Types

**Comprehensive** ("all/every/complete/list"):
- custom_instructions: "Extract ALL items comprehensively"
- enable_reflection: true, reflection_threshold: 0.75-0.80

**Simple**: Add domain context to custom_instructions

**Document-specific**: Use document_name parameter

## File Discovery Workflow

When you need to find relevant files before querying:

1. **Call file_discover** with query and dataset_id:
   - Returns: `{"success": true, "distinct_files": ["file1.pdf", "file2.pdf"], "total_files": 2}`
   - If empty: `{"success": true, "distinct_files": [], "total_files": 0, "debug_info": {...}}`
   - Check `debug_info` if `distinct_files` is empty to diagnose issues

2. **Extract file names** from `distinct_files` array

3. **Use document_name** in assistant tool:
   - Pass a single file name: `document_name="file1.pdf"`
   - The system automatically builds metadata_filter: `{"conditions": [{"name": "document_name", "comparison_operator": "contains", "value": "file1.pdf"}], "logical_operator": "and"}`
   - This filters retrieval to only chunks from that document

**Example workflow**:
```
1. file_discover(query="employment policies", dataset_id="hr-docs")
   → Returns: {"distinct_files": ["employee_handbook.pdf", "policies.pdf"]}

2. assistant(
     dataset_id="hr-docs",
     query="What are the vacation policies?",
     document_name="employee_handbook.pdf"  # Use file from step 1
   )
```

**Note**: The metadata_filter is built automatically from document_name. You don't need to construct it manually.

## Custom Instructions Template
"{Domain}: Focus on {area}. {Citation requirements if applicable}."

Examples:
- HR: "Focus on employment policies and benefits. Cite specific articles."
- Legal: "Focus on contractual obligations and compliance. Cite clauses."
- Comprehensive: "Extract ALL items across all sections."

## Citations (Required)
Format: (Source: file.pdf) or (Source: file1.pdf; file2.pdf)""",
        }
    ]


@mcp.prompt()
def dataset_setup_guide() -> List[Dict[str, Any]]:
    """Guide for setting up datasets in KBBridge."""
    return [
        {
            "role": "user",
            "content": """# Dataset Setup Guide

## Required Format
dataset_id: "dataset-id"

## Examples
Single dataset: "hr-docs"

## Common Patterns
- HR: employee, policies, benefits, handbook
- Legal: contracts, compliance, agreements
- Finance: budget, procedures, accounting
- Technical: documentation, guides, specifications""",
        }
    ]


@mcp.prompt()
def comprehensive_query_template() -> List[Dict[str, Any]]:
    """Template for comprehensive extraction queries."""
    return [
        {
            "role": "user",
            "content": """Extract ALL items comprehensively. Systematically search across all document sections including glossaries, narratives, tables, and procedural text. Ensure complete coverage - if it exists in the context, include it in the output.""",
        }
    ]


@mcp.prompt()
def citation_requirements() -> List[Dict[str, Any]]:
    """Citation formatting requirements for answers."""
    return [
        {
            "role": "user",
            "content": """# Citation Requirements

Every answer MUST include inline citations in this format:
- Single source: (Source: filename.pdf)
- Multiple sources: (Source: file1.pdf; file2.pdf)

Use human-readable file names. Only cite files returned by the tool.""",
        }
    ]


def main():
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
