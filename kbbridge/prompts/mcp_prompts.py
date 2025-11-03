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
- dataset_info (required): [{"id": "...", "source_path": ""}]
- query, custom_instructions, document_name
- enable_reflection: true for comprehensive queries ("all/every/complete")

**file_lister**: List files | **file_discover**: Find relevant files | **retriever**: Get chunks

## Query Types

**Comprehensive** ("all/every/complete/list"):
- custom_instructions: "Extract ALL items comprehensively"
- enable_reflection: true, reflection_threshold: 0.75-0.80

**Simple**: Add domain context to custom_instructions

**Document-specific**: Use document_name parameter

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
dataset_info: [{"id": "dataset-id", "source_path": "optional/path"}]

## Examples
Single dataset: [{"id": "hr-docs", "source_path": ""}]
With path: [{"id": "legal", "source_path": "contracts/2024"}]
Multiple: [{"id": "hr", "source_path": ""}, {"id": "finance", "source_path": ""}]

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
