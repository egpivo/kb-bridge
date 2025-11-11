"""Core utility modules: JSON, profiling, and text processing utilities."""

from .json_utils import UUID_PATTERN, parse_json_from_markdown
from .profiling_utils import profile_stage
from .text_processing_utils import build_file_surrogate_text

__all__ = [
    "parse_json_from_markdown",
    "UUID_PATTERN",
    "profile_stage",
    "build_file_surrogate_text",
]
