"""
Extraction module

This module contains components for extracting and processing content from documents,
including paragraph location and content clustering.
"""

from .content_cluster import ContentCluster
from .paragraph_locator import ParagraphLocator

__all__ = [
    # Content Processing
    "ContentCluster",
    "ParagraphLocator",
]
