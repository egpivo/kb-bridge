from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class RewriteStrategy(Enum):
    """Query rewrite strategies"""

    EXPANSION = "expansion"
    RELAXATION = "relaxation"
    NO_CHANGE = "no_change"


@dataclass
class RewriteResult:
    """Result of query rewriting"""

    strategy: RewriteStrategy
    rewritten_query: str
    confidence: float
    reason: str
    metadata: Dict[str, Any]
