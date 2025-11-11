from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from kbbridge.config.constants import AssistantDefaults


class HealthStatus(str, Enum):
    """Server health status"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class SearchMethod(str, Enum):
    """Search method types"""

    HYBRID_SEARCH = "hybrid_search"
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"


@dataclass
class ServerInfo:
    """Server information model"""

    name: str
    version: str
    status: HealthStatus
    uptime: Optional[float] = None
    tools: List[str] = None


@dataclass
class ToolInfo:
    """Tool information model"""

    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str


@dataclass
class ConfigInfo:
    """Configuration information model"""

    dify_endpoint: Optional[str] = None
    llm_api_url: Optional[str] = None
    llm_model: Optional[str] = None
    max_workers: int = AssistantDefaults.MAX_WORKERS.value
    verbose: bool = AssistantDefaults.VERBOSE.value


@dataclass
class AuthRequest:
    """Authentication request model"""

    dify_endpoint: str
    dify_api_key: str
    llm_api_url: str
    llm_model: str
    llm_api_token: Optional[str] = None
    rerank_url: Optional[str] = None
    rerank_model: Optional[str] = None


@dataclass
class SearchRequest:
    """Base search request model"""

    query: str
    dataset_id: str
    verbose: bool = False


@dataclass
class HybridSearchRequest(SearchRequest):
    """Hybrid search request model"""

    max_keywords: int = 5
    top_k_per_keyword: int = 3
    rerank_threshold: float = 0.7
    relevance_score_threshold: float = 0.5
    max_workers: int = AssistantDefaults.MAX_WORKERS.value


@dataclass
class CoreSearchRequest(SearchRequest):
    """Core search request model"""

    max_workers: int = AssistantDefaults.MAX_WORKERS.value
    use_content_booster: bool = True
    max_boost_keywords: int = AssistantDefaults.MAX_BOOST_KEYWORDS.value
    adaptive_top_k_enabled: bool = AssistantDefaults.ADAPTIVE_TOP_K_ENABLED.value
    total_segment_budget: int = AssistantDefaults.TOTAL_SEGMENT_BUDGET.value
    score_threshold: Optional[float] = None
    top_k: int = AssistantDefaults.TOP_K.value


@dataclass
class ParseRequest:
    """Query parsing request model"""

    query: str
    max_sets: int = 5


@dataclass
class ParsedQuery:
    """Parsed query model"""

    original_query: str
    keyword_sets: List[List[str]]
    intent: Optional[str] = None
    entities: List[str] = None


@dataclass
class SearchResponse:
    """Base search response model"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CoreSearchResponse(SearchResponse):
    """Core search response model"""

    answer: Optional[str] = None
    candidates: List[Dict[str, Any]] = None
    debug_info: List[str] = None
    profiling: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingConfig:
    """Configuration for KB Assistant processing"""

    dataset_id: str
    query: str
    verbose: bool = False
    score_threshold: Optional[float] = None
    top_k: int = AssistantDefaults.TOP_K.value
    max_workers: int = AssistantDefaults.MAX_WORKERS.value
    use_content_booster: bool = True
    max_boost_keywords: int = AssistantDefaults.MAX_BOOST_KEYWORDS.value
    adaptive_top_k_enabled: bool = AssistantDefaults.ADAPTIVE_TOP_K_ENABLED.value
    total_segment_budget: int = AssistantDefaults.TOTAL_SEGMENT_BUDGET.value


@dataclass
class Credentials:
    """Backend-agnostic credentials for various services"""

    retrieval_endpoint: str  # Generic retrieval backend (Dify, OpenSearch, etc.)
    retrieval_api_key: str
    llm_api_url: str
    llm_model: str
    llm_api_token: Optional[str] = None
    llm_temperature: float = AssistantDefaults.LLM_TEMPERATURE.value
    llm_timeout: Optional[int] = None
    rerank_url: Optional[str] = None
    rerank_model: Optional[str] = None

    def is_reranking_available(self) -> bool:
        """Check if reranking is available based on credentials."""
        return bool(self.rerank_url and self.rerank_model)


@dataclass
class WorkerDistribution:
    """Distribution of workers across different processing levels"""

    dataset_workers: int
    approach_workers: int
    file_workers: int


@dataclass
class CandidateAnswer:
    """
    Standardized candidate answer structure.

    This is a backend-agnostic model representing an answer candidate
    from any retrieval system (Dify, OpenSearch, Pinecone, etc.).

    Fields:
        source: Processing source ("direct" or "advanced")
        answer: The extracted answer text
        success: Whether the extraction was successful
        dataset_id: Knowledge base/collection/index identifier
        file_name: Source document name (optional)
        display_source: Human-readable source citation (optional)
        metadata: Additional backend-specific metadata (optional)
    """

    source: str  # "direct" or "advanced"
    answer: str
    success: bool
    dataset_id: str = ""
    file_name: str = ""
    display_source: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility"""
        return {
            "source": self.source,
            "answer": self.answer,
            "success": self.success,
            "dataset_id": self.dataset_id,
            "file_name": self.file_name,
            "display_source": self.display_source,
            **(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateAnswer":
        """Create from dictionary for backward compatibility"""
        known_fields = {
            "source",
            "answer",
            "success",
            "dataset_id",
            "file_name",
            "display_source",
        }
        metadata = {k: v for k, v in data.items() if k not in known_fields}
        return cls(
            source=data.get("source", ""),
            answer=data.get("answer", ""),
            success=data.get("success", False),
            dataset_id=data.get("dataset_id", ""),
            file_name=data.get("file_name", ""),
            display_source=data.get("display_source", ""),
            metadata=metadata if metadata else None,
        )


@dataclass
class DatasetResult:
    """Result from processing a single dataset"""

    dataset_id: str
    direct_result: Dict[str, Any]
    advanced_result: Dict[str, Any]
    candidates: List[Dict[str, Any]]  # TODO: Migrate to List[CandidateAnswer]
    debug_info: List[str]
    profiling: Optional[Dict[str, Any]] = None
