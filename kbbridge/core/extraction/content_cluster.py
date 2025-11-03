import json
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field

from kbbridge.config.constants import ContentClusterDefaults


class ClusterInfo(BaseModel):
    """Individual cluster information"""

    theme: str = Field(description="The theme or topic of this cluster")
    anchors: List[str] = Field(description="List of anchor strings in this cluster")
    description: str = Field(description="Brief description of the cluster")


class ContentClusterSignature(dspy.Signature):
    """You are an expert at organizing and clustering related content based on themes and topics.

    ## Task
    Given a list of content anchors (titles, headings, or key phrases), group them into logical clusters based on their themes and relationships.

    ## Guidelines
    1. **Thematic Grouping**: Group anchors by similar themes or topics
    2. **Logical Relationships**: Consider how content pieces relate to each other
    3. **Hierarchical Structure**: Organize clusters in a logical hierarchy
    4. **Completeness**: Ensure all anchors are properly categorized

    ## Output Format
    Return clustered content with:
    - clusters: List of cluster objects with theme, anchors, and description
    - total_clusters: Total number of clusters
    - unclustered: Any anchors that don't fit (empty list if all are clustered)"""

    original_query: str = dspy.InputField(
        desc="The original user query that generated these anchors"
    )
    anchor_list_json: str = dspy.InputField(
        desc="JSON string of content anchors to cluster"
    )
    clusters: List[ClusterInfo] = dspy.OutputField(
        desc="List of clusters with theme, anchors, and description"
    )
    total_clusters: int = dspy.OutputField(desc="Total number of clusters")
    unclustered: List[str] = dspy.OutputField(
        desc="Anchors that don't fit in any cluster (empty if all clustered)"
    )


class ContentCluster(dspy.Module):
    """
    Clusters and selects the most relevant group of anchor results using LLM analysis
    """

    def __init__(
        self,
        llm_api_url: str,
        llm_model: str,
        llm_api_token: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        use_cot: bool = False,
    ):
        """
        Initialize the content cluster

        Args:
            llm_api_url: LLM API service URL
            llm_model: LLM model name
            llm_api_token: Optional API token (falls back to default if None)
            llm_temperature: Optional temperature (falls back to default if None)
            llm_timeout: Optional timeout in seconds (falls back to default if None)
            max_tokens: Optional max tokens (falls back to default if None)
            use_cot: Whether to use Chain of Thought reasoning (default: False)
        """
        super().__init__()

        self.llm_api_url = llm_api_url
        self.llm_model = llm_model

        # Use provided values or fall back to defaults
        self.llm_api_token = llm_api_token
        self.llm_temperature = (
            llm_temperature
            if llm_temperature is not None
            else ContentClusterDefaults.TEMPERATURE.value
        )
        self.llm_timeout = (
            llm_timeout
            if llm_timeout is not None
            else ContentClusterDefaults.TIMEOUT_SECONDS.value
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else ContentClusterDefaults.MAX_TOKENS.value
        )

        # Configure DSPy LM
        self._configure_dspy_lm()

        # Initialize predictor with appropriate signature
        if use_cot:
            self.predictor = dspy.ChainOfThought(ContentClusterSignature)
        else:
            self.predictor = dspy.Predict(ContentClusterSignature)

    def _configure_dspy_lm(self):
        """Configure DSPy language model with instance-specific settings"""
        if not self.llm_api_token:
            raise ValueError(
                "LLM API token is required (should be provided by config/service layer)"
            )

        # Create a local LM instance (not global configuration)
        self._lm = dspy.LM(
            model=f"openai/{self.llm_model}",
            api_base=self.llm_api_url,
            api_key=self.llm_api_token,
            temperature=self.llm_temperature,
            max_tokens=self.max_tokens,
            timeout=self.llm_timeout,
        )

    def forward(self, original_query: str, anchor_list_json: str) -> dspy.Prediction:
        """
        DSPy forward method for content clustering

        Args:
            original_query: The original user query
            anchor_list_json: JSON string of anchors to cluster

        Returns:
            DSPy Prediction with clusters, total_clusters, and unclustered fields
        """
        # Use context manager for thread-safe LM configuration
        with dspy.settings.context(lm=self._lm):
            return self.predictor(
                original_query=original_query, anchor_list_json=anchor_list_json
            )

    def cluster(self, anchor_list: List[str], original_query: str) -> Dict[str, Any]:
        """
        Cluster anchor results and select the most relevant group

        Args:
            anchor_list: List of anchor strings to cluster
            original_query: The original user query that generated these anchors

        Returns:
            Dict containing the result or error information
        """
        try:
            # Skip clustering if we have too few anchors
            if len(anchor_list) <= 1:
                return self._build_success_response(
                    anchor_list, original_query, "insufficient_data"
                )

            # Format the anchor list as JSON for the prompt
            anchor_list_json = json.dumps(anchor_list, indent=2)

            # Call DSPy forward method
            result = self.forward(
                original_query=original_query, anchor_list_json=anchor_list_json
            )

            # Extract clusters from DSPy result
            clusters = result.clusters if hasattr(result, "clusters") else []
            total_clusters = (
                result.total_clusters if hasattr(result, "total_clusters") else 0
            )
            result.unclustered if hasattr(result, "unclustered") else []

            # Flatten all anchors from all clusters
            clustered_anchors = []
            for cluster in clusters:
                if hasattr(cluster, "anchors"):
                    clustered_anchors.extend(cluster.anchors)

            # If we got valid clusters, return them
            if clustered_anchors:
                return self._build_success_response(
                    clustered_anchors, original_query, "clustered", total_clusters
                )
            else:
                # Fallback to original list if clustering failed
                return self._build_success_response(
                    anchor_list, original_query, "fallback"
                )

        except Exception as e:
            return self._build_exception_response(e, original_query, anchor_list)

    def _build_base_response(
        self, original_query: str, anchor_count: int
    ) -> Dict[str, Any]:
        """Build base response with common fields"""
        return {
            "original_query": original_query,
            "input_anchor_count": anchor_count,
            "model_used": self.llm_model,
            "tool_type": "content_cluster",
        }

    def _build_success_response(
        self,
        clustered_anchors: List[str],
        original_query: str,
        cluster_method: str,
        total_clusters: int = None,
    ) -> Dict[str, Any]:
        """Build successful response"""
        response = self._build_base_response(original_query, len(clustered_anchors))
        response.update(
            {
                "success": True,
                "clustered_anchors": clustered_anchors,
                "output_anchor_count": len(clustered_anchors),
                "cluster_method": cluster_method,
            }
        )
        if total_clusters is not None:
            response["total_clusters"] = total_clusters
        return response

    def _build_exception_response(
        self, exception: Exception, original_query: str, anchor_list: List[str]
    ) -> Dict[str, Any]:
        """Build error response from unexpected exception"""
        response = self._build_base_response(original_query, len(anchor_list))
        response.update(
            {
                "success": False,
                "error": "unknown_error",
                "message": f"Error clustering anchors: {str(exception)}",
                "fallback_anchors": anchor_list,
            }
        )
        return response
