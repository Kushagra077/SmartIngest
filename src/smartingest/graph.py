"""LangGraph assembly for the SmartIngest pipeline.

Wires the four agent nodes into a ``StateGraph``:

    classifier -> extractor -> validator --(low conf?)--> extractor   (retry)
                                          \\--(ok)-------> router -> END

The Validator -> Extractor edge is the conditional retry loop; the Router is
the terminal node that records the final decision.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from smartingest.agents.classifier import classifier_node
from smartingest.agents.extractor import extractor_node
from smartingest.agents.router import router_node
from smartingest.agents.validator import needs_retry, validator_node
from smartingest.logging_config import get_logger
from smartingest.models import (
    DocumentType,
    ExtractedFields,
    JobStatus,
    PipelineResult,
)
from smartingest.state import AgentState

logger = get_logger(__name__)


def build_graph():
    """Construct and compile the document-processing state graph."""
    builder = StateGraph(AgentState)

    builder.add_node("classifier", classifier_node)
    builder.add_node("extractor", extractor_node)
    builder.add_node("validator", validator_node)
    builder.add_node("router", router_node)

    builder.set_entry_point("classifier")
    builder.add_edge("classifier", "extractor")
    builder.add_edge("extractor", "validator")

    # Conditional retry loop: validator decides extractor (retry) or router.
    builder.add_conditional_edges(
        "validator",
        needs_retry,
        {"extractor": "extractor", "router": "router"},
    )
    builder.add_edge("router", END)

    return builder.compile()


# Compile once at import; the graph is stateless and reusable across jobs.
_GRAPH = None


def get_graph():
    """Return the lazily-compiled, cached graph instance."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
        logger.info("Compiled SmartIngest LangGraph pipeline.")
    return _GRAPH


def run_pipeline(job_id: str, file_path: str, mime_type: str) -> PipelineResult:
    """Run a single document end-to-end through the graph.

    Args:
        job_id: Identifier used for logging and the result record.
        file_path: Path to the document on disk.
        mime_type: MIME type of the document.

    Returns:
        A fully-populated :class:`PipelineResult`.
    """
    graph = get_graph()
    initial: AgentState = {
        "job_id": job_id,
        "file_path": file_path,
        "mime_type": mime_type,
        "retries": 0,
    }

    logger.info("[%s] Starting pipeline for %s", job_id, file_path)
    final_state: AgentState = graph.invoke(initial)
    return _state_to_result(job_id, final_state)


def _state_to_result(job_id: str, state: AgentState) -> PipelineResult:
    """Convert terminal graph state into the public result model."""
    error = state.get("error")
    return PipelineResult(
        job_id=job_id,
        status=JobStatus.FAILED if error else JobStatus.COMPLETED,
        document_type=state.get("document_type", DocumentType.UNKNOWN),
        classification_confidence=state.get("classification_confidence", 0.0),
        extraction_confidence=state.get("extraction_confidence", 0.0),
        fields=state.get("fields", ExtractedFields()),
        validation_issues=state.get("validation_issues", []),
        route=state.get("route"),
        route_reason=state.get("route_reason", ""),
        retries=state.get("retries", 0),
        error=error,
    )
