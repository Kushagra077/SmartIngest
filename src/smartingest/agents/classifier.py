"""Classifier agent — determines a document's type and confidence.

This is the first node in the graph. It calls the multimodal LLM to assign a
:class:`~smartingest.models.DocumentType` and a confidence score.
"""

from __future__ import annotations

from smartingest.config import get_settings
from smartingest.llm import LLMClient, LLMError, get_llm_client
from smartingest.logging_config import get_logger
from smartingest.models import DocumentType
from smartingest.state import AgentState

logger = get_logger(__name__)


def classifier_node(state: AgentState, llm: LLMClient | None = None) -> AgentState:
    """Classify the uploaded document.

    Args:
        state: Current graph state. Must contain ``file_path`` and ``mime_type``.
        llm: Optional injected LLM client (used in tests). When omitted, the
            client is resolved from settings.

    Returns:
        A partial state update with ``document_type`` and
        ``classification_confidence`` (or ``error`` on failure).
    """
    llm = llm or get_llm_client(get_settings())
    job_id = state.get("job_id", "?")

    try:
        result = llm.classify(state["file_path"], state.get("mime_type", "text/plain"))
    except LLMError as exc:
        logger.error("[%s] Classification failed: %s", job_id, exc)
        return {"error": f"classification_failed: {exc}"}

    logger.info(
        "[%s] Classified as %s (confidence=%.2f)",
        job_id,
        result.document_type.value,
        result.confidence,
    )
    return {
        "document_type": result.document_type,
        "classification_confidence": result.confidence,
    }
