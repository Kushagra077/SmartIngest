"""Extractor agent — pulls structured, Pydantic-typed fields from a document.

Runs after the Classifier. The Validator may route control *back* here for a
retry when extraction confidence is low; the ``retries`` counter in the state
tracks how many attempts have been made.
"""

from __future__ import annotations

from smartingest.config import get_settings
from smartingest.llm import LLMClient, LLMError, get_llm_client
from smartingest.logging_config import get_logger
from smartingest.models import DocumentType
from smartingest.state import AgentState

logger = get_logger(__name__)


def extractor_node(state: AgentState, llm: LLMClient | None = None) -> AgentState:
    """Extract structured fields for the classified document type.

    Args:
        state: Current graph state. Must contain ``file_path`` and
            ``document_type``.
        llm: Optional injected LLM client (used in tests).

    Returns:
        A partial state update with ``fields`` and ``extraction_confidence``.
        The ``retries`` counter is incremented so retry loops are observable.
    """
    llm = llm or get_llm_client(get_settings())
    job_id = state.get("job_id", "?")
    doc_type = state.get("document_type", DocumentType.UNKNOWN)
    retries = state.get("retries", 0)

    try:
        fields, confidence = llm.extract(
            state["file_path"], state.get("mime_type", "text/plain"), doc_type
        )
    except LLMError as exc:
        logger.error("[%s] Extraction failed: %s", job_id, exc)
        return {"error": f"extraction_failed: {exc}", "retries": retries}

    logger.info(
        "[%s] Extracted fields for %s (confidence=%.2f, attempt=%d)",
        job_id,
        doc_type.value,
        confidence,
        retries + 1,
    )
    return {
        "fields": fields,
        "extraction_confidence": confidence,
        "retries": retries + 1,
    }
