"""The typed state object threaded through the LangGraph pipeline.

LangGraph passes a single mutable state dict between nodes. Using a
``TypedDict`` gives us editor/type-checker support and documents exactly which
keys each node may read or write.
"""

from __future__ import annotations

from typing import TypedDict

from smartingest.models import (
    DocumentType,
    ExtractedFields,
    RouteDecision,
    ValidationIssue,
)


class AgentState(TypedDict, total=False):
    """Shared state for the document-processing graph.

    Keys are populated progressively:
      * ``Classifier`` sets ``document_type`` / ``classification_confidence``
      * ``Extractor`` sets ``fields`` / ``extraction_confidence``
      * ``Validator`` sets ``validation_issues`` and may bump ``retries``
      * ``Router`` sets ``route`` / ``route_reason``
    """

    # --- Inputs (set before the graph runs) ---
    job_id: str
    file_path: str
    mime_type: str

    # --- Classifier outputs ---
    document_type: DocumentType
    classification_confidence: float

    # --- Extractor outputs ---
    fields: ExtractedFields
    extraction_confidence: float

    # --- Validator outputs ---
    validation_issues: list[ValidationIssue]
    retries: int

    # --- Router outputs ---
    route: RouteDecision
    route_reason: str

    # --- Error channel ---
    error: str
