"""Pydantic data models shared across the pipeline and API.

These models define the *contracts* between agents. The Extractor produces a
:class:`ExtractedFields`, the Validator annotates it, and the Router turns the
whole thing into a :class:`PipelineResult` returned by the API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Document categories the Classifier can assign."""

    INVOICE = "invoice"
    CONTRACT = "contract"
    RESUME = "resume"
    ID_DOCUMENT = "id_document"
    UNKNOWN = "unknown"


class JobStatus(str, Enum):
    """Lifecycle states of an ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RouteDecision(str, Enum):
    """Final routing actions produced by the Router."""

    AUTO_APPROVE = "auto_approve"
    FLAG_FOR_REVIEW = "flag_for_review"
    REJECT = "reject"


class LineItem(BaseModel):
    """A single invoice line item."""

    description: str = ""
    quantity: float = 1.0
    unit_price: float = 0.0
    amount: float = 0.0


class ExtractedFields(BaseModel):
    """Structured fields extracted from a document.

    The schema is intentionally a superset across document types; only the
    fields relevant to a given :class:`DocumentType` are expected to be
    populated. Unused fields stay ``None`` / empty.
    """

    # Invoice
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    total_amount: float | None = None
    currency: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)

    # Contract
    parties: list[str] = Field(default_factory=list)
    effective_date: str | None = None
    contract_value: float | None = None

    # Resume
    candidate_name: str | None = None
    email: str | None = None
    phone: str | None = None
    years_experience: float | None = None

    # ID document
    full_name: str | None = None
    id_number: str | None = None
    date_of_birth: str | None = None

    # Free-form catch-all for fields outside the typed schema.
    extra: dict[str, str] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    """A single problem discovered by the Validator."""

    field: str
    message: str
    severity: str = "error"  # "error" | "warning"


class SecurityCategory(str, Enum):
    """Categories of security/guardrail findings."""

    INPUT = "input"  # file-level validation (size, type)
    INJECTION = "injection"  # prompt-injection attempt in document text
    PII = "pii"  # personally identifiable information detected
    GROUNDING = "grounding"  # extracted value not found in source (hallucination)


class SecurityFinding(BaseModel):
    """A single guardrail finding raised during processing."""

    category: SecurityCategory
    message: str
    severity: str = "warning"  # "error" | "warning"
    detail: str = ""


class ClassificationResult(BaseModel):
    """Output of the Classifier agent."""

    document_type: DocumentType = DocumentType.UNKNOWN
    confidence: float = 0.0
    reasoning: str = ""


class PipelineResult(BaseModel):
    """The complete result of running a document through the graph."""

    job_id: str
    status: JobStatus = JobStatus.COMPLETED
    document_type: DocumentType = DocumentType.UNKNOWN
    classification_confidence: float = 0.0
    extraction_confidence: float = 0.0
    fields: ExtractedFields = Field(default_factory=ExtractedFields)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    security_findings: list[SecurityFinding] = Field(default_factory=list)
    route: RouteDecision | None = None
    route_reason: str = ""
    retries: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UploadResponse(BaseModel):
    """Response returned immediately from ``POST /upload``."""

    job_id: str
    status: JobStatus
    filename: str


class StatusResponse(BaseModel):
    """Response returned from ``GET /status/{job_id}``."""

    job_id: str
    status: JobStatus
    document_type: DocumentType | None = None
    route: RouteDecision | None = None
    error: str | None = None
