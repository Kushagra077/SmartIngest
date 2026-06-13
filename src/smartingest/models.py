"""Pydantic data models shared across the pipeline and API.

These models define the *contracts* between agents. The Extractor produces a
:class:`ExtractedFields`, the Validator annotates it, and the Router turns the
whole thing into a :class:`PipelineResult` returned by the API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


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
    # ``line_total`` is the spec name; ``amount`` is kept for back-compat. When
    # both are set ``line_total`` wins. ``effective_total`` resolves the two.
    line_total: float | None = None
    hsn_sac_code: str | None = None  # Indian GST tax classification code
    tax_rate: float | None = None  # per-line tax rate (percent)

    @property
    def effective_total(self) -> float:
        """The authoritative per-line total (``line_total`` if set, else ``amount``)."""
        return self.line_total if self.line_total is not None else self.amount


class WorkExperience(BaseModel):
    """A single role in a resume's work history."""

    company: str = ""
    title: str = ""
    start_date: str | None = None
    end_date: str | None = None
    responsibilities: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """A single education entry on a resume."""

    institution: str = ""
    degree: str | None = None
    field: str | None = None
    graduation_year: str | None = None


class ExtractedFields(BaseModel):
    """Structured fields extracted from a document.

    The schema is intentionally a superset across document types; only the
    fields relevant to a given :class:`DocumentType` are expected to be
    populated. Unused fields stay ``None`` / empty.
    """

    # --- Invoice: header ---
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    po_number: str | None = None
    currency: str | None = None

    # --- Invoice: vendor block ---
    vendor_address: str | None = None
    vendor_tax_id: str | None = None  # GSTIN / VAT / EIN
    vendor_bank_details: str | None = None  # SENSITIVE: account / IBAN / IFSC

    # --- Invoice: buyer block ---
    bill_to: str | None = None
    ship_to: str | None = None

    # --- Invoice: line items + totals ---
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax_amount: float | None = None
    cgst: float | None = None  # Indian GST: central
    sgst: float | None = None  # Indian GST: state
    igst: float | None = None  # Indian GST: integrated
    discount: float | None = None
    shipping: float | None = None
    grand_total: float | None = None
    total_amount: float | None = None  # back-compat alias for grand_total

    # --- Invoice: payment ---
    payment_terms: str | None = None
    payment_method: str | None = None

    # --- Contract: parties ---
    parties: list[str] = Field(default_factory=list)  # back-compat for party_names
    party_names: list[str] = Field(default_factory=list)
    party_roles: dict[str, str] = Field(default_factory=dict)  # name -> role

    # --- Contract: dates / term ---
    effective_date: str | None = None
    expiration_date: str | None = None
    term_length: str | None = None
    renewal_type: str | None = None  # "auto" | "manual"
    notice_period: str | None = None

    # --- Contract: money ---
    contract_value: float | None = None
    payment_schedule: str | None = None

    # --- Contract: key clauses (flags / short snippets) ---
    termination_clause: str | None = None
    liability_cap: str | None = None
    confidentiality: str | None = None
    governing_law: str | None = None
    jurisdiction: str | None = None
    signatures_present: bool | None = None
    signatory_names: list[str] = Field(default_factory=list)

    # --- Resume: identity ---
    candidate_name: str | None = None
    full_name: str | None = None  # shared with ID document
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[str] = Field(default_factory=list)  # LinkedIn / GitHub / portfolio
    years_experience: float | None = None  # back-compat
    total_years_experience: float | None = None

    # --- Resume: history / qualifications ---
    work_history: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

    # --- ID document (whole document is sensitive PII) ---
    id_type: str | None = None  # passport / drivers_license / aadhaar / pan
    id_number: str | None = None  # SENSITIVE: mask in logs
    date_of_birth: str | None = None
    expiry_date: str | None = None
    issuing_authority: str | None = None
    nationality: str | None = None

    # Free-form catch-all for fields outside the typed schema.
    extra: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_aliases(self) -> "ExtractedFields":
        """Keep new field names and their back-compat aliases in lockstep.

        The Extractor may populate either the spec name (``grand_total``,
        ``party_names``, ``total_years_experience``) or the legacy one; mirror
        whichever is set onto the other so every consumer (required-field
        checks, grounding, reconciliation) sees a consistent value.
        """
        if self.grand_total is None and self.total_amount is not None:
            self.grand_total = self.total_amount
        elif self.total_amount is None and self.grand_total is not None:
            self.total_amount = self.grand_total

        if not self.party_names and self.parties:
            self.party_names = list(self.parties)
        elif not self.parties and self.party_names:
            self.parties = list(self.party_names)

        if self.total_years_experience is None and self.years_experience is not None:
            self.total_years_experience = self.years_experience
        elif self.years_experience is None and self.total_years_experience is not None:
            self.years_experience = self.total_years_experience

        return self

    @property
    def effective_total(self) -> float | None:
        """Authoritative invoice total (kept equal to ``grand_total``)."""
        return self.grand_total if self.grand_total is not None else self.total_amount

    @property
    def effective_parties(self) -> list[str]:
        """Authoritative party list (kept equal to ``party_names``)."""
        return self.party_names or self.parties


# Field names whose values are sensitive and must never appear in logs.
SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {"vendor_bank_details", "id_number", "date_of_birth"}
)


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
