"""Validator agent — applies deterministic business rules to extracted fields.

Unlike the other agents this node makes **no LLM call**. It runs cheap,
auditable, deterministic checks (required fields, totals, dates, vendor
whitelist) sourced from ``config/rules.yaml``. Determinism here is a feature:
routing decisions must be reproducible and explainable.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from smartingest.config import Settings, get_settings
from smartingest.guardrails import check_grounding, detect_sensitive_fields
from smartingest.logging_config import get_logger
from smartingest.models import DocumentType, ExtractedFields, ValidationIssue
from smartingest.rules import Rules, get_rules
from smartingest.state import AgentState

logger = get_logger(__name__)


def _parse_iso(value: str | None) -> date | None:
    """Parse a ``YYYY-MM-DD`` string to a date, or None if absent/invalid."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _read_source_text(file_path: str) -> str:
    """Best-effort text read of the source document for grounding."""
    if not file_path:
        return ""
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:  # pragma: no cover - defensive
        return ""


def _check_required_fields(
    fields: ExtractedFields, required: list[str]
) -> list[ValidationIssue]:
    """Flag any required field that is missing or empty."""
    issues: list[ValidationIssue] = []
    for name in required:
        value = getattr(fields, name, None)
        if value in (None, "", [], 0):
            issues.append(
                ValidationIssue(field=name, message=f"Required field '{name}' is missing.")
            )
    return issues


def _check_invoice(fields: ExtractedFields, rules: Rules) -> list[ValidationIssue]:
    """Invoice checks: total reconciliation, due-date order, tax-id, whitelist."""
    issues: list[ValidationIssue] = []
    total = fields.effective_total

    # Reconcile the grand total. Invoices vary in whether line totals and the
    # stated subtotal are tax-inclusive or tax-exclusive, so we accept the total
    # if *any* common interpretation reconciles within tolerance:
    #   (a) subtotal + tax + shipping - discount        (canonical pre-tax base)
    #   (b) sum(line_totals) + tax + shipping - discount (line totals excl. tax)
    #   (c) sum(line_totals) + shipping - discount       (line totals incl. tax)
    if total is not None and (fields.line_items or fields.subtotal is not None):
        line_sum = round(sum(item.effective_total for item in fields.line_items), 2)
        tax = fields.tax_amount or 0.0
        shipping = fields.shipping or 0.0
        discount = fields.discount or 0.0
        tolerance = rules.doc_type_rules("invoice").get("total_tolerance", 0.01)

        candidates: list[float] = []
        if fields.subtotal is not None:
            candidates.append(round(fields.subtotal + tax + shipping - discount, 2))
        if fields.line_items:
            candidates.append(round(line_sum + tax + shipping - discount, 2))
            candidates.append(round(line_sum + shipping - discount, 2))

        if candidates and not any(abs(c - total) <= tolerance for c in candidates):
            issues.append(
                ValidationIssue(
                    field="grand_total",
                    message=(
                        f"Grand total {total} does not reconcile with components "
                        f"(candidates {sorted(set(candidates))}, tolerance {tolerance})."
                    ),
                )
            )

    # due_date must not precede invoice_date.
    invoice_d = _parse_iso(fields.invoice_date)
    due_d = _parse_iso(fields.due_date)
    if invoice_d and due_d and due_d < invoice_d:
        issues.append(
            ValidationIssue(
                field="due_date",
                message=f"due_date {fields.due_date} precedes invoice_date {fields.invoice_date}.",
            )
        )

    # Vendor tax id should be well-formed against one of the known schemes.
    if fields.vendor_tax_id:
        patterns = rules.tax_id_formats.values()
        normalized = fields.vendor_tax_id.replace(" ", "").upper()
        if patterns and not any(re.match(p, normalized) for p in patterns):
            issues.append(
                ValidationIssue(
                    field="vendor_tax_id",
                    message=f"Tax id '{fields.vendor_tax_id}' does not match a known GSTIN/PAN/VAT/EIN format.",
                    severity="warning",
                )
            )

    # Vendor whitelist is a soft check (warning, not a hard error). A missing
    # vendor is already an error via required-fields, so only warn when present.
    if fields.vendor_name and not rules.is_vendor_whitelisted(fields.vendor_name):
        issues.append(
            ValidationIssue(
                field="vendor_name",
                message=f"Vendor '{fields.vendor_name}' is not in the whitelist.",
                severity="warning",
            )
        )
    return issues


def _check_contract(fields: ExtractedFields) -> list[ValidationIssue]:
    """Contract checks: date order, both parties present, signature presence."""
    issues: list[ValidationIssue] = []

    eff = _parse_iso(fields.effective_date)
    exp = _parse_iso(fields.expiration_date)
    if eff and exp and exp <= eff:
        issues.append(
            ValidationIssue(
                field="expiration_date",
                message=f"expiration_date {fields.expiration_date} is not after effective_date {fields.effective_date}.",
            )
        )

    if len(fields.effective_parties) < 2:
        issues.append(
            ValidationIssue(
                field="party_names",
                message="A contract must name at least two parties.",
            )
        )

    # Missing signatures are suspicious but not necessarily invalid -> warning.
    if fields.signatures_present is False:
        issues.append(
            ValidationIssue(
                field="signatures_present",
                message="No signatures detected on the contract.",
                severity="warning",
            )
        )
    return issues


def _check_id_document(fields: ExtractedFields, rules: Rules) -> list[ValidationIssue]:
    """ID checks: not expired, and id_number well-formed for its id_type."""
    issues: list[ValidationIssue] = []

    expiry = _parse_iso(fields.expiry_date)
    if expiry and expiry < date.today():
        issues.append(
            ValidationIssue(
                field="expiry_date",
                message=f"ID expired on {fields.expiry_date}.",
            )
        )

    if fields.id_type and fields.id_number:
        pattern = rules.id_number_formats.get(fields.id_type.lower())
        if pattern and not re.match(pattern, fields.id_number.strip().upper()):
            issues.append(
                ValidationIssue(
                    field="id_number",
                    message=f"ID number does not match the expected format for {fields.id_type}.",
                    severity="warning",
                )
            )
    return issues


def _check_dates(fields: ExtractedFields) -> list[ValidationIssue]:
    """Validate that any populated date fields parse as ISO ``YYYY-MM-DD``."""
    issues: list[ValidationIssue] = []
    date_fields = [
        "invoice_date",
        "due_date",
        "effective_date",
        "expiration_date",
        "date_of_birth",
        "expiry_date",
    ]
    for name in date_fields:
        value = getattr(fields, name, None)
        if not value:
            continue
        if _parse_iso(value) is None:
            issues.append(
                ValidationIssue(
                    field=name,
                    message=f"Field '{name}' value '{value}' is not a valid ISO date.",
                )
            )
    return issues


def validator_node(
    state: AgentState,
    rules: Rules | None = None,
    settings: Settings | None = None,
) -> AgentState:
    """Validate extracted fields against business rules.

    Args:
        state: Current graph state. Must contain ``fields`` and ``document_type``.
        rules: Optional injected rule set (used in tests).
        settings: Optional injected settings (used in tests).

    Returns:
        A partial state update containing ``validation_issues`` and any
        appended grounding ``security_findings``.
    """
    settings = settings or get_settings()
    rules = rules or get_rules(settings.smartingest_rules_path)
    job_id = state.get("job_id", "?")
    doc_type = state.get("document_type", DocumentType.UNKNOWN)
    fields = state.get("fields", ExtractedFields())

    issues = _check_required_fields(fields, rules.required_fields(doc_type.value))
    issues += _check_dates(fields)
    if doc_type == DocumentType.INVOICE:
        issues += _check_invoice(fields, rules)
    elif doc_type == DocumentType.CONTRACT:
        issues += _check_contract(fields)
    elif doc_type == DocumentType.ID_DOCUMENT:
        issues += _check_id_document(fields, rules)

    errors = [i for i in issues if i.severity == "error"]
    logger.info(
        "[%s] Validation produced %d issue(s) (%d error, %d warning)",
        job_id,
        len(issues),
        len(errors),
        len(issues) - len(errors),
    )

    # Post-extraction guardrail: ensure extracted values are grounded in the
    # source. Appended to the existing security findings from the entry scan.
    updates: AgentState = {"validation_issues": issues}
    if settings.smartingest_enable_guardrails:
        source_text = _read_source_text(state.get("file_path", ""))
        extra_findings = check_grounding(fields, source_text)
        extra_findings += detect_sensitive_fields(fields)
        if extra_findings:
            existing = list(state.get("security_findings", []))
            logger.info(
                "[%s] Post-extraction guardrails raised %d finding(s).",
                job_id,
                len(extra_findings),
            )
            updates["security_findings"] = existing + extra_findings

    return updates


def needs_retry(state: AgentState, settings: Settings | None = None) -> str:
    """Conditional-edge function: decide whether to re-run the Extractor.

    A retry happens only when extraction confidence is below the configured
    floor *and* we still have retry budget left. Otherwise control proceeds to
    the Router.

    Returns:
        ``"extractor"`` to retry, or ``"router"`` to continue.
    """
    settings = settings or get_settings()
    confidence = state.get("extraction_confidence", 1.0)
    retries = state.get("retries", 0)

    if confidence < settings.smartingest_min_confidence and retries < settings.smartingest_max_retries:
        logger.info(
            "[%s] Low confidence (%.2f < %.2f), retrying extraction (attempt %d).",
            state.get("job_id", "?"),
            confidence,
            settings.smartingest_min_confidence,
            retries + 1,
        )
        return "extractor"
    return "router"
