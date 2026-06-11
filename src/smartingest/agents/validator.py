"""Validator agent — applies deterministic business rules to extracted fields.

Unlike the other agents this node makes **no LLM call**. It runs cheap,
auditable, deterministic checks (required fields, totals, dates, vendor
whitelist) sourced from ``config/rules.yaml``. Determinism here is a feature:
routing decisions must be reproducible and explainable.
"""

from __future__ import annotations

from datetime import date

from smartingest.config import Settings, get_settings
from smartingest.logging_config import get_logger
from smartingest.models import DocumentType, ExtractedFields, ValidationIssue
from smartingest.rules import Rules, get_rules
from smartingest.state import AgentState

logger = get_logger(__name__)


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
    """Invoice-specific checks: total reconciliation and vendor whitelist."""
    issues: list[ValidationIssue] = []

    # Reconcile line-item sum against the stated total.
    if fields.line_items and fields.total_amount is not None:
        line_sum = round(sum(item.amount for item in fields.line_items), 2)
        tolerance = rules.doc_type_rules("invoice").get("total_tolerance", 0.01)
        if abs(line_sum - fields.total_amount) > tolerance:
            issues.append(
                ValidationIssue(
                    field="total_amount",
                    message=(
                        f"Line items sum to {line_sum} but total is "
                        f"{fields.total_amount} (tolerance {tolerance})."
                    ),
                )
            )

    # Vendor whitelist is a soft check (warning, not a hard error).
    if not rules.is_vendor_whitelisted(fields.vendor_name):
        issues.append(
            ValidationIssue(
                field="vendor_name",
                message=f"Vendor '{fields.vendor_name}' is not in the whitelist.",
                severity="warning",
            )
        )
    return issues


def _check_dates(fields: ExtractedFields) -> list[ValidationIssue]:
    """Validate that any populated date fields parse as ISO ``YYYY-MM-DD``."""
    issues: list[ValidationIssue] = []
    date_fields = ["invoice_date", "due_date", "effective_date", "date_of_birth"]
    for name in date_fields:
        value = getattr(fields, name, None)
        if not value:
            continue
        try:
            date.fromisoformat(value)
        except (ValueError, TypeError):
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
        A partial state update containing ``validation_issues``.
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

    errors = [i for i in issues if i.severity == "error"]
    logger.info(
        "[%s] Validation produced %d issue(s) (%d error, %d warning)",
        job_id,
        len(issues),
        len(errors),
        len(issues) - len(errors),
    )
    return {"validation_issues": issues}


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
