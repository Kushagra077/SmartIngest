"""Router agent — turns validation results into a final routing action.

Per the architecture, routing is a **deterministic conditional decision**, not
an LLM call. Given the document type, confidences and validation issues it
chooses exactly one of: auto-approve, flag-for-review, or reject-with-reason.
"""

from __future__ import annotations

from smartingest.config import Settings, get_settings
from smartingest.logging_config import get_logger
from smartingest.models import DocumentType, ExtractedFields, RouteDecision
from smartingest.rules import Rules, get_rules
from smartingest.state import AgentState

logger = get_logger(__name__)


def route_decision(
    state: AgentState,
    rules: Rules,
    settings: Settings,
) -> tuple[RouteDecision, str]:
    """Compute the routing decision and a human-readable reason.

    Decision order (first match wins):
      1. Unknown type or upstream error  -> REJECT
      2. Any error-severity issue         -> REJECT
      3. Warnings, or amount over the
         auto-approve cap, or borderline
         confidence                       -> FLAG_FOR_REVIEW
      4. Otherwise                        -> AUTO_APPROVE
    """
    issues = state.get("validation_issues", [])
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    doc_type = state.get("document_type", DocumentType.UNKNOWN)
    fields = state.get("fields", ExtractedFields())

    if state.get("error"):
        return RouteDecision.REJECT, f"Pipeline error: {state['error']}"

    if doc_type == DocumentType.UNKNOWN:
        return RouteDecision.REJECT, "Document type could not be determined."

    if errors:
        summary = "; ".join(f"{i.field}: {i.message}" for i in errors)
        return RouteDecision.REJECT, f"Validation errors: {summary}"

    # Large invoices always need a human, even when otherwise clean.
    max_amount = settings_max_amount(rules)
    if (
        doc_type == DocumentType.INVOICE
        and fields.total_amount is not None
        and max_amount is not None
        and fields.total_amount > max_amount
    ):
        return (
            RouteDecision.FLAG_FOR_REVIEW,
            f"Invoice total {fields.total_amount} exceeds auto-approve cap {max_amount}.",
        )

    if warnings:
        summary = "; ".join(f"{i.field}: {i.message}" for i in warnings)
        return RouteDecision.FLAG_FOR_REVIEW, f"Warnings require review: {summary}"

    if state.get("classification_confidence", 1.0) < settings.smartingest_min_confidence:
        return (
            RouteDecision.FLAG_FOR_REVIEW,
            "Classification confidence below threshold; manual confirmation advised.",
        )

    return RouteDecision.AUTO_APPROVE, "All checks passed."


def settings_max_amount(rules: Rules) -> float | None:
    """Pull the invoice auto-approve cap from the thresholds block."""
    value = rules.thresholds.get("invoice_auto_approve_max_amount")
    return float(value) if value is not None else None


def router_node(
    state: AgentState,
    rules: Rules | None = None,
    settings: Settings | None = None,
) -> AgentState:
    """Terminal node: record the routing decision in the state.

    Returns:
        A partial state update with ``route`` and ``route_reason``.
    """
    settings = settings or get_settings()
    rules = rules or get_rules(settings.smartingest_rules_path)

    decision, reason = route_decision(state, rules, settings)
    logger.info("[%s] Routed to %s: %s", state.get("job_id", "?"), decision.value, reason)
    return {"route": decision, "route_reason": reason}
