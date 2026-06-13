"""Tests for the deterministic Validator agent and retry logic."""

from __future__ import annotations

from smartingest.agents.validator import needs_retry, validator_node
from smartingest.models import DocumentType, ExtractedFields, LineItem


def _invoice_state(**fields):
    return {
        "job_id": "t",
        "document_type": DocumentType.INVOICE,
        "fields": ExtractedFields(**fields),
    }


def test_valid_invoice_has_no_errors(rules, settings):
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        total_amount=100.0,
        line_items=[LineItem(description="x", amount=100.0)],
    )
    out = validator_node(state, rules=rules, settings=settings)
    errors = [i for i in out["validation_issues"] if i.severity == "error"]
    assert errors == []


def test_missing_required_field_flagged(rules, settings):
    state = _invoice_state(vendor_name="Acme Corp", total_amount=10.0)
    out = validator_node(state, rules=rules, settings=settings)
    fields_flagged = {i.field for i in out["validation_issues"]}
    assert "invoice_number" in fields_flagged
    assert "invoice_date" in fields_flagged


def test_total_mismatch_flagged(rules, settings):
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        total_amount=999.0,
        line_items=[LineItem(description="x", amount=100.0)],
    )
    out = validator_node(state, rules=rules, settings=settings)
    assert any(i.field == "total_amount" for i in out["validation_issues"])


def test_non_whitelisted_vendor_is_warning(rules, settings):
    state = _invoice_state(
        vendor_name="Shady LLC",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        total_amount=100.0,
        line_items=[LineItem(description="x", amount=100.0)],
    )
    out = validator_node(state, rules=rules, settings=settings)
    vendor_issues = [i for i in out["validation_issues"] if i.field == "vendor_name"]
    assert vendor_issues and vendor_issues[0].severity == "warning"


def test_invalid_date_flagged(rules, settings):
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="15/01/2026",
        total_amount=100.0,
    )
    out = validator_node(state, rules=rules, settings=settings)
    assert any(i.field == "invoice_date" for i in out["validation_issues"])


def test_needs_retry_when_low_confidence(settings):
    state = {"extraction_confidence": 0.4, "retries": 0}
    assert needs_retry(state, settings) == "extractor"


def test_no_retry_when_budget_exhausted(settings):
    state = {"extraction_confidence": 0.4, "retries": settings.smartingest_max_retries}
    assert needs_retry(state, settings) == "router"


def test_no_retry_when_confident(settings):
    state = {"extraction_confidence": 0.95, "retries": 0}
    assert needs_retry(state, settings) == "router"
