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


def test_zero_total_is_not_treated_as_missing(rules, settings):
    """A legitimate 0.00 total must not be flagged as a missing required field."""
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        total_amount=0.0,
    )
    out = validator_node(state, rules=rules, settings=settings)
    missing = [
        i for i in out["validation_issues"]
        if i.field == "total_amount" and "missing" in i.message
    ]
    assert missing == []


def test_total_mismatch_flagged(rules, settings):
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        total_amount=999.0,
        line_items=[LineItem(description="x", amount=100.0)],
    )
    out = validator_node(state, rules=rules, settings=settings)
    assert any(i.field == "grand_total" for i in out["validation_issues"])


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


def test_due_date_before_invoice_date_flagged(rules, settings):
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        due_date="2026-01-01",
        total_amount=100.0,
    )
    out = validator_node(state, rules=rules, settings=settings)
    assert any(i.field == "due_date" for i in out["validation_issues"])


def test_reconciliation_includes_tax_and_shipping(rules, settings):
    # 100 (line) + 18 tax + 10 shipping - 5 discount = 123 == grand_total
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        grand_total=123.0,
        tax_amount=18.0,
        shipping=10.0,
        discount=5.0,
        line_items=[LineItem(description="x", line_total=100.0)],
    )
    out = validator_node(state, rules=rules, settings=settings)
    assert not any(i.field == "grand_total" for i in out["validation_issues"])


def test_malformed_tax_id_warns(rules, settings):
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        total_amount=100.0,
        vendor_tax_id="NOT-A-REAL-ID",
    )
    out = validator_node(state, rules=rules, settings=settings)
    tax = [i for i in out["validation_issues"] if i.field == "vendor_tax_id"]
    assert tax and tax[0].severity == "warning"


def test_contract_single_party_flagged(rules, settings):
    state = {
        "job_id": "t",
        "document_type": DocumentType.CONTRACT,
        "fields": ExtractedFields(party_names=["Acme Corp"], effective_date="2026-01-01"),
    }
    out = validator_node(state, rules=rules, settings=settings)
    assert any(i.field == "party_names" for i in out["validation_issues"])


def test_contract_expiration_before_effective_flagged(rules, settings):
    state = {
        "job_id": "t",
        "document_type": DocumentType.CONTRACT,
        "fields": ExtractedFields(
            party_names=["A", "B"],
            effective_date="2026-01-01",
            expiration_date="2025-01-01",
        ),
    }
    out = validator_node(state, rules=rules, settings=settings)
    assert any(i.field == "expiration_date" for i in out["validation_issues"])


def test_expired_id_flagged(rules, settings):
    state = {
        "job_id": "t",
        "document_type": DocumentType.ID_DOCUMENT,
        "fields": ExtractedFields(
            full_name="Jane Doe", id_number="X1234567", expiry_date="2000-01-01"
        ),
    }
    out = validator_node(state, rules=rules, settings=settings)
    assert any(i.field == "expiry_date" for i in out["validation_issues"])


def test_sensitive_field_raises_finding(rules, settings):
    state = _invoice_state(
        vendor_name="Acme Corp",
        invoice_number="INV-1",
        invoice_date="2026-01-15",
        total_amount=100.0,
        vendor_bank_details="IBAN GB33BUKB20201555555555",
    )
    out = validator_node(state, rules=rules, settings=settings)
    findings = out.get("security_findings", [])
    assert any("vendor_bank_details" in f.detail for f in findings)


def test_needs_retry_when_low_confidence(settings):
    state = {"extraction_confidence": 0.4, "retries": 0}
    assert needs_retry(state, settings) == "extractor"


def test_no_retry_when_budget_exhausted(settings):
    state = {"extraction_confidence": 0.4, "retries": settings.smartingest_max_retries}
    assert needs_retry(state, settings) == "router"


def test_no_retry_when_confident(settings):
    state = {"extraction_confidence": 0.95, "retries": 0}
    assert needs_retry(state, settings) == "router"
