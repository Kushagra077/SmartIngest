"""Tests for the deterministic Router decision logic."""

from __future__ import annotations

from smartingest.agents.router import route_decision
from smartingest.models import (
    DocumentType,
    ExtractedFields,
    RouteDecision,
    ValidationIssue,
)


def test_auto_approve_clean_invoice(rules, settings):
    state = {
        "document_type": DocumentType.INVOICE,
        "classification_confidence": 0.95,
        "fields": ExtractedFields(vendor_name="Acme Corp", total_amount=100.0),
        "validation_issues": [],
    }
    decision, _ = route_decision(state, rules, settings)
    assert decision == RouteDecision.AUTO_APPROVE


def test_reject_on_error_issue(rules, settings):
    state = {
        "document_type": DocumentType.INVOICE,
        "classification_confidence": 0.95,
        "fields": ExtractedFields(),
        "validation_issues": [ValidationIssue(field="total_amount", message="bad")],
    }
    decision, reason = route_decision(state, rules, settings)
    assert decision == RouteDecision.REJECT
    assert "total_amount" in reason


def test_flag_on_warning(rules, settings):
    state = {
        "document_type": DocumentType.INVOICE,
        "classification_confidence": 0.95,
        "fields": ExtractedFields(vendor_name="Acme Corp", total_amount=100.0),
        "validation_issues": [
            ValidationIssue(field="vendor_name", message="unknown", severity="warning")
        ],
    }
    decision, _ = route_decision(state, rules, settings)
    assert decision == RouteDecision.FLAG_FOR_REVIEW


def test_flag_on_large_invoice(rules, settings):
    state = {
        "document_type": DocumentType.INVOICE,
        "classification_confidence": 0.95,
        "fields": ExtractedFields(vendor_name="Acme Corp", total_amount=50000.0),
        "validation_issues": [],
    }
    decision, reason = route_decision(state, rules, settings)
    assert decision == RouteDecision.FLAG_FOR_REVIEW
    assert "cap" in reason


def test_reject_unknown_type(rules, settings):
    state = {
        "document_type": DocumentType.UNKNOWN,
        "validation_issues": [],
        "fields": ExtractedFields(),
    }
    decision, _ = route_decision(state, rules, settings)
    assert decision == RouteDecision.REJECT
