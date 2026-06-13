"""Tests for the mock LLM client (classification + extraction heuristics)."""

from __future__ import annotations

from smartingest.llm import MockLLMClient, _safe_json_loads
from smartingest.models import DocumentType


def test_mock_classifies_invoice(sample_invoice):
    client = MockLLMClient()
    result = client.classify(sample_invoice, "text/plain")
    assert result.document_type == DocumentType.INVOICE
    assert result.confidence > 0.5


def test_mock_extracts_invoice_fields(sample_invoice):
    client = MockLLMClient()
    fields, confidence = client.extract(sample_invoice, "text/plain", DocumentType.INVOICE)
    assert fields.vendor_name == "Acme Corp"
    assert fields.invoice_number == "INV-1001"
    assert fields.total_amount == 140.0
    assert len(fields.line_items) == 2
    assert confidence >= 0.75


def test_mock_unknown_for_empty(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("lorem ipsum dolor", encoding="utf-8")
    result = MockLLMClient().classify(str(path), "text/plain")
    assert result.document_type == DocumentType.UNKNOWN


def test_safe_json_loads_strips_code_fence():
    raw = '```json\n{"a": 1}\n```'
    assert _safe_json_loads(raw) == {"a": 1}


def test_safe_json_loads_handles_garbage():
    assert _safe_json_loads("not json at all") == {}
