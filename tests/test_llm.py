"""Tests for the mock LLM client (classification + extraction heuristics)."""

from __future__ import annotations

import pytest

from smartingest.config import Settings
from smartingest.llm import (
    GeminiClient,
    LLMError,
    MockLLMClient,
    _fields_from_payload,
    _safe_json_loads,
)
from smartingest.models import DocumentType, ExtractedFields


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


def test_null_in_nested_field_does_not_discard_siblings():
    """A null inside a nested list entry must not sink the whole extraction.

    Regression: extractors emit ``"institution": null`` for absent values; that
    once raised a validation error and discarded every good top-level field.
    """
    fields = ExtractedFields.model_validate(
        {
            "candidate_name": "Jane Doe",
            "email": "jane.doe@example.com",
            "education": [{"institution": None, "degree": "B.Sc."}],
        }
    )
    assert fields.candidate_name == "Jane Doe"
    assert fields.email == "jane.doe@example.com"
    assert fields.education[0].institution == ""
    assert fields.education[0].degree == "B.Sc."


class _FakeModels:
    """Stand-in for ``client.models`` that scripts per-model responses."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[str] = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.behaviour[model]
        if isinstance(outcome, Exception):
            raise outcome
        return type("R", (), {"text": outcome})()


def _client_with(behaviour, models):
    client = GeminiClient(api_key="test-key", models=models)
    # ``client.models`` is read-only on the real SDK Client, so swap the whole
    # client for a fake exposing just the ``.models`` attribute we use.
    client._client = type("FakeClient", (), {"models": _FakeModels(behaviour)})()
    return client


def test_model_chain_dedups_and_orders():
    s = Settings(gemini_model="gemini-2.0-flash", gemini_model_fallbacks="gemini-1.5-flash, gemini-2.0-flash ,x")
    assert s.gemini_model_chain == ["gemini-2.0-flash", "gemini-1.5-flash", "x"]


def test_gemini_fails_over_on_quota(tmp_path):
    from google.genai import errors

    doc = tmp_path / "d.txt"
    doc.write_text("INVOICE", encoding="utf-8")
    quota = errors.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
    client = _client_with(
        {"primary": quota, "backup": '{"document_type": "invoice", "confidence": 0.9}'},
        models=["primary", "backup"],
    )

    result = client.classify(str(doc), "text/plain")
    assert client._client.models.calls == ["primary", "backup"]
    assert result.document_type == DocumentType.INVOICE


def test_gemini_non_quota_error_does_not_fail_over(tmp_path):
    from google.genai import errors

    doc = tmp_path / "d.txt"
    doc.write_text("INVOICE", encoding="utf-8")
    bad_request = errors.ClientError(400, {"error": {"status": "INVALID_ARGUMENT"}})
    client = _client_with(
        {"primary": bad_request, "backup": '{"document_type": "invoice"}'},
        models=["primary", "backup"],
    )

    with pytest.raises(LLMError):
        client.classify(str(doc), "text/plain")
    assert client._client.models.calls == ["primary"]  # no failover on 400


def test_gemini_raises_when_all_models_exhausted(tmp_path):
    from google.genai import errors

    doc = tmp_path / "d.txt"
    doc.write_text("INVOICE", encoding="utf-8")
    quota = errors.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
    client = _client_with({"a": quota, "b": quota}, models=["a", "b"])

    with pytest.raises(LLMError, match="exhausted"):
        client.classify(str(doc), "text/plain")
    assert client._client.models.calls == ["a", "b"]


def test_fields_from_payload_salvages_malformed_field():
    """A genuinely malformed value drops only its own key, not the rest."""
    fields, confidence = _fields_from_payload(
        {
            "confidence": 0.9,
            "vendor_name": "Acme Corp",
            "line_items": "this is not a list",  # malformed -> dropped
        }
    )
    assert fields.vendor_name == "Acme Corp"
    assert fields.line_items == []
    assert confidence == 0.9
