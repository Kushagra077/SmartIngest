"""End-to-end tests for the compiled LangGraph pipeline."""

from __future__ import annotations

from smartingest.graph import run_pipeline
from smartingest.models import DocumentType, JobStatus, RouteDecision


def test_pipeline_auto_approves_clean_invoice(sample_invoice, monkeypatch):
    # Force mock mode regardless of ambient environment.
    monkeypatch.setenv("SMARTINGEST_MOCK_LLM", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = run_pipeline("job-1", sample_invoice, "text/plain")

    assert result.status == JobStatus.COMPLETED
    assert result.document_type == DocumentType.INVOICE
    assert result.route == RouteDecision.AUTO_APPROVE
    assert result.fields.vendor_name == "Acme Corp"
    assert result.retries >= 1


def test_pipeline_flags_unknown_vendor(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTINGEST_MOCK_LLM", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    doc = tmp_path / "inv.txt"
    doc.write_text(
        "INVOICE\nVendor: Sketchy Inc\nInvoice #: X9\nDate: 2026-02-01\n"
        "1 x Item @ $20.00 = $20.00\nTotal: $20.00\n",
        encoding="utf-8",
    )
    result = run_pipeline("job-2", str(doc), "text/plain")
    assert result.route == RouteDecision.FLAG_FOR_REVIEW
