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


def test_pipeline_handles_binary_document_without_crashing(tmp_path, monkeypatch):
    # A binary (image/PDF) upload must flow through the graph cleanly in mock
    # mode — the text-based guardrails skip it instead of choking on the bytes.
    monkeypatch.setenv("SMARTINGEST_MOCK_LLM", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    doc = tmp_path / "scan.png"
    doc.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00binary image bytes\x00")
    result = run_pipeline("job-3", str(doc), "image/png")

    assert result.status == JobStatus.COMPLETED  # no crash
    assert result.route is not None  # a routing decision was reached
    # Mock can't read images, so no grounding findings are raised on the noise.
    assert not any(f.category.value == "grounding" for f in result.security_findings)
