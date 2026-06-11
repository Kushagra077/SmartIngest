"""Tests for the evaluation harness (dataset, metrics, runner)."""

from __future__ import annotations

from smartingest.eval.dataset import EvalExample, load_dataset
from smartingest.eval.langsmith_eval import (
    classification_evaluator,
    field_f1_evaluator,
    routing_evaluator,
)
from smartingest.eval.metrics import (
    field_extraction_metrics,
    field_match,
    evaluate_all,
)
from smartingest.eval.runner import run_evaluation
from smartingest.models import (
    DocumentType,
    ExtractedFields,
    PipelineResult,
    RouteDecision,
)


def test_load_golden_dataset():
    examples = load_dataset("data/eval/golden.jsonl")
    assert len(examples) == 5
    names = {e.name for e in examples}
    assert "invoice_acme" in names
    assert all(isinstance(e, EvalExample) for e in examples)


def test_field_match_normalises():
    assert field_match("Acme Corp", "acme corp")
    assert field_match(200.0, 200)
    assert not field_match("Acme", "Globex")


def test_field_extraction_metrics_counts():
    example = EvalExample(
        name="t",
        file="x",
        mime_type="text/plain",
        expected_type="invoice",
        expected_fields={"vendor_name": "Acme Corp", "total_amount": 100.0},
    )
    result = PipelineResult(
        job_id="t",
        fields=ExtractedFields(vendor_name="Acme Corp", total_amount=999.0),
    )
    score = field_extraction_metrics(example, result)
    assert score.true_positive == 1
    assert score.false_negative == 1


def test_evaluate_all_aggregates():
    example = EvalExample(
        name="t", file="x", mime_type="text/plain",
        expected_type="invoice", expected_fields={"vendor_name": "Acme Corp"},
        expected_route="auto_approve",
    )
    result = PipelineResult(
        job_id="t",
        document_type=DocumentType.INVOICE,
        fields=ExtractedFields(vendor_name="Acme Corp"),
        route=RouteDecision.AUTO_APPROVE,
    )
    report = evaluate_all([(example, result)])
    d = report.as_dict()
    assert d["classification_accuracy"] == 1.0
    assert d["routing_accuracy"] == 1.0
    assert d["field_f1"] == 1.0


def test_langsmith_evaluators():
    outputs = {
        "document_type": "invoice",
        "route": "auto_approve",
        "fields": {"vendor_name": "Acme Corp"},
    }
    reference = {
        "expected_type": "invoice",
        "expected_route": "auto_approve",
        "expected_fields": {"vendor_name": "acme corp"},
    }
    assert classification_evaluator(outputs, reference)["score"] == 1.0
    assert routing_evaluator(outputs, reference)["score"] == 1.0
    assert field_f1_evaluator(outputs, reference)["score"] == 1.0


def test_run_evaluation_end_to_end(monkeypatch):
    monkeypatch.setenv("SMARTINGEST_MOCK_LLM", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from smartingest.config import get_settings

    get_settings.cache_clear()
    report = run_evaluation("data/eval/golden.jsonl")
    get_settings.cache_clear()

    # In mock mode the heuristics handle the curated set perfectly.
    assert report.classification_accuracy == 1.0
    assert report.routing_accuracy == 1.0
