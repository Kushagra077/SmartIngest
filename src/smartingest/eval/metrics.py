"""Evaluation metrics for the document pipeline.

Metrics intentionally mirror what a client cares about:

* **classification accuracy** — right document type?
* **field-level precision/recall/F1** — are extracted fields correct?
* **routing accuracy** — right action (approve/flag/reject)?
* **confidence calibration** — does low confidence predict errors?

All metrics operate on plain Python structures so they're trivially testable
and reusable as LangSmith evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smartingest.eval.dataset import EvalExample
from smartingest.models import PipelineResult


def _norm(value: Any) -> str:
    """Normalise a value for forgiving comparison (case/whitespace/number)."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return " ".join(str(value).split()).lower()


def field_match(expected: Any, actual: Any) -> bool:
    """Whether an expected field value matches the extracted one."""
    return _norm(expected) == _norm(actual)


@dataclass
class FieldScore:
    """Precision/recall/F1 for field extraction over the dataset."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalReport:
    """Aggregate evaluation results across the dataset."""

    n: int = 0
    classification_correct: int = 0
    routing_correct: int = 0
    routing_total: int = 0
    field_score: FieldScore = field(default_factory=FieldScore)
    per_example: list[dict[str, Any]] = field(default_factory=list)

    @property
    def classification_accuracy(self) -> float:
        return self.classification_correct / self.n if self.n else 0.0

    @property
    def routing_accuracy(self) -> float:
        return self.routing_correct / self.routing_total if self.routing_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "classification_accuracy": round(self.classification_accuracy, 3),
            "routing_accuracy": round(self.routing_accuracy, 3),
            "field_precision": round(self.field_score.precision, 3),
            "field_recall": round(self.field_score.recall, 3),
            "field_f1": round(self.field_score.f1, 3),
        }


def classification_accuracy(example: EvalExample, result: PipelineResult) -> bool:
    """Single-example classification correctness."""
    return result.document_type.value == example.expected_type


def routing_accuracy(example: EvalExample, result: PipelineResult) -> bool | None:
    """Single-example routing correctness (None if no label)."""
    if example.expected_route is None:
        return None
    return (result.route.value if result.route else None) == example.expected_route


def field_extraction_metrics(example: EvalExample, result: PipelineResult) -> FieldScore:
    """Per-example field precision/recall accounting.

    For each *expected* field: a correct value is a true positive, a wrong or
    missing value is a false negative; we don't penalise extra extracted fields
    here since the dataset only labels a high-value subset.
    """
    score = FieldScore()
    actual = result.fields.model_dump()
    for name, expected_value in example.expected_fields.items():
        if field_match(expected_value, actual.get(name)):
            score.true_positive += 1
        else:
            score.false_negative += 1
    return score


def evaluate_all(
    pairs: list[tuple[EvalExample, PipelineResult]],
) -> EvalReport:
    """Aggregate metrics across all (example, result) pairs."""
    report = EvalReport()
    for example, result in pairs:
        report.n += 1

        cls_ok = classification_accuracy(example, result)
        report.classification_correct += int(cls_ok)

        route_ok = routing_accuracy(example, result)
        if route_ok is not None:
            report.routing_total += 1
            report.routing_correct += int(route_ok)

        fs = field_extraction_metrics(example, result)
        report.field_score.true_positive += fs.true_positive
        report.field_score.false_positive += fs.false_positive
        report.field_score.false_negative += fs.false_negative

        report.per_example.append(
            {
                "name": example.name,
                "classification_ok": cls_ok,
                "routing_ok": route_ok,
                "field_f1": round(fs.f1, 3),
                "predicted_type": result.document_type.value,
                "predicted_route": result.route.value if result.route else None,
            }
        )
    return report
