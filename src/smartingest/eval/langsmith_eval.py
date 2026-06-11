"""Optional LangSmith integration for the evaluation harness.

Pushes the golden dataset to LangSmith and runs the pipeline as an *experiment*
so eval results appear next to production traces, with per-node spans. This is
the "eval sets" capability shown in the architecture diagram.

The local runner (``runner.py``) is the primary, offline path; this module is
used only when LangSmith credentials are configured. ``langsmith`` is imported
lazily so the package has no hard dependency on it at import time.
"""

from __future__ import annotations

from typing import Any, Callable

from smartingest.eval.dataset import DEFAULT_DATASET, load_dataset
from smartingest.eval.metrics import field_match
from smartingest.graph import run_pipeline
from smartingest.logging_config import get_logger
from smartingest.models import PipelineResult

logger = get_logger(__name__)


# --- Evaluators (LangSmith-compatible) -------------------------------------
# Each takes the run outputs and the reference (expected) outputs and returns a
# {"key", "score"} dict. They're plain functions so they're unit-testable.


def classification_evaluator(outputs: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Score 1.0 when the predicted document type matches the label."""
    score = float(outputs.get("document_type") == reference.get("expected_type"))
    return {"key": "classification_accuracy", "score": score}


def routing_evaluator(outputs: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Score 1.0 when the routing decision matches the label."""
    score = float(outputs.get("route") == reference.get("expected_route"))
    return {"key": "routing_accuracy", "score": score}


def field_f1_evaluator(outputs: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Score = fraction of expected fields extracted correctly."""
    expected = reference.get("expected_fields", {})
    actual = outputs.get("fields", {})
    if not expected:
        return {"key": "field_recall", "score": 1.0}
    correct = sum(1 for k, v in expected.items() if field_match(v, actual.get(k)))
    return {"key": "field_recall", "score": correct / len(expected)}


# --- Experiment target -----------------------------------------------------


def pipeline_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the pipeline for one dataset example; returns flat outputs."""
    result: PipelineResult = run_pipeline(
        inputs.get("name", "eval"), inputs["file"], inputs.get("mime_type", "text/plain")
    )
    return {
        "document_type": result.document_type.value,
        "route": result.route.value if result.route else None,
        "fields": result.fields.model_dump(),
    }


# --- Orchestration ---------------------------------------------------------


def run_langsmith_eval(
    dataset_path: str = DEFAULT_DATASET,
    dataset_name: str = "smartingest-golden",
    target: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Any:
    """Upload the dataset (if needed) and run a LangSmith experiment.

    Requires ``langsmith`` installed and ``LANGSMITH_API_KEY`` set. Returns the
    LangSmith experiment results object.
    """
    try:
        from langsmith import Client, evaluate  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("langsmith is not installed; `pip install langsmith`.") from exc

    client = Client()
    examples = load_dataset(dataset_path)

    if not client.has_dataset(dataset_name=dataset_name):
        ds = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[{"name": e.name, "file": e.file, "mime_type": e.mime_type} for e in examples],
            outputs=[
                {"expected_type": e.expected_type, "expected_fields": e.expected_fields,
                 "expected_route": e.expected_route}
                for e in examples
            ],
            dataset_id=ds.id,
        )
        logger.info("Uploaded %d examples to LangSmith dataset '%s'.", len(examples), dataset_name)

    return evaluate(
        target or pipeline_target,
        data=dataset_name,
        evaluators=[classification_evaluator, routing_evaluator, field_f1_evaluator],
        experiment_prefix="smartingest-eval",
    )
