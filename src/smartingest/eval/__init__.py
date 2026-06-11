"""Evaluation harness for the SmartIngest pipeline.

Provides a labeled golden dataset, extraction/classification/routing metrics,
and a local runner. The same evaluators can be pushed to LangSmith (see
``langsmith_eval``) so eval runs show up alongside production traces.
"""

from smartingest.eval.dataset import EvalExample, load_dataset
from smartingest.eval.metrics import (
    EvalReport,
    classification_accuracy,
    field_extraction_metrics,
    routing_accuracy,
)

__all__ = [
    "EvalExample",
    "load_dataset",
    "EvalReport",
    "classification_accuracy",
    "field_extraction_metrics",
    "routing_accuracy",
]
