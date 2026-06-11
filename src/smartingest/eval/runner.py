"""Local evaluation runner.

Runs every golden example through the real pipeline (in whatever LLM mode is
configured — mock by default) and prints an aggregate report. Exits non-zero
when metrics fall below thresholds, so it doubles as a CI regression gate.

Usage:
    python -m smartingest.eval.runner
    python -m smartingest.eval.runner --dataset data/eval/golden.jsonl --min-routing 0.9
"""

from __future__ import annotations

import argparse
import json
import sys

from smartingest.eval.dataset import DEFAULT_DATASET, EvalExample, load_dataset
from smartingest.eval.metrics import EvalReport, evaluate_all
from smartingest.graph import run_pipeline
from smartingest.logging_config import configure_logging, get_logger
from smartingest.models import PipelineResult

logger = get_logger(__name__)


def run_evaluation(dataset_path: str = DEFAULT_DATASET) -> EvalReport:
    """Execute the pipeline over the dataset and aggregate metrics."""
    examples = load_dataset(dataset_path)
    pairs: list[tuple[EvalExample, PipelineResult]] = []
    for ex in examples:
        result = run_pipeline(ex.name, ex.file, ex.mime_type)
        pairs.append((ex, result))
    return evaluate_all(pairs)


def _print_report(report: EvalReport) -> None:
    print("\n=== SmartIngest evaluation ===")
    print(json.dumps(report.as_dict(), indent=2))
    print("\nPer-example:")
    for row in report.per_example:
        cls = "✓" if row["classification_ok"] else "✗"
        route = {True: "✓", False: "✗", None: "—"}[row["routing_ok"]]
        print(
            f"  [{cls} type | {route} route | F1 {row['field_f1']:.2f}] "
            f"{row['name']}: type={row['predicted_type']}, route={row['predicted_route']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the SmartIngest pipeline.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--min-classification", type=float, default=0.8)
    parser.add_argument("--min-routing", type=float, default=0.8)
    parser.add_argument("--min-field-f1", type=float, default=0.7)
    args = parser.parse_args(argv)

    configure_logging()
    report = run_evaluation(args.dataset)
    _print_report(report)

    failures = []
    if report.classification_accuracy < args.min_classification:
        failures.append(f"classification {report.classification_accuracy:.2f} < {args.min_classification}")
    if report.routing_accuracy < args.min_routing:
        failures.append(f"routing {report.routing_accuracy:.2f} < {args.min_routing}")
    if report.field_score.f1 < args.min_field_f1:
        failures.append(f"field F1 {report.field_score.f1:.2f} < {args.min_field_f1}")

    if failures:
        print("\n❌ Eval gate FAILED: " + "; ".join(failures))
        return 1
    print("\n✅ Eval gate PASSED")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
