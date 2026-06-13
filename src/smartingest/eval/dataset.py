"""Golden dataset loading for evaluation.

The dataset is a JSONL file where each line is one labeled example: the input
document plus the expected document type, a subset of expected fields, and the
expected routing decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smartingest.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "data/eval/golden.jsonl"


@dataclass
class EvalExample:
    """A single labeled evaluation example."""

    name: str
    file: str
    mime_type: str
    expected_type: str
    expected_fields: dict[str, Any] = field(default_factory=dict)
    expected_route: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalExample":
        return cls(
            name=data["name"],
            file=data["file"],
            mime_type=data.get("mime_type", "text/plain"),
            expected_type=data["expected_type"],
            expected_fields=data.get("expected_fields", {}),
            expected_route=data.get("expected_route"),
        )


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[EvalExample]:
    """Load and parse the JSONL golden dataset.

    Args:
        path: Path to the JSONL dataset file.

    Returns:
        A list of :class:`EvalExample`.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found at {path}")

    examples: list[EvalExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(EvalExample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.error("Skipping malformed dataset line %d: %s", line_no, exc)

    logger.info("Loaded %d eval example(s) from %s", len(examples), path)
    return examples
