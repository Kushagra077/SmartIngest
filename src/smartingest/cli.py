"""Command-line entry point for running a single document through the pipeline.

Useful for quick local checks and demos without starting the API/frontend:

    python -m smartingest.cli data/samples/invoice_acme.txt
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import uuid

from smartingest.config import get_settings
from smartingest.graph import run_pipeline
from smartingest.guardrails import redact_pii
from smartingest.logging_config import configure_logging


def _force_mock_llm() -> None:
    """Pin the process to offline mock-LLM mode, regardless of ``.env``.

    An actual environment variable overrides any ``.env`` value in
    ``pydantic-settings``; clearing the cache ensures the override is picked up
    even if settings were read earlier in the process.
    """
    os.environ["SMARTINGEST_MOCK_LLM"] = "true"
    get_settings.cache_clear()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a document through SmartIngest.")
    parser.add_argument("path", help="Path to the document to process.")
    parser.add_argument("--mime", help="Override the MIME type.", default=None)
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Mask detected PII (emails, phones, SSNs, cards) in the output.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force offline mock-LLM mode (no API calls), overriding .env.",
    )
    args = parser.parse_args(argv)

    if args.mock:
        _force_mock_llm()

    configure_logging()
    mime = args.mime or mimetypes.guess_type(args.path)[0] or "text/plain"
    result = run_pipeline(uuid.uuid4().hex, args.path, mime)

    output = result.model_dump_json(indent=2)
    print(redact_pii(output) if args.redact else output)
    return 0 if result.error is None else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
