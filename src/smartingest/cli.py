"""Command-line entry point for running a single document through the pipeline.

Useful for quick local checks and demos without starting the API/frontend:

    python -m smartingest.cli data/samples/invoice_acme.txt
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
import uuid

from smartingest.graph import run_pipeline
from smartingest.logging_config import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a document through SmartIngest.")
    parser.add_argument("path", help="Path to the document to process.")
    parser.add_argument("--mime", help="Override the MIME type.", default=None)
    args = parser.parse_args(argv)

    configure_logging()
    mime = args.mime or mimetypes.guess_type(args.path)[0] or "text/plain"
    result = run_pipeline(uuid.uuid4().hex, args.path, mime)

    print(result.model_dump_json(indent=2))
    return 0 if result.error is None else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
