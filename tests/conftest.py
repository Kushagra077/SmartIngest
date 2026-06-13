"""Shared pytest fixtures and path setup for the SmartIngest test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the ``src`` layout is importable when running pytest directly.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smartingest.config import Settings  # noqa: E402
from smartingest.rules import load_rules  # noqa: E402


@pytest.fixture
def rules():
    """Load the project's default business rules."""
    root = Path(__file__).resolve().parents[1]
    return load_rules(root / "config" / "rules.yaml")


@pytest.fixture
def settings(tmp_path):
    """A Settings instance pinned to mock mode and a temp DB/upload dir."""
    return Settings(
        smartingest_mock_llm=True,
        smartingest_db_path=str(tmp_path / "jobs.db"),
        smartingest_upload_dir=str(tmp_path / "uploads"),
        smartingest_min_confidence=0.75,
        smartingest_max_retries=2,
    )


@pytest.fixture
def sample_invoice(tmp_path):
    """Write a clean, whitelisted invoice text document and return its path."""
    content = (
        "INVOICE\n"
        "Vendor: Acme Corp\n"
        "Invoice #: INV-1001\n"
        "Date: 2026-01-15\n"
        "2 x Widget @ $50.00 = $100.00\n"
        "1 x Gadget @ $40.00 = $40.00\n"
        "Total: $140.00\n"
        "Amount due upon receipt.\n"
    )
    path = tmp_path / "invoice.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)
