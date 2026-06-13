"""Loader for the deterministic business-rules configuration.

The Validator agent consumes these rules. They live in YAML (see
``config/rules.yaml``) so that thresholds and whitelists can be tuned without
touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from smartingest.logging_config import get_logger

logger = get_logger(__name__)


class Rules:
    """Typed accessor over the raw rules YAML document."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data or {}

    @property
    def thresholds(self) -> dict[str, Any]:
        return self._data.get("thresholds", {})

    @property
    def vendor_whitelist(self) -> list[str]:
        return self._data.get("vendor_whitelist", [])

    @property
    def tax_id_formats(self) -> dict[str, str]:
        """Regex patterns for well-formed vendor tax IDs, keyed by scheme."""
        return self._data.get("tax_id_formats", {})

    @property
    def id_number_formats(self) -> dict[str, str]:
        """Regex patterns for well-formed ID numbers, keyed by id_type."""
        return self._data.get("id_number_formats", {})

    def doc_type_rules(self, document_type: str) -> dict[str, Any]:
        """Return the per-document-type rule block (empty if undefined)."""
        return self._data.get("document_types", {}).get(document_type, {})

    def required_fields(self, document_type: str) -> list[str]:
        return self.doc_type_rules(document_type).get("required_fields", [])

    def is_vendor_whitelisted(self, vendor: str | None) -> bool:
        """Case-insensitive substring match against the vendor whitelist."""
        if not vendor:
            return False
        vendor_lc = vendor.lower()
        return any(allowed.lower() in vendor_lc for allowed in self.vendor_whitelist)


def load_rules(path: str | Path) -> Rules:
    """Load and parse the rules YAML file.

    Args:
        path: Path to the YAML rules file.

    Returns:
        A :class:`Rules` instance. If the file is missing, an empty rule set is
        returned and a warning is logged rather than raising — the pipeline
        should still run (everything simply gets flagged for review).
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Rules file not found at %s; using empty rule set", path)
        return Rules({})

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        logger.error("Failed to parse rules file %s: %s", path, exc)
        return Rules({})

    logger.info("Loaded business rules from %s", path)
    return Rules(data)


@lru_cache
def get_rules(path: str) -> Rules:
    """Cached rules loader keyed by path."""
    return load_rules(path)
