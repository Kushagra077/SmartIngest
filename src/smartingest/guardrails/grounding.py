"""Grounding check — a cheap hallucination guard.

After extraction, verify that key string/number values actually appear in the
source document. An extracted ``total_amount`` or ``invoice_number`` that is
nowhere in the source is a strong hallucination signal and should be reviewed
rather than auto-approved.

This only runs meaningfully on text-readable sources; for binary documents
(images/PDF passed straight to vision) the source text may be empty, in which
case grounding is skipped.
"""

from __future__ import annotations

import re

from smartingest.models import ExtractedFields, SecurityCategory, SecurityFinding

# Fields worth grounding (high-impact, usually copied verbatim from the doc).
_GROUNDED_STRING_FIELDS = ["vendor_name", "invoice_number", "candidate_name", "id_number"]
_GROUNDED_NUMERIC_FIELDS = ["total_amount", "contract_value"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def check_grounding(fields: ExtractedFields, source_text: str) -> list[SecurityFinding]:
    """Flag extracted values that do not appear in the source text.

    Args:
        fields: Structured fields produced by the Extractor.
        source_text: Text of the source document (empty for opaque binaries).

    Returns:
        ``warning``-severity findings for ungrounded values (empty if all
        present, or if the source text is unavailable).
    """
    if not source_text:
        return []

    haystack = _normalize(source_text)
    findings: list[SecurityFinding] = []

    for name in _GROUNDED_STRING_FIELDS:
        value = getattr(fields, name, None)
        if value and _normalize(str(value)) not in haystack:
            findings.append(
                SecurityFinding(
                    category=SecurityCategory.GROUNDING,
                    message=f"Extracted '{name}' not found verbatim in source.",
                    severity="warning",
                    detail=f"{name}={value}",
                )
            )

    for name in _GROUNDED_NUMERIC_FIELDS:
        value = getattr(fields, name, None)
        if value is None:
            continue
        # Match the number ignoring thousands separators / trailing zeros.
        digits = re.sub(r"[^\d]", "", f"{value:.2f}").lstrip("0") or "0"
        source_digits = re.sub(r"[^\d]", "", haystack)
        if digits not in source_digits:
            findings.append(
                SecurityFinding(
                    category=SecurityCategory.GROUNDING,
                    message=f"Extracted '{name}' value not found in source.",
                    severity="warning",
                    detail=f"{name}={value}",
                )
            )

    return findings
