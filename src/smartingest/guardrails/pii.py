"""PII detection and redaction.

Detects common PII (emails, phone numbers, SSNs, credit-card numbers) using
regex. :func:`redact_pii` masks matches so PII never leaks into logs or stored
artifacts; :func:`detect_pii` reports *which categories* were seen without
echoing the sensitive values themselves.

Regex keeps the project dependency-free. For higher recall (names, addresses)
swap in Microsoft Presidio behind the same interface.
"""

from __future__ import annotations

import re

from smartingest.models import (
    SENSITIVE_FIELD_NAMES,
    ExtractedFields,
    SecurityCategory,
    SecurityFinding,
)

# (label, pattern, redaction placeholder)
_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(r"[\w.\-]+@[\w.\-]+\.\w+"), "[EMAIL]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]?){13,16}\b"),
        "[CARD]",
    ),
    (
        "phone",
        re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b"),
        "[PHONE]",
    ),
]


def detect_pii(text: str) -> list[SecurityFinding]:
    """Report which PII categories appear in ``text`` (values not echoed).

    Args:
        text: Text to scan.

    Returns:
        One ``warning``-severity :class:`SecurityFinding` per detected category.
    """
    if not text:
        return []

    findings: list[SecurityFinding] = []
    for label, pattern, _ in _PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            findings.append(
                SecurityFinding(
                    category=SecurityCategory.PII,
                    message=f"Detected {len(matches)} {label} value(s).",
                    severity="warning",
                    detail=f"category={label}",
                )
            )
    return findings


def redact_pii(text: str) -> str:
    """Return ``text`` with detected PII replaced by category placeholders."""
    if not text:
        return text
    for _, pattern, placeholder in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def mask_value(value: str | None) -> str:
    """Mask a sensitive value for safe logging (keep last 4 chars only)."""
    if not value:
        return ""
    s = str(value)
    if len(s) <= 4:
        return "*" * len(s)
    return "*" * (len(s) - 4) + s[-4:]


def detect_sensitive_fields(fields: ExtractedFields) -> list[SecurityFinding]:
    """Report populated sensitive structured fields (values never echoed).

    Bank details, ID numbers and dates of birth are flagged so downstream
    consumers know to handle the record carefully. Reported as ``PII`` so they
    drive awareness/redaction rather than the routing decision.
    """
    findings: list[SecurityFinding] = []
    for name in sorted(SENSITIVE_FIELD_NAMES):
        if getattr(fields, name, None):
            findings.append(
                SecurityFinding(
                    category=SecurityCategory.PII,
                    message=f"Sensitive field '{name}' is present and must be handled securely.",
                    severity="warning",
                    detail=f"field={name}",
                )
            )
    return findings
