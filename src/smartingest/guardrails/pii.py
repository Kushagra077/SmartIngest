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

from smartingest.models import SecurityCategory, SecurityFinding

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
