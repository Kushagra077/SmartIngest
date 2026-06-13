"""Security guardrails for the SmartIngest pipeline.

Documents are *untrusted input* fed to an LLM, so this package provides:

* :mod:`injection`   — detect prompt-injection attempts in document text
* :mod:`pii`         — detect and redact personally identifiable information
* :mod:`input_validation` — validate uploaded files (size, type)
* :mod:`grounding`   — verify extracted values actually appear in the source

Each detector returns :class:`~smartingest.models.SecurityFinding` objects so
findings flow through the graph and influence the final routing decision.
"""

from smartingest.guardrails.grounding import check_grounding
from smartingest.guardrails.injection import scan_for_injection
from smartingest.guardrails.input_validation import (
    FileValidationError,
    validate_upload,
)
from smartingest.guardrails.pii import (
    detect_pii,
    detect_sensitive_fields,
    mask_value,
    redact_pii,
)

__all__ = [
    "scan_for_injection",
    "detect_pii",
    "detect_sensitive_fields",
    "mask_value",
    "redact_pii",
    "validate_upload",
    "FileValidationError",
    "check_grounding",
]
