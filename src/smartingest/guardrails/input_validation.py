"""Upload input validation.

Rejects files that are too large or of a disallowed type *before* any bytes are
written to disk or sent to the LLM — the cheapest and most reliable guardrail.
"""

from __future__ import annotations

from pathlib import Path

from smartingest.logging_config import get_logger

logger = get_logger(__name__)

# Extensions we are willing to process.
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".tiff", ".webp"}

# MIME prefixes we accept (documents and images, plus plain text).
ALLOWED_MIME_PREFIXES = ("image/", "application/pdf", "text/")


class FileValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_upload(
    filename: str,
    content: bytes,
    mime_type: str | None,
    max_size_mb: float,
) -> None:
    """Validate an uploaded file. Raises :class:`FileValidationError` on failure.

    Args:
        filename: Original client filename.
        content: Raw file bytes.
        mime_type: Declared MIME type (may be ``None``).
        max_size_mb: Maximum allowed size in megabytes.
    """
    if not content:
        raise FileValidationError("Uploaded file is empty.")

    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise FileValidationError(
            f"File is {size_mb:.1f} MB, exceeding the {max_size_mb} MB limit."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{suffix or '(none)'}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    if mime_type and not mime_type.startswith(ALLOWED_MIME_PREFIXES):
        # Declared MIME mismatch is suspicious but extensions are the stronger
        # signal; log rather than hard-fail on MIME alone.
        logger.warning(
            "Declared MIME '%s' for %s is outside the allowed set.", mime_type, filename
        )
