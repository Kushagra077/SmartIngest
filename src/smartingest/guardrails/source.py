"""Shared helper for reading a document's *text* source.

Text-based guardrails (injection scan, PII detection, grounding) only make
sense on text documents. Images and PDFs are passed straight to the vision
model, so there is no meaningful text to scan — reading their bytes as UTF-8
yields garbage that would otherwise produce false findings (e.g. grounding
flagging every extracted value as "not found in source").

:func:`read_text_source` returns the decoded text for genuine text files and an
empty string for binary documents, so callers can simply skip on ``""``.
"""

from __future__ import annotations

from pathlib import Path


def read_text_source(file_path: str) -> str:
    """Return a document's text, or ``""`` if it is binary or unreadable.

    Binary documents (images, PDFs) are detected by the presence of NUL bytes,
    which never occur in normal text files.
    """
    if not file_path:
        return ""
    try:
        data = Path(file_path).read_bytes()
    except OSError:
        return ""
    if b"\x00" in data:  # image / PDF / other binary — not groundable text
        return ""
    return data.decode("utf-8", errors="ignore")
