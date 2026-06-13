"""Centralised logging configuration.

Call :func:`configure_logging` once at process start (API server, worker, or
test session). Modules obtain loggers via ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once.

    Args:
        level: Log level name or numeric value. Defaults to the ``LOG_LEVEL``
            environment variable, or ``INFO``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    configure_logging()
    return logging.getLogger(name)
