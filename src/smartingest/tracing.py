"""LangSmith tracing wiring.

LangGraph/LangChain emit traces when the standard ``LANGSMITH_*`` environment
variables are set. We translate our typed settings into those variables once at
startup so every node run shows up as a traced step in LangSmith.
"""

from __future__ import annotations

import os

from smartingest.config import Settings
from smartingest.logging_config import get_logger

logger = get_logger(__name__)


def configure_tracing(settings: Settings) -> None:
    """Enable LangSmith tracing from settings, if configured.

    Sets the environment variables LangChain reads. Safe to call repeatedly.
    """
    if not settings.langsmith_tracing:
        logger.debug("LangSmith tracing disabled.")
        return

    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING is on but no API key is set; disabling.")
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith tracing enabled (project=%s).", settings.langsmith_project)
