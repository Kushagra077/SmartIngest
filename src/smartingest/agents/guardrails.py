"""Guardrail node — the security entry point of the pipeline.

Runs *before* the Classifier. It reads the document's text (when available) and
applies the input-trust guardrails: prompt-injection scanning and PII
detection. Findings are written to ``security_findings`` so downstream nodes —
especially the Router — can act on them.

The grounding check is a separate, post-extraction guardrail and lives in the
Validator (it needs the extracted fields).
"""

from __future__ import annotations

from smartingest.config import Settings, get_settings
from smartingest.guardrails import detect_pii, read_text_source, scan_for_injection
from smartingest.logging_config import get_logger
from smartingest.state import AgentState

logger = get_logger(__name__)


def guardrail_node(state: AgentState, settings: Settings | None = None) -> AgentState:
    """Scan the input document for injection and PII before processing.

    Returns:
        A partial state update with ``security_findings``.
    """
    settings = settings or get_settings()
    job_id = state.get("job_id", "?")

    if not settings.smartingest_enable_guardrails:
        return {"security_findings": []}

    text = read_text_source(state["file_path"])
    findings = scan_for_injection(text) + detect_pii(text)

    if findings:
        logger.info("[%s] Guardrail scan raised %d finding(s).", job_id, len(findings))
    return {"security_findings": findings}
