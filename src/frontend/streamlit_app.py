"""Streamlit frontend for SmartIngest.

Upload a document, watch the pipeline run in real time (by polling the FastAPI
backend), and inspect the extracted fields, validation issues and final
routing decision.

Run with:
    streamlit run src/frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("SMARTINGEST_API_URL", "http://localhost:8000")

ROUTE_STYLE = {
    "auto_approve": ("✅ Auto-approved", "success"),
    "flag_for_review": ("⚠️ Flagged for review", "warning"),
    "reject": ("❌ Rejected", "error"),
}

PIPELINE_STAGES = ["Classifier", "Extractor", "Validator", "Router"]


def upload_document(file) -> str | None:
    """Upload a file and return its job_id, or None on failure."""
    try:
        resp = requests.post(
            f"{API_URL}/upload",
            files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")
        return None
    return resp.json()["job_id"]


def poll_result(job_id: str, placeholder, timeout: float = 60.0) -> dict | None:
    """Poll /status until the job finishes, animating a progress bar."""
    deadline = time.time() + timeout
    progress = placeholder.progress(0, text="Queued…")
    step = 0
    while time.time() < deadline:
        try:
            status = requests.get(f"{API_URL}/status/{job_id}", timeout=10).json()
        except requests.RequestException as exc:
            st.error(f"Status check failed: {exc}")
            return None

        state = status["status"]
        step = min(step + 1, len(PIPELINE_STAGES))
        label = PIPELINE_STAGES[step - 1] if state == "running" else state.title()
        progress.progress(step / (len(PIPELINE_STAGES) + 1), text=f"{label}…")

        if state in ("completed", "failed"):
            progress.progress(1.0, text="Done")
            result = requests.get(f"{API_URL}/results/{job_id}", timeout=10)
            return result.json() if result.status_code == 200 else status
        time.sleep(0.4)

    st.error("Timed out waiting for the pipeline.")
    return None


def render_result(result: dict) -> None:
    """Render the final pipeline result."""
    route = result.get("route")
    label, kind = ROUTE_STYLE.get(route, ("Unknown", "info"))
    getattr(st, kind)(f"{label} — {result.get('route_reason', '')}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Document type", result.get("document_type", "?"))
    col2.metric("Classification conf.", f"{result.get('classification_confidence', 0):.0%}")
    col3.metric("Extraction conf.", f"{result.get('extraction_confidence', 0):.0%}")

    st.subheader("Extracted fields")
    fields = {k: v for k, v in (result.get("fields") or {}).items() if v not in (None, [], {}, "")}
    st.json(fields)

    issues = result.get("validation_issues") or []
    if issues:
        st.subheader("Validation issues")
        for issue in issues:
            icon = "🟥" if issue["severity"] == "error" else "🟨"
            st.write(f"{icon} **{issue['field']}** — {issue['message']}")

    if result.get("retries"):
        st.caption(f"Extractor attempts: {result['retries']}")


def main() -> None:
    st.set_page_config(page_title="SmartIngest", page_icon="📄", layout="centered")
    st.title("📄 SmartIngest")
    st.caption("Agentic document intelligence — classify · extract · validate · route")

    with st.sidebar:
        st.markdown(f"**Backend:** `{API_URL}`")
        try:
            ok = requests.get(f"{API_URL}/healthz", timeout=5).status_code == 200
            if ok:
                st.success("Backend online")
            else:
                st.error("Backend unreachable")
        except requests.RequestException:
            st.error("Backend unreachable")
        st.markdown("---")
        st.markdown(
            "Try a `.txt` invoice like:\n\n"
            "```\nINVOICE\nVendor: Acme Corp\nInvoice #: INV-1\n"
            "Date: 2026-01-15\n1 x Item @ $50.00 = $50.00\nTotal: $50.00\n```"
        )

    uploaded = st.file_uploader(
        "Upload a document (PDF, image, or text)",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
    )
    if uploaded and st.button("Run pipeline", type="primary"):
        job_id = upload_document(uploaded)
        if job_id:
            st.caption(f"Job ID: `{job_id}`")
            result = poll_result(job_id, st.empty())
            if result:
                render_result(result)


if __name__ == "__main__":
    main()
