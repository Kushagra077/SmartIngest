"""Integration tests for the FastAPI endpoints using a TestClient.

These exercise the full async upload -> poll -> results flow against the real
graph (in mock-LLM mode), so they cover the worker hand-off too.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from smartingest.api import app
from smartingest.models import JobStatus


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTINGEST_MOCK_LLM", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SMARTINGEST_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("SMARTINGEST_UPLOAD_DIR", str(tmp_path / "uploads"))
    # get_settings is cached; clear so env overrides take effect.
    from smartingest.config import get_settings

    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _wait_for_completion(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/status/{job_id}")
        if resp.json()["status"] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
            return resp.json()
        time.sleep(0.05)
    raise AssertionError("Job did not complete in time")


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_upload_status_results_flow(client):
    files = {
        "file": (
            "invoice.txt",
            b"INVOICE\nVendor: Acme Corp\nInvoice #: INV-9\nDate: 2026-03-01\n"
            b"1 x Item @ $10.00 = $10.00\nTotal: $10.00\n",
            "text/plain",
        )
    }
    upload = client.post("/upload", files=files)
    assert upload.status_code == 200
    job_id = upload.json()["job_id"]
    assert upload.json()["status"] == JobStatus.QUEUED.value

    status = _wait_for_completion(client, job_id)
    assert status["status"] == JobStatus.COMPLETED.value

    result = client.get(f"/results/{job_id}")
    assert result.status_code == 200
    body = result.json()
    assert body["document_type"] == "invoice"
    assert body["route"] == "auto_approve"


def test_upload_empty_file_rejected(client):
    files = {"file": ("empty.txt", b"", "text/plain")}
    assert client.post("/upload", files=files).status_code == 400


def test_rate_limit_returns_429(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTINGEST_MOCK_LLM", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SMARTINGEST_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("SMARTINGEST_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SMARTINGEST_RATE_LIMIT_PER_MINUTE", "1")
    from smartingest.config import get_settings

    get_settings.cache_clear()
    files = {"file": ("invoice.txt", b"INVOICE\nTotal: $1.00\n", "text/plain")}
    with TestClient(app) as c:
        assert c.post("/upload", files=files).status_code == 200
        second = c.post("/upload", files=files)
        assert second.status_code == 429
        assert "Retry-After" in second.headers
    get_settings.cache_clear()


def test_status_unknown_job(client):
    assert client.get("/status/does-not-exist").status_code == 404


def test_results_unknown_job(client):
    assert client.get("/results/does-not-exist").status_code == 404
