"""Tests for the SQLite job state store."""

from __future__ import annotations

from smartingest.models import (
    DocumentType,
    JobStatus,
    PipelineResult,
    RouteDecision,
)
from smartingest.store import JobStore


def test_create_and_status(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create("j1", "invoice.pdf")
    assert store.get_status("j1") == JobStatus.QUEUED
    assert store.exists("j1")
    assert not store.exists("nope")
    store.close()


def test_save_and_get_result(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create("j1", "invoice.pdf")
    result = PipelineResult(
        job_id="j1",
        status=JobStatus.COMPLETED,
        document_type=DocumentType.INVOICE,
        route=RouteDecision.AUTO_APPROVE,
    )
    store.save_result("j1", result)

    assert store.get_status("j1") == JobStatus.COMPLETED
    loaded = store.get_result("j1")
    assert loaded is not None
    assert loaded.route == RouteDecision.AUTO_APPROVE
    assert loaded.document_type == DocumentType.INVOICE
    store.close()


def test_unknown_job(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    assert store.get_status("ghost") is None
    assert store.get_result("ghost") is None
    store.close()
