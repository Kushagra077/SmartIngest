"""FastAPI backend exposing the SmartIngest pipeline.

Endpoints:
  * ``POST /upload``            — accept a file, return a ``job_id`` immediately
  * ``GET  /status/{job_id}``   — poll job status
  * ``GET  /results/{job_id}``  — fetch the full pipeline result
  * ``GET  /healthz``           — liveness probe

Uploads are processed asynchronously: the route persists the file, creates a
job record, hands it to the background worker, and returns. The client polls
``/status`` and then reads ``/results``.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from smartingest.config import get_settings
from smartingest.guardrails import FileValidationError, validate_upload
from smartingest.logging_config import get_logger
from smartingest.models import (
    JobStatus,
    PipelineResult,
    StatusResponse,
    UploadResponse,
)
from smartingest.ratelimit import RateLimiter, RateLimitExceeded
from smartingest.store import JobStore
from smartingest.tracing import configure_tracing
from smartingest.worker import PipelineWorker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources (store, worker, tracing) on startup."""
    settings = get_settings()
    configure_tracing(settings)
    app.state.settings = settings
    app.state.store = JobStore(settings.smartingest_db_path)
    app.state.worker = PipelineWorker(app.state.store)
    app.state.rate_limiter = RateLimiter(
        per_minute=settings.smartingest_rate_limit_per_minute,
        per_day=settings.smartingest_rate_limit_per_day,
        enabled=settings.smartingest_rate_limit_enabled,
    )
    app.state.upload_dir = Path(settings.smartingest_upload_dir)
    app.state.upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info("SmartIngest API started (mock_llm=%s).", settings.use_mock_llm)
    try:
        yield
    finally:
        app.state.worker.shutdown()
        app.state.store.close()
        logger.info("SmartIngest API shut down.")


app = FastAPI(
    title="SmartIngest",
    description="Agentic document intelligence pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Accept a document and enqueue it for processing.

    Returns immediately with a ``job_id``; processing happens in the
    background.
    """
    # Rate guard: protect the LLM quota before doing any work.
    limiter: RateLimiter = app.state.rate_limiter
    client_id = request.client.host if request.client else "unknown"
    try:
        limiter.check(client_id)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.message,
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc

    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    settings = app.state.settings
    max_bytes = int(settings.smartingest_max_file_size_mb * 1024 * 1024)

    # Reject oversized uploads up front (via the declared size) so we never read
    # an enormous body into memory just to fail the size check afterwards.
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.smartingest_max_file_size_mb} MB limit.",
        )

    contents = await file.read()

    # Input guardrail: re-validate size/type on the actual bytes before
    # persisting or processing (defends against an understated size header).
    try:
        validate_upload(
            file.filename,
            contents,
            file.content_type,
            settings.smartingest_max_file_size_mb,
        )
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    suffix = Path(file.filename).suffix
    dest = app.state.upload_dir / f"{job_id}{suffix}"
    dest.write_bytes(contents)

    store: JobStore = app.state.store
    store.create(job_id, file.filename)

    worker: PipelineWorker = app.state.worker
    mime_type = file.content_type or "application/octet-stream"
    worker.submit(job_id, str(dest), mime_type)

    return UploadResponse(job_id=job_id, status=JobStatus.QUEUED, filename=file.filename)


@app.get("/status/{job_id}", response_model=StatusResponse)
def status(job_id: str) -> StatusResponse:
    """Return the current status of a job."""
    store: JobStore = app.state.store
    job_status = store.get_status(job_id)
    if job_status is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    result = store.get_result(job_id)
    return StatusResponse(
        job_id=job_id,
        status=job_status,
        document_type=result.document_type if result else None,
        route=result.route if result else None,
        error=result.error if result else None,
    )


@app.get("/results/{job_id}", response_model=PipelineResult)
def results(job_id: str) -> PipelineResult:
    """Return the full pipeline result, or 404/409 if not ready."""
    store: JobStore = app.state.store
    if not store.exists(job_id):
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    result = store.get_result(job_id)
    if result is None:
        raise HTTPException(
            status_code=409, detail="Job is still processing; poll /status."
        )
    return result
