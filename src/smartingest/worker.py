"""Background worker that executes the pipeline for queued jobs.

The API hands work off here so that ``POST /upload`` can return a ``job_id``
immediately. A simple thread pool keeps the project dependency-light; the
interface (submit a job, results land in the store) mirrors what a Celery/RQ
worker would provide.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from smartingest.graph import run_pipeline
from smartingest.logging_config import get_logger
from smartingest.models import JobStatus, PipelineResult
from smartingest.store import JobStore

logger = get_logger(__name__)


class PipelineWorker:
    """Runs ingestion jobs off the request thread via a thread pool."""

    def __init__(self, store: JobStore, max_workers: int = 4) -> None:
        self._store = store
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ingest"
        )

    def submit(self, job_id: str, file_path: str, mime_type: str) -> None:
        """Queue a job for asynchronous processing."""
        self._executor.submit(self._run, job_id, file_path, mime_type)
        logger.info("[%s] Job submitted to worker pool.", job_id)

    def _run(self, job_id: str, file_path: str, mime_type: str) -> None:
        """Execute one job, persisting status transitions and the result."""
        self._store.set_status(job_id, JobStatus.RUNNING)
        try:
            result = run_pipeline(job_id, file_path, mime_type)
        except Exception as exc:  # noqa: BLE001 - never let a job kill the worker
            logger.exception("[%s] Pipeline crashed: %s", job_id, exc)
            result = PipelineResult(
                job_id=job_id, status=JobStatus.FAILED, error=str(exc)
            )
        self._store.save_result(job_id, result)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
