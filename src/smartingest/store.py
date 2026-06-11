"""SQLite-backed job state store.

The API returns a ``job_id`` immediately and a background worker runs the
graph asynchronously; the client polls for status. This store is the shared
state between those two halves. SQLite keeps the project zero-dependency to run
locally while presenting the same interface a Redis-backed store would.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from smartingest.logging_config import get_logger
from smartingest.models import JobStatus, PipelineResult

logger = get_logger(__name__)


class JobStore:
    """Thread-safe SQLite store for ingestion jobs."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the worker thread and request threads share
        # this connection; all access is serialised by self._lock.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("JobStore initialised at %s", db_path)

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id     TEXT PRIMARY KEY,
                    filename   TEXT NOT NULL,
                    status     TEXT NOT NULL,
                    result     TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def create(self, job_id: str, filename: str) -> None:
        """Insert a new job in the QUEUED state."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, filename, status, result, created_at, updated_at)"
                " VALUES (?, ?, ?, NULL, ?, ?)",
                (job_id, filename, JobStatus.QUEUED.value, now, now),
            )
            self._conn.commit()
        logger.info("Created job %s for %s", job_id, filename)

    def set_status(self, job_id: str, status: JobStatus) -> None:
        """Update only the status of a job."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status.value, datetime.utcnow().isoformat(), job_id),
            )
            self._conn.commit()

    def save_result(self, job_id: str, result: PipelineResult) -> None:
        """Persist the final pipeline result and its terminal status."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, result = ?, updated_at = ? WHERE job_id = ?",
                (
                    result.status.value,
                    result.model_dump_json(),
                    datetime.utcnow().isoformat(),
                    job_id,
                ),
            )
            self._conn.commit()
        logger.info("Saved result for job %s (status=%s)", job_id, result.status.value)

    def get_status(self, job_id: str) -> JobStatus | None:
        """Return the current status, or ``None`` if the job is unknown."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return JobStatus(row["status"]) if row else None

    def get_result(self, job_id: str) -> PipelineResult | None:
        """Return the stored result, or ``None`` if absent/not finished."""
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not row or not row["result"]:
            return None
        return PipelineResult.model_validate(json.loads(row["result"]))

    def exists(self, job_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return row is not None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
