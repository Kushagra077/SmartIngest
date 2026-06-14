"""In-memory rate limiting for the upload endpoint.

A public demo backed by a free-tier LLM key needs a spend guard: without one,
anyone with the URL can drain the daily quota. This limiter enforces two
fixed-window limits — a *per-client* per-minute cap (fairness / burst control)
and a *global* per-day cap (the hard ceiling that protects the quota).

It is intentionally dependency-free and in-memory: that is the right fit for a
single-instance demo. Horizontal scaling would move the counters into Redis
behind the same :meth:`RateLimiter.check` interface.
"""

from __future__ import annotations

import threading
import time

from smartingest.logging_config import get_logger

logger = get_logger(__name__)

# Cap on how many distinct client buckets we retain before pruning stale ones,
# so a stream of unique IPs can't grow the map without bound.
_MAX_CLIENT_BUCKETS = 10_000


class RateLimitExceeded(Exception):
    """Raised when a request exceeds a configured rate limit."""

    def __init__(self, retry_after: int, message: str) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.message = message


class RateLimiter:
    """Thread-safe fixed-window limiter with per-client and global caps."""

    def __init__(
        self, per_minute: int, per_day: int, enabled: bool = True
    ) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self.enabled = enabled
        self._lock = threading.Lock()
        # client_id -> (minute_window, count)
        self._minute_buckets: dict[str, tuple[int, int]] = {}
        self._day_window = -1
        self._day_count = 0

    def check(self, client_id: str) -> None:
        """Record a request from ``client_id``; raise if a limit is exceeded.

        The global daily cap is checked first (it is the quota-protecting
        ceiling), then the per-client minute cap. Counters are only incremented
        once *both* checks pass, so a rejected request never consumes budget.
        """
        if not self.enabled:
            return

        now = time.time()
        minute = int(now // 60)
        day = int(now // 86400)

        with self._lock:
            if day != self._day_window:
                self._day_window = day
                self._day_count = 0
            if self.per_day and self._day_count >= self.per_day:
                retry = 86400 - int(now % 86400)
                logger.warning("Global daily rate limit reached (%d).", self.per_day)
                raise RateLimitExceeded(retry, "Daily processing limit reached. Try again tomorrow.")

            window, count = self._minute_buckets.get(client_id, (minute, 0))
            if window != minute:
                window, count = minute, 0
            if self.per_minute and count >= self.per_minute:
                retry = 60 - int(now % 60)
                raise RateLimitExceeded(retry, "Too many requests; please slow down.")

            self._minute_buckets[client_id] = (window, count + 1)
            self._day_count += 1
            if len(self._minute_buckets) > _MAX_CLIENT_BUCKETS:
                self._prune(minute)

    def _prune(self, current_minute: int) -> None:
        """Drop buckets from earlier windows (caller holds the lock)."""
        self._minute_buckets = {
            cid: bucket
            for cid, bucket in self._minute_buckets.items()
            if bucket[0] == current_minute
        }
