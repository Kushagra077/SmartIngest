"""Tests for the in-memory rate limiter."""

from __future__ import annotations

import pytest

from smartingest.ratelimit import RateLimiter, RateLimitExceeded


def test_per_client_minute_cap():
    limiter = RateLimiter(per_minute=2, per_day=1000)
    limiter.check("1.1.1.1")
    limiter.check("1.1.1.1")
    with pytest.raises(RateLimitExceeded) as exc:
        limiter.check("1.1.1.1")
    assert exc.value.retry_after > 0


def test_clients_are_independent():
    limiter = RateLimiter(per_minute=1, per_day=1000)
    limiter.check("a")  # uses a's whole minute budget
    limiter.check("b")  # b is unaffected
    with pytest.raises(RateLimitExceeded):
        limiter.check("a")


def test_global_daily_cap_spans_clients():
    limiter = RateLimiter(per_minute=100, per_day=2)
    limiter.check("a")
    limiter.check("b")
    with pytest.raises(RateLimitExceeded, match="Daily"):
        limiter.check("c")


def test_rejected_request_does_not_consume_global_budget():
    # Per-minute cap of 1 trips on the 2nd call; that rejection must not have
    # spent a slot of the generous daily budget.
    limiter = RateLimiter(per_minute=1, per_day=5)
    limiter.check("a")
    with pytest.raises(RateLimitExceeded):
        limiter.check("a")
    assert limiter._day_count == 1


def test_disabled_limiter_never_blocks():
    limiter = RateLimiter(per_minute=1, per_day=1, enabled=False)
    for _ in range(50):
        limiter.check("a")
