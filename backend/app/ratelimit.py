"""A small in-process sliding-window rate limiter.

Deliberately dependency-free and per-process: it protects a single instance
from an abusive client. Behind several replicas put a shared limiter (Redis,
or your ingress/WAF) in front of the app as well.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from .config import settings

_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check(key: str, limit: int, window_seconds: int = 3600) -> bool:
    """Return True when the call is allowed."""
    now = time.time()
    with _lock:
        bucket = _hits[key]
        while bucket and now - bucket[0] > window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def reset() -> None:
    with _lock:
        _hits.clear()


def limit_submissions(request: Request) -> None:
    """FastAPI dependency guarding the candidate upload/submit endpoints."""
    if not settings.RATE_LIMIT_ENABLED:
        return
    key = f"submit:{client_key(request)}"
    if not check(key, settings.RATE_LIMIT_SUBMISSIONS_PER_HOUR):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a few minutes and try again.",
        )
