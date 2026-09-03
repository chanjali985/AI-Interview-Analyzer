"""Background analysis worker.

Interview analysis takes tens of seconds, so the HTTP request that submits an
interview only enqueues it. A small thread pool drains the queue; interview
state lives in the database, so a restart can pick up anything left in flight.

For multi-instance deployments swap `enqueue` for a Celery/RQ task without
touching the callers.
"""
from __future__ import annotations

import atexit
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from .config import settings

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()
_in_flight: dict[int, Future] = {}


def get_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max(1, settings.ANALYSIS_WORKERS),
                thread_name_prefix="analysis",
            )
            atexit.register(shutdown)
    return _executor


def enqueue(interview_id: int) -> None:
    """Queue an interview for analysis (idempotent while one is in flight)."""
    from .services.pipeline import process_interview

    with _lock:
        existing = _in_flight.get(interview_id)
        if existing is not None and not existing.done():
            logger.info("Interview %s is already queued", interview_id)
            return

    def run() -> None:
        try:
            process_interview(interview_id)
        finally:
            with _lock:
                _in_flight.pop(interview_id, None)

    future = get_executor().submit(run)
    with _lock:
        _in_flight[interview_id] = future
    logger.info("Queued interview %s for analysis", interview_id)


def pending_count() -> int:
    with _lock:
        return sum(1 for future in _in_flight.values() if not future.done())


def requeue_unfinished() -> int:
    """On startup, pick up interviews left SUBMITTED or PROCESSING by a restart."""
    from .database import session_scope
    from .models import Interview, InterviewStatus

    with session_scope() as db:
        stuck = (
            db.query(Interview.id)
            .filter(Interview.status.in_([InterviewStatus.SUBMITTED, InterviewStatus.PROCESSING]))
            .all()
        )
        ids = [row[0] for row in stuck]

    for interview_id in ids:
        enqueue(interview_id)
    if ids:
        logger.info("Requeued %s unfinished interview(s) after restart", len(ids))
    return len(ids)


def shutdown(wait: bool = True) -> None:
    global _executor
    with _lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=wait)
