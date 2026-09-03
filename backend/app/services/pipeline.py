"""End-to-end processing of a submitted interview: analyse, persist, notify."""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import joinedload

from ..database import session_scope
from ..models import Answer, Interview, InterviewStatus
from .analyzer import AnalysisError, AnswerInput, analyze_interview
from .notifications import send_candidate_report, send_failure_alert, send_recruiter_report, try_send

logger = logging.getLogger(__name__)


def _load(db, interview_id: int) -> Interview | None:
    return (
        db.query(Interview)
        .options(
            joinedload(Interview.candidate),
            joinedload(Interview.role),
            joinedload(Interview.answers).joinedload(Answer.question),
        )
        .filter(Interview.id == interview_id)
        .one_or_none()
    )


def process_interview(interview_id: int) -> None:
    """Analyse one submitted interview and email the results. Never raises."""
    logger.info("Processing interview %s", interview_id, extra={"interview_id": interview_id})

    with session_scope() as db:
        interview = _load(db, interview_id)
        if interview is None:
            logger.error("Interview %s no longer exists", interview_id)
            return
        if interview.status == InterviewStatus.PROCESSING:
            logger.warning("Interview %s is already being processed", interview_id)
        interview.status = InterviewStatus.PROCESSING
        interview.error_message = None

        role_title = interview.role.title
        resume_text = interview.candidate.resume_text or ""
        answer_inputs = [
            AnswerInput(
                question_id=answer.question_id,
                question_text=answer.question.text if answer.question else "",
                audio_path=answer.audio_path or None,
                transcript=answer.transcript or None,
            )
            for answer in interview.answers
        ]

    if not answer_inputs:
        _fail(interview_id, "The candidate did not submit any recordings")
        return

    try:
        analysis = analyze_interview(role_title, resume_text, answer_inputs)
    except AnalysisError as exc:
        _fail(interview_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - background job must not crash the worker
        logger.exception("Unexpected analysis failure for interview %s", interview_id)
        _fail(interview_id, f"Unexpected error during analysis: {exc}")
        return

    # ---------------------------------------------------------------- persist
    with session_scope() as db:
        interview = _load(db, interview_id)
        if interview is None:
            return

        by_question = {a.question_id: a for a in analysis.answers}
        for answer in interview.answers:
            analysed = by_question.get(answer.question_id)
            if not analysed:
                continue
            answer.transcript = analysed.transcript
            answer.scores = analysed.scores
            answer.skills = analysed.skills
            answer.feedback = analysed.feedback
            if analysed.duration_seconds:
                answer.duration_seconds = analysed.duration_seconds

        interview.dimension_scores = analysis.dimension_scores
        interview.skills_resume = analysis.skills_resume
        interview.skills_answer = analysis.skills_answer
        interview.relevance_score = analysis.relevance_score
        interview.overall_score = analysis.overall_score
        interview.verdict = analysis.verdict
        interview.summary = analysis.summary
        interview.strengths = analysis.strengths
        interview.improvements = analysis.improvements
        interview.analysis_meta = analysis.meta
        interview.status = InterviewStatus.COMPLETED
        interview.completed_at = datetime.now(UTC)

    # ----------------------------------------------------------------- notify
    with session_scope() as db:
        interview = _load(db, interview_id)
        if interview is None:
            return

        sent, error = try_send(lambda: send_candidate_report(interview), "Candidate report email")
        if sent:
            interview.candidate_email_sent_at = datetime.now(UTC)
        elif error:
            meta = dict(interview.analysis_meta or {})
            meta["candidate_email_error"] = error
            interview.analysis_meta = meta

        sent, error = try_send(lambda: send_recruiter_report(interview), "Recruiter report email")
        if sent:
            interview.recruiter_email_sent_at = datetime.now(UTC)
        elif error:
            meta = dict(interview.analysis_meta or {})
            meta["recruiter_email_error"] = error
            interview.analysis_meta = meta

    logger.info("Interview %s completed", interview_id, extra={"interview_id": interview_id})


def _fail(interview_id: int, message: str) -> None:
    logger.error("Interview %s failed: %s", interview_id, message)
    with session_scope() as db:
        interview = _load(db, interview_id)
        if interview is None:
            return
        interview.status = InterviewStatus.FAILED
        interview.error_message = message[:2000]
        interview.completed_at = datetime.now(UTC)
        try_send(lambda: send_failure_alert(interview, message), "Failure alert email")
