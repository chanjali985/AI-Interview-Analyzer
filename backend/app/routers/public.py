"""Candidate-facing endpoints. Auth is the signed invite token, nothing else."""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import get_interview_by_invite
from ..models import Answer, Interview, InterviewStatus
from ..ratelimit import limit_submissions
from ..schemas import InterviewSession, InterviewStatusOut, PublicQuestion, SubmitResponse
from ..services import storage
from ..services.resume import ResumeParseError, extract_text
from ..worker import enqueue

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/public", tags=["candidate"])

STATUS_MESSAGES = {
    InterviewStatus.INVITED: "Ready to start.",
    InterviewStatus.IN_PROGRESS: "Interview in progress.",
    InterviewStatus.SUBMITTED: "Submitted. Your answers are queued for analysis.",
    InterviewStatus.PROCESSING: "We are analysing your answers right now.",
    InterviewStatus.COMPLETED: "Analysis complete. Your report has been emailed to you.",
    InterviewStatus.FAILED: "We could not analyse your answers. Our team has been notified.",
    InterviewStatus.EXPIRED: "This interview link has expired.",
}


def _ensure_open(interview: Interview, db: Session) -> None:
    """Reject work on interviews that are expired or already submitted."""
    if interview.expires_at:
        expires = interview.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if expires < datetime.now(UTC) and interview.status in (
            InterviewStatus.INVITED,
            InterviewStatus.IN_PROGRESS,
        ):
            interview.status = InterviewStatus.EXPIRED
            db.commit()
            raise HTTPException(status_code=status.HTTP_410_GONE, detail="This interview link has expired")

    if interview.status in (
        InterviewStatus.SUBMITTED,
        InterviewStatus.PROCESSING,
        InterviewStatus.COMPLETED,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This interview has already been submitted",
        )
    if interview.status == InterviewStatus.EXPIRED:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="This interview link has expired")


@router.get("/session", response_model=InterviewSession)
def get_session(interview: Interview = Depends(get_interview_by_invite)) -> InterviewSession:
    return InterviewSession(
        public_id=interview.public_id,
        status=interview.status,
        candidate_name=interview.candidate.full_name,
        candidate_email=interview.candidate.email,
        role_title=interview.role.title,
        role_description=interview.role.description,
        company_name=settings.COMPANY_NAME,
        questions=[
            PublicQuestion(
                id=question.id,
                text=question.text,
                order_index=question.order_index,
                time_limit_seconds=question.time_limit_seconds,
            )
            for question in interview.role.questions
        ],
        resume_on_file=bool((interview.candidate.resume_text or "").strip()),
        expires_at=interview.expires_at,
    )


@router.post("/session/start", response_model=InterviewStatusOut)
def start_session(
    interview: Interview = Depends(get_interview_by_invite),
    db: Session = Depends(get_db),
) -> InterviewStatusOut:
    _ensure_open(interview, db)
    if interview.status == InterviewStatus.INVITED:
        interview.status = InterviewStatus.IN_PROGRESS
        interview.started_at = datetime.now(UTC)
        db.commit()
        db.refresh(interview)
    return InterviewStatusOut(
        public_id=interview.public_id,
        status=interview.status,
        message=STATUS_MESSAGES.get(interview.status, ""),
    )


@router.post("/resume", response_model=InterviewStatusOut)
def upload_resume(
    resume_file: UploadFile | None = File(default=None),
    resume_text: str = Form(default=""),
    interview: Interview = Depends(get_interview_by_invite),
    db: Session = Depends(get_db),
    _: None = Depends(limit_submissions),
) -> InterviewStatusOut:
    _ensure_open(interview, db)

    if resume_file is not None and resume_file.filename:
        try:
            path = storage.save_resume(resume_file, interview.public_id)
        except storage.StorageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        try:
            text = extract_text(path)
        except ResumeParseError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        interview.candidate.resume_text = text
        interview.candidate.resume_filename = os.path.basename(path)
    elif resume_text.strip():
        if len(resume_text.strip()) < 40:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Please paste a little more of your resume so the skill match is meaningful",
            )
        interview.candidate.resume_text = resume_text.strip()[:20000]
        interview.candidate.resume_filename = "pasted.txt"
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload a resume file or paste the text")

    db.commit()
    return InterviewStatusOut(
        public_id=interview.public_id,
        status=interview.status,
        message="Resume saved.",
    )


@router.post("/answers/{question_id}", response_model=InterviewStatusOut)
def upload_answer(
    question_id: int,
    audio: UploadFile = File(...),
    duration_seconds: float | None = Form(default=None),
    interview: Interview = Depends(get_interview_by_invite),
    db: Session = Depends(get_db),
    _: None = Depends(limit_submissions),
) -> InterviewStatusOut:
    _ensure_open(interview, db)

    question = next((q for q in interview.role.questions if q.id == question_id), None)
    if question is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found for this interview")

    try:
        path = storage.save_audio(audio, interview.public_id, question_id)
    except storage.StorageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    answer = (
        db.query(Answer)
        .filter(Answer.interview_id == interview.id, Answer.question_id == question_id)
        .one_or_none()
    )
    if answer is None:
        answer = Answer(interview_id=interview.id, question_id=question_id)
        db.add(answer)
    else:
        # Candidate re-recorded: drop the previous take.
        if answer.audio_path and answer.audio_path != path and os.path.exists(answer.audio_path):
            try:
                os.unlink(answer.audio_path)
            except OSError:
                pass

    answer.audio_path = path
    answer.audio_filename = os.path.basename(path)
    answer.duration_seconds = duration_seconds
    answer.transcript = ""
    answer.scores = None

    if interview.status == InterviewStatus.INVITED:
        interview.status = InterviewStatus.IN_PROGRESS
        interview.started_at = interview.started_at or datetime.now(UTC)

    db.commit()
    return InterviewStatusOut(
        public_id=interview.public_id,
        status=interview.status,
        message=f"Answer saved for question {question.order_index + 1}.",
    )


@router.post("/submit", response_model=SubmitResponse)
def submit_interview(
    interview: Interview = Depends(get_interview_by_invite),
    db: Session = Depends(get_db),
    _: None = Depends(limit_submissions),
) -> SubmitResponse:
    _ensure_open(interview, db)

    answered = {answer.question_id for answer in interview.answers if answer.audio_path}
    required = {question.id for question in interview.role.questions}
    missing = required - answered
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{len(missing)} question(s) still need a recording",
        )
    if not (interview.candidate.resume_text or "").strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Please add your resume before submitting",
        )

    interview.status = InterviewStatus.SUBMITTED
    interview.submitted_at = datetime.now(UTC)
    db.commit()

    enqueue(interview.id)
    logger.info("Interview %s submitted by %s", interview.id, interview.candidate.email)

    return SubmitResponse(
        public_id=interview.public_id,
        status=InterviewStatus.SUBMITTED,
        message="Thanks! Your interview is being analysed and your report will arrive by email shortly.",
    )


@router.get("/status", response_model=InterviewStatusOut)
def interview_status(interview: Interview = Depends(get_interview_by_invite)) -> InterviewStatusOut:
    return InterviewStatusOut(
        public_id=interview.public_id,
        status=interview.status,
        message=STATUS_MESSAGES.get(interview.status, ""),
        overall_score=interview.overall_score if interview.status == InterviewStatus.COMPLETED else None,
        email_sent=interview.candidate_email_sent_at is not None,
    )
