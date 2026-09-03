"""Recruiter-facing interview management: invite, list, report, re-run, resend."""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import Answer, Candidate, Interview, InterviewStatus, Role, User
from ..schemas import (
    DashboardStats,
    InterviewReport,
    InterviewSummary,
    InviteRequest,
    InviteResponse,
    RoleSummary,
)
from ..security import create_invite_token
from ..services import storage
from ..services.notifications import send_candidate_report, send_invite_email, try_send
from ..worker import enqueue

logger = logging.getLogger(__name__)
router = APIRouter(tags=["interviews"])


def _invite_url(token: str) -> str:
    return f"{settings.PUBLIC_BASE_URL.rstrip('/')}/interview/{token}"


def _load_interview(db: Session, interview_id: int) -> Interview:
    interview = (
        db.query(Interview)
        .options(
            joinedload(Interview.candidate),
            joinedload(Interview.role).joinedload(Role.questions),
            joinedload(Interview.answers).joinedload(Answer.question),
        )
        .filter(Interview.id == interview_id)
        .one_or_none()
    )
    if interview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")
    return interview


def _to_summary(interview: Interview, question_count: int = 0) -> InterviewSummary:
    return InterviewSummary(
        id=interview.id,
        public_id=interview.public_id,
        status=interview.status,
        overall_score=interview.overall_score,
        relevance_score=interview.relevance_score,
        verdict=interview.verdict,
        invited_at=interview.invited_at,
        submitted_at=interview.submitted_at,
        completed_at=interview.completed_at,
        candidate=interview.candidate,
        role=RoleSummary(
            id=interview.role.id,
            title=interview.role.title,
            department=interview.role.department,
            is_active=interview.role.is_active,
            question_count=question_count or len(interview.role.questions),
        ),
    )


# ------------------------------------------------------------------- invites
@router.post("/interviews/invite", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
def invite_candidate(
    payload: InviteRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> InviteResponse:
    role = db.query(Role).options(joinedload(Role.questions)).filter(Role.id == payload.role_id).one_or_none()
    if role is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    if not role.is_active:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This role is no longer active")
    if not role.questions:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This role has no questions yet")

    email = payload.email.lower()
    candidate = db.query(Candidate).filter(Candidate.email == email).one_or_none()
    if candidate is None:
        candidate = Candidate(full_name=payload.full_name.strip(), email=email, phone=payload.phone.strip())
        db.add(candidate)
        db.flush()
    else:
        candidate.full_name = payload.full_name.strip() or candidate.full_name
        if payload.phone:
            candidate.phone = payload.phone.strip()
    if payload.resume_text.strip():
        candidate.resume_text = payload.resume_text.strip()

    open_interview = (
        db.query(Interview)
        .filter(
            Interview.candidate_id == candidate.id,
            Interview.role_id == role.id,
            Interview.status.in_([InterviewStatus.INVITED, InterviewStatus.IN_PROGRESS]),
        )
        .first()
    )
    if open_interview is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This candidate already has an open invitation for this role",
        )

    hours = payload.expires_in_hours or settings.INVITE_TOKEN_EXPIRE_HOURS
    interview = Interview(
        candidate_id=candidate.id,
        role_id=role.id,
        status=InterviewStatus.INVITED,
        expires_at=datetime.now(UTC) + timedelta(hours=hours),
    )
    db.add(interview)
    db.commit()
    db.refresh(interview)

    token = create_invite_token(interview.public_id, hours=hours)
    url = _invite_url(token)

    email_sent = False
    email_error: str | None = None
    if payload.send_email:
        interview = _load_interview(db, interview.id)
        email_sent, email_error = try_send(lambda: send_invite_email(interview, url), "Invite email")
        if not email_sent and email_error is None:
            email_error = "Email delivery is disabled (MAIL_ENABLED=false)"

    logger.info("Invited %s to role %s (interview %s)", email, role.title, interview.id)
    return InviteResponse(
        interview_id=interview.id,
        public_id=interview.public_id,
        invite_url=url,
        expires_at=interview.expires_at,
        email_sent=email_sent,
        email_error=email_error,
    )


@router.post("/interviews/{interview_id}/invite-link", response_model=InviteResponse)
def regenerate_invite_link(
    interview_id: int,
    send_email: bool = Query(default=False),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> InviteResponse:
    interview = _load_interview(db, interview_id)
    if interview.status in (InterviewStatus.COMPLETED, InterviewStatus.PROCESSING, InterviewStatus.SUBMITTED):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This interview has already been submitted")

    hours = settings.INVITE_TOKEN_EXPIRE_HOURS
    interview.expires_at = datetime.now(UTC) + timedelta(hours=hours)
    if interview.status == InterviewStatus.EXPIRED:
        interview.status = InterviewStatus.INVITED
    db.commit()
    db.refresh(interview)

    token = create_invite_token(interview.public_id, hours=hours)
    url = _invite_url(token)
    sent, error = (False, None)
    if send_email:
        sent, error = try_send(lambda: send_invite_email(interview, url), "Invite email")

    return InviteResponse(
        interview_id=interview.id,
        public_id=interview.public_id,
        invite_url=url,
        expires_at=interview.expires_at,
        email_sent=sent,
        email_error=error,
    )


# --------------------------------------------------------------------- lists
@router.get("/interviews", response_model=list[InterviewSummary])
def list_interviews(
    status_filter: InterviewStatus | None = Query(default=None, alias="status"),
    role_id: int | None = None,
    search: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[InterviewSummary]:
    query = (
        db.query(Interview)
        .options(
            joinedload(Interview.candidate),
            joinedload(Interview.role).joinedload(Role.questions),
        )
        .join(Candidate, Interview.candidate_id == Candidate.id)
    )
    if status_filter:
        query = query.filter(Interview.status == status_filter)
    if role_id:
        query = query.filter(Interview.role_id == role_id)
    if search:
        pattern = f"%{search.strip().lower()}%"
        query = query.filter(
            or_(func.lower(Candidate.full_name).like(pattern), func.lower(Candidate.email).like(pattern))
        )

    interviews = query.order_by(Interview.invited_at.desc()).offset(offset).limit(limit).all()
    return [_to_summary(interview) for interview in interviews]


@router.get("/interviews/{interview_id}", response_model=InterviewReport)
def get_interview(
    interview_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> Interview:
    return _load_interview(db, interview_id)


@router.get("/interviews/{interview_id}/answers/{answer_id}/audio")
def get_answer_audio(
    interview_id: int,
    answer_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> FileResponse:
    answer = (
        db.query(Answer)
        .filter(Answer.id == answer_id, Answer.interview_id == interview_id)
        .one_or_none()
    )
    if answer is None or not answer.audio_path or not os.path.exists(answer.audio_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not available")
    return FileResponse(answer.audio_path, filename=os.path.basename(answer.audio_path))


@router.get("/interviews/{interview_id}/report.pdf")
def download_report_pdf(
    interview_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    from fastapi.responses import Response

    from ..services.notifications import build_pdf_attachment

    interview = _load_interview(db, interview_id)
    if interview.status != InterviewStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This interview has not been analysed yet")

    attachment = build_pdf_attachment(interview)
    if attachment is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="PDF generation is not available")
    return Response(
        content=attachment.content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{attachment.filename}"'},
    )


# ------------------------------------------------------------------- actions
@router.post("/interviews/{interview_id}/reanalyze", status_code=status.HTTP_202_ACCEPTED)
def reanalyze(
    interview_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    interview = _load_interview(db, interview_id)
    if interview.status in (InterviewStatus.INVITED, InterviewStatus.IN_PROGRESS):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The candidate has not submitted yet")
    if not interview.answers:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="There are no recordings to analyse")

    interview.status = InterviewStatus.SUBMITTED
    interview.error_message = None
    db.commit()
    enqueue(interview.id)
    return {"status": "queued", "interview_id": interview.id}


@router.post("/interviews/{interview_id}/resend-report")
def resend_report(
    interview_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    interview = _load_interview(db, interview_id)
    if interview.status != InterviewStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This interview has no finished report")

    sent, error = try_send(lambda: send_candidate_report(interview), "Candidate report email")
    if sent:
        interview.candidate_email_sent_at = datetime.now(UTC)
        db.commit()
        return {"sent": True, "to": interview.candidate.email}
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=error or "Email is disabled (MAIL_ENABLED=false)",
    )


@router.delete("/interviews/{interview_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_interview(
    interview_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> None:
    interview = _load_interview(db, interview_id)
    public_id = interview.public_id
    db.delete(interview)
    db.commit()
    storage.delete_interview_files(public_id)


# ----------------------------------------------------------------- dashboard
@router.get("/dashboard", response_model=DashboardStats)
def dashboard(db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> DashboardStats:
    total_roles = db.query(func.count(Role.id)).scalar() or 0
    active_roles = db.query(func.count(Role.id)).filter(Role.is_active.is_(True)).scalar() or 0
    total_candidates = db.query(func.count(Candidate.id)).scalar() or 0
    total_interviews = db.query(func.count(Interview.id)).scalar() or 0
    completed = db.query(func.count(Interview.id)).filter(Interview.status == InterviewStatus.COMPLETED).scalar() or 0
    failed = db.query(func.count(Interview.id)).filter(Interview.status == InterviewStatus.FAILED).scalar() or 0
    pending = (
        db.query(func.count(Interview.id))
        .filter(
            Interview.status.in_(
                [
                    InterviewStatus.INVITED,
                    InterviewStatus.IN_PROGRESS,
                    InterviewStatus.SUBMITTED,
                    InterviewStatus.PROCESSING,
                ]
            )
        )
        .scalar()
        or 0
    )
    average = db.query(func.avg(Interview.overall_score)).filter(Interview.overall_score.isnot(None)).scalar()
    shortlisted = (
        db.query(func.count(Interview.id))
        .filter(Interview.overall_score >= settings.SHORTLIST_THRESHOLD)
        .scalar()
        or 0
    )

    recent = (
        db.query(Interview)
        .options(
            joinedload(Interview.candidate),
            joinedload(Interview.role).joinedload(Role.questions),
        )
        .order_by(Interview.invited_at.desc())
        .limit(8)
        .all()
    )

    return DashboardStats(
        total_roles=total_roles,
        active_roles=active_roles,
        total_candidates=total_candidates,
        total_interviews=total_interviews,
        completed_interviews=completed,
        pending_interviews=pending,
        failed_interviews=failed,
        average_score=round(float(average), 2) if average is not None else None,
        shortlisted=shortlisted,
        recent=[_to_summary(interview) for interview in recent],
    )
