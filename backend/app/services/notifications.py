"""Turns Interview records into the emails candidates and recruiters receive."""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from ..config import settings
from ..models import Interview
from .analyzer import DIMENSION_LABELS, VERDICT_LABELS
from .mailer import Attachment, MailError, render, send_email
from .report_pdf import build_report_pdf

logger = logging.getLogger(__name__)

SCORE_BACKGROUNDS = {
    "strong_hire": "#047857",
    "shortlist": "#2563eb",
    "review": "#b45309",
    "not_recommended": "#4b5563",
}
SCORE_COLORS = {
    "strong_hire": "#047857",
    "shortlist": "#2563eb",
    "review": "#b45309",
    "not_recommended": "#4b5563",
}


def _fmt_datetime(value: datetime | None) -> str:
    if not value:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.strftime("%d %b %Y, %H:%M UTC")


def _dimension_pairs(interview: Interview) -> list[tuple[str, float]]:
    scores = interview.dimension_scores or {}
    return [(label, float(scores.get(key, 0.0))) for key, label in DIMENSION_LABELS.items() if key in scores]


def _safe_filename(name: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip()]
    return "".join(keep).strip("_") or "candidate"


def build_pdf_attachment(interview: Interview) -> Attachment | None:
    if not settings.ATTACH_PDF_REPORT:
        return None
    answers = [
        {
            "question": answer.question.text if answer.question else "",
            "transcript": answer.transcript,
            "scores": answer.scores or {},
        }
        for answer in interview.answers
    ]
    content = build_report_pdf(
        candidate_name=interview.candidate.full_name,
        candidate_email=interview.candidate.email,
        role_title=interview.role.title,
        company_name=settings.COMPANY_NAME,
        overall_score=float(interview.overall_score or 0),
        verdict_label=VERDICT_LABELS.get(interview.verdict or "", "Reviewed"),
        relevance_score=float(interview.relevance_score or 0),
        dimensions=_dimension_pairs(interview),
        summary=interview.summary or [],
        strengths=interview.strengths or [],
        improvements=interview.improvements or [],
        skills_resume=interview.skills_resume or [],
        skills_answer=interview.skills_answer or [],
        answers=answers,
        submitted_at=_fmt_datetime(interview.submitted_at),
        include_transcripts=settings.PDF_INCLUDE_TRANSCRIPTS,
    )
    if not content:
        return None
    filename = f"interview-report-{_safe_filename(interview.candidate.full_name)}.pdf"
    return Attachment(filename=filename, content=content, mimetype="application/pdf")


# ------------------------------------------------------------------- invites
def send_invite_email(interview: Interview, invite_url: str) -> bool:
    questions = interview.role.questions
    estimated = max(5, sum((q.time_limit_seconds or 180) for q in questions) // 60 + 3)
    html = render(
        "emails/invite.html",
        subject=f"Your interview for {interview.role.title}",
        company_name=settings.COMPANY_NAME,
        header_subtitle="Interview invitation",
        candidate_name=interview.candidate.full_name.split(" ")[0] or interview.candidate.full_name,
        role_title=interview.role.title,
        department=interview.role.department,
        question_count=len(questions),
        estimated_minutes=estimated,
        invite_url=invite_url,
        expires_at=_fmt_datetime(interview.expires_at),
    )
    return send_email(
        to=[interview.candidate.email],
        subject=f"Your interview for {interview.role.title} at {settings.COMPANY_NAME}",
        html=html,
    )


# ------------------------------------------------------------------- reports
def send_candidate_report(interview: Interview) -> bool:
    if not settings.SEND_SCORES_TO_CANDIDATE:
        logger.info("SEND_SCORES_TO_CANDIDATE=false — not emailing candidate %s", interview.candidate.email)
        return False

    attachment = build_pdf_attachment(interview)
    verdict = interview.verdict or "review"
    html = render(
        "emails/report_candidate.html",
        subject=f"Your interview results — {interview.role.title}",
        company_name=settings.COMPANY_NAME,
        header_subtitle="Your interview results",
        candidate_name=interview.candidate.full_name.split(" ")[0] or interview.candidate.full_name,
        role_title=interview.role.title,
        overall_score=interview.overall_score or 0,
        relevance_score=interview.relevance_score or 0,
        verdict_label=VERDICT_LABELS.get(verdict, "Reviewed"),
        score_bg=SCORE_BACKGROUNDS.get(verdict, "#2563eb"),
        dimensions=_dimension_pairs(interview),
        summary=interview.summary or [],
        strengths=interview.strengths or [],
        improvements=interview.improvements or [],
        skills_answer=(interview.skills_answer or [])[:18],
        attached_pdf=attachment is not None,
    )
    return send_email(
        to=[interview.candidate.email],
        subject=f"Your interview results — {interview.role.title}",
        html=html,
        attachments=[attachment] if attachment else None,
        reply_to=settings.RECRUITER_NOTIFY_EMAIL,
    )


def send_recruiter_report(interview: Interview) -> bool:
    recipient = settings.RECRUITER_NOTIFY_EMAIL
    if not recipient:
        return False

    meta = interview.analysis_meta or {}
    notes: list[str] = []
    if meta.get("degraded_answers"):
        notes.append(f"{meta['degraded_answers']} answer(s) fell back to heuristic scoring — the model was unreachable.")
    if meta.get("transcription_failures"):
        notes.append("Transcription issues: " + "; ".join(meta["transcription_failures"][:3]))

    attachment = build_pdf_attachment(interview)
    verdict = interview.verdict or "review"
    html = render(
        "emails/report_recruiter.html",
        subject=f"[{VERDICT_LABELS.get(verdict, 'Result')}] {interview.candidate.full_name} — {interview.role.title}",
        company_name=settings.COMPANY_NAME,
        header_subtitle="New interview result",
        candidate_name=interview.candidate.full_name,
        candidate_email=interview.candidate.email,
        role_title=interview.role.title,
        overall_score=interview.overall_score or 0,
        relevance_score=interview.relevance_score or 0,
        verdict_label=VERDICT_LABELS.get(verdict, "Reviewed"),
        score_color=SCORE_COLORS.get(verdict, "#2563eb"),
        dimensions=_dimension_pairs(interview),
        summary=interview.summary or [],
        skills_answer=(interview.skills_answer or [])[:18],
        submitted_at=_fmt_datetime(interview.submitted_at),
        report_url=f"{settings.PUBLIC_BASE_URL.rstrip('/')}/reports/{interview.id}",
        degraded_note=" ".join(notes) if notes else "",
    )
    return send_email(
        to=[recipient],
        subject=f"[{VERDICT_LABELS.get(verdict, 'Result')} {float(interview.overall_score or 0):.1f}/10] "
        f"{interview.candidate.full_name} — {interview.role.title}",
        html=html,
        attachments=[attachment] if attachment else None,
    )


def send_failure_alert(interview: Interview, error_message: str) -> bool:
    recipient = settings.RECRUITER_NOTIFY_EMAIL
    if not recipient:
        return False
    html = render(
        "emails/analysis_failed.html",
        subject=f"Interview analysis failed — {interview.candidate.full_name}",
        company_name=settings.COMPANY_NAME,
        header_subtitle="Action needed",
        candidate_name=interview.candidate.full_name,
        role_title=interview.role.title,
        error_message=error_message[:1500],
        report_url=f"{settings.PUBLIC_BASE_URL.rstrip('/')}/reports/{interview.id}",
    )
    return send_email(to=[recipient], subject=f"Interview analysis failed — {interview.candidate.full_name}", html=html)


def try_send(action, description: str) -> tuple[bool, str | None]:
    """Run a send_* function, converting mail failures into (False, reason)."""
    try:
        return bool(action()), None
    except MailError as exc:
        logger.error("%s failed: %s", description, exc)
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s raised unexpectedly", description)
        return False, str(exc)
