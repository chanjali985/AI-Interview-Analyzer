"""Liveness, readiness and email diagnostics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..schemas import HealthOut
from ..services import mailer
from ..services.llm import get_llm
from ..services.transcription import transcription_ready
from ..worker import pending_count

router = APIRouter(prefix="/health", tags=["health"])

VERSION = "2.0.0"


@router.get("", response_model=HealthOut)
@router.get("/", response_model=HealthOut, include_in_schema=False)
def health(db: Session = Depends(get_db)) -> HealthOut:
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:  # noqa: BLE001
        database = f"error: {exc}"

    llm = get_llm()
    return HealthOut(
        status="ok" if database == "ok" else "degraded",
        version=VERSION,
        environment=settings.ENVIRONMENT,
        database=database,
        llm_provider=llm.name,
        llm_ready=llm.health(),
        transcription_provider=settings.TRANSCRIPTION_PROVIDER,
        transcription_ready=transcription_ready(),
        mail_configured=settings.mail_configured,
    )


@router.get("/live", include_in_schema=False)
def live() -> dict:
    """Kubernetes-style liveness probe: cheap and dependency-free."""
    return {"status": "alive"}


@router.get("/ready")
def ready(db: Session = Depends(get_db)) -> dict:
    checks = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:  # noqa: BLE001
        checks["database"] = False
    checks["llm"] = get_llm().health()
    checks["transcription"] = transcription_ready()

    ok = checks["database"] and checks["llm"] and checks["transcription"]
    payload = {"ready": ok, "checks": checks, "queue_depth": pending_count()}
    if not ok:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=payload)
    return payload


@router.get("/email")
def email_status(_: User = Depends(get_current_user)) -> dict:
    result = mailer.verify_connection()
    return {
        "enabled": settings.MAIL_ENABLED,
        "host": settings.SMTP_HOST,
        "port": settings.SMTP_PORT,
        "username": settings.SMTP_USERNAME,
        "from": settings.MAIL_FROM,
        "recruiter_notify": settings.RECRUITER_NOTIFY_EMAIL,
        **result,
    }


@router.post("/email/test")
def send_test_email(user: User = Depends(get_current_user)) -> dict:
    html = mailer.render(
        "emails/base.html",
        subject="Test email",
        company_name=settings.COMPANY_NAME,
        header_subtitle="SMTP test",
    )
    body = html.replace(
        "{% block content %}{% endblock %}", ""
    )  # base has no content block filled; send a plain note instead
    try:
        mailer.send_email(
            to=[user.email],
            subject=f"{settings.APP_NAME}: SMTP test",
            html=body or "<p>SMTP is configured correctly.</p>",
            text="SMTP is configured correctly.",
        )
    except mailer.MailError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {"sent": True, "to": user.email}
