"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, joinedload

from .database import get_db
from .models import Answer, Interview, InterviewStatus, User
from .security import TokenError, decode_token, read_invite_token

bearer_scheme = HTTPBearer(auto_error=False)

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise CREDENTIALS_ERROR
    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except TokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    user = db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise CREDENTIALS_ERROR
    return user


def get_current_superuser(user: User = Depends(get_current_user)) -> User:
    if not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")
    return user


def get_interview_by_invite(
    x_interview_token: str | None = Header(default=None, alias="X-Interview-Token"),
    db: Session = Depends(get_db),
) -> Interview:
    """Resolve the candidate's interview from their signed invite token."""
    if not x_interview_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing interview token")
    try:
        public_id = read_invite_token(x_interview_token)
    except TokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    interview = (
        db.query(Interview)
        .options(
            joinedload(Interview.candidate),
            joinedload(Interview.role),
            joinedload(Interview.answers).joinedload(Answer.question),
        )
        .filter(Interview.public_id == public_id)
        .one_or_none()
    )
    if interview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")
    if interview.status == InterviewStatus.EXPIRED:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="This interview link has expired")
    return interview
