"""Password hashing and JWT helpers (recruiter access tokens + candidate invites)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from .config import settings

ALGORITHM = "HS256"
BCRYPT_MAX_BYTES = 72


class TokenError(Exception):
    """Raised when a token is missing, expired, or malformed."""


def hash_password(password: str) -> str:
    payload = password.encode("utf-8")[:BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(payload, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:BCRYPT_MAX_BYTES], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _create_token(subject: str, token_type: str, expires: timedelta, extra: dict[str, Any] | None = None) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires).timestamp()),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.PyJWTError as exc:
        raise TokenError("Token is invalid") from exc

    if expected_type and payload.get("type") != expected_type:
        raise TokenError("Token has the wrong type")
    return payload


def create_access_token(user_id: int, email: str) -> str:
    return _create_token(
        subject=str(user_id),
        token_type="access",
        expires=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        extra={"email": email},
    )


def create_invite_token(interview_public_id: str, hours: int | None = None) -> str:
    return _create_token(
        subject=interview_public_id,
        token_type="invite",
        expires=timedelta(hours=hours if hours is not None else settings.INVITE_TOKEN_EXPIRE_HOURS),
    )


def read_invite_token(token: str) -> str:
    """Return the interview public id encoded in an invite token."""
    payload = decode_token(token, expected_type="invite")
    subject = payload.get("sub")
    if not subject:
        raise TokenError("Invite token has no subject")
    return str(subject)
