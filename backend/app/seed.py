"""First-run bootstrap: an admin account and (optionally) a starter role."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from .config import settings
from .models import Question, Role, User
from .security import hash_password

logger = logging.getLogger(__name__)

STARTER_QUESTIONS = [
    ("Tell us about your background and the work you are most proud of.", "background", 180),
    ("Walk us through a technical problem you solved recently. What made it hard?", "technical", 240),
    ("Describe a time you disagreed with a teammate. How did you handle it?", "behavioural", 180),
    ("Which tools and technologies do you reach for most, and why?", "technical", 180),
]


def ensure_admin(db: Session) -> User:
    email = settings.ADMIN_EMAIL.lower()
    user = db.query(User).filter(User.email == email).one_or_none()
    if user is not None:
        return user

    user = User(
        email=email,
        full_name=settings.ADMIN_NAME,
        hashed_password=hash_password(settings.ADMIN_PASSWORD),
        is_active=True,
        is_superuser=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.warning(
        "Created the bootstrap admin account %s. Change ADMIN_PASSWORD before going live.", email
    )
    return user


def ensure_starter_role(db: Session, owner: User) -> None:
    if not settings.SEED_STARTER_ROLE:
        return
    if db.query(Role).count():
        return

    role = Role(
        title="Software Engineer",
        department="Engineering",
        description=(
            "Starter question set created automatically on first run. "
            "Edit it or create your own role from the dashboard."
        ),
        owner_id=owner.id,
    )
    for index, (text, category, limit) in enumerate(STARTER_QUESTIONS):
        role.questions.append(
            Question(text=text, category=category, time_limit_seconds=limit, order_index=index)
        )
    db.add(role)
    db.commit()
    logger.info("Seeded starter role '%s' with %s questions", role.title, len(STARTER_QUESTIONS))


def run(db: Session) -> None:
    admin = ensure_admin(db)
    ensure_starter_role(db, admin)
