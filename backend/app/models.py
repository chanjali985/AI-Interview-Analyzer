"""SQLAlchemy ORM models."""
from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_uuid() -> str:
    return uuid.uuid4().hex


class InterviewStatus(enum.StrEnum):
    INVITED = "invited"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class User(Base):
    """A recruiter / hiring manager who can create roles and invite candidates."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    roles: Mapped[list[Role]] = relationship(back_populates="owner")


class Role(Base):
    """A job opening with its own question set."""

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    department: Mapped[str] = mapped_column(String(120), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    owner: Mapped[User | None] = relationship(back_populates="roles")
    questions: Mapped[list[Question]] = relationship(
        back_populates="role",
        cascade="all, delete-orphan",
        order_by="Question.order_index",
    )
    interviews: Mapped[list[Interview]] = relationship(back_populates="role")


class Question(Base):
    """One interview question belonging to a role."""

    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(80), default="general")
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    time_limit_seconds: Mapped[int] = mapped_column(Integer, default=180)

    role: Mapped[Role] = relationship(back_populates="questions")
    answers: Mapped[list[Answer]] = relationship(back_populates="question", cascade="all, delete-orphan")


class Candidate(Base):
    """A person interviewing for one or more roles."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    phone: Mapped[str] = mapped_column(String(40), default="")
    resume_text: Mapped[str] = mapped_column(Text, default="")
    resume_filename: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    interviews: Mapped[list[Interview]] = relationship(back_populates="candidate", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("email", name="uq_candidate_email"),)


class Interview(Base):
    """One candidate's attempt at one role's question set."""

    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True, default=new_uuid)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id", ondelete="CASCADE"), index=True)

    status: Mapped[InterviewStatus] = mapped_column(
        SAEnum(InterviewStatus, values_callable=lambda e: [m.value for m in e]),
        default=InterviewStatus.INVITED,
        index=True,
    )

    invited_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    overall_score: Mapped[float | None] = mapped_column(Float)
    relevance_score: Mapped[float | None] = mapped_column(Float)
    verdict: Mapped[str | None] = mapped_column(String(40))
    dimension_scores: Mapped[dict | None] = mapped_column(JSON)
    skills_resume: Mapped[list | None] = mapped_column(JSON)
    skills_answer: Mapped[list | None] = mapped_column(JSON)
    summary: Mapped[list | None] = mapped_column(JSON)
    strengths: Mapped[list | None] = mapped_column(JSON)
    improvements: Mapped[list | None] = mapped_column(JSON)

    error_message: Mapped[str | None] = mapped_column(Text)
    candidate_email_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    recruiter_email_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    analysis_meta: Mapped[dict | None] = mapped_column(JSON)

    candidate: Mapped[Candidate] = relationship(back_populates="interviews")
    role: Mapped[Role] = relationship(back_populates="interviews")
    answers: Mapped[list[Answer]] = relationship(
        back_populates="interview", cascade="all, delete-orphan", order_by="Answer.id"
    )


class Answer(Base):
    """A recorded answer to a single question."""

    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id", ondelete="CASCADE"), index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"), index=True)

    audio_path: Mapped[str] = mapped_column(String(512), default="")
    audio_filename: Mapped[str] = mapped_column(String(255), default="")
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    transcript: Mapped[str] = mapped_column(Text, default="")
    scores: Mapped[dict | None] = mapped_column(JSON)
    skills: Mapped[list | None] = mapped_column(JSON)
    feedback: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    interview: Mapped[Interview] = relationship(back_populates="answers")
    question: Mapped[Question] = relationship(back_populates="answers")

    __table_args__ = (UniqueConstraint("interview_id", "question_id", name="uq_answer_per_question"),)
