"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .models import InterviewStatus


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------- auth
class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(ORMModel):
    id: int
    email: EmailStr
    full_name: str
    is_superuser: bool
    created_at: datetime


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = ""
    password: str = Field(min_length=8, max_length=128)


# --------------------------------------------------------------------- roles
class QuestionCreate(BaseModel):
    text: str = Field(min_length=5, max_length=2000)
    category: str = "general"
    time_limit_seconds: int = Field(default=180, ge=30, le=900)


class QuestionOut(ORMModel):
    id: int
    text: str
    category: str
    order_index: int
    time_limit_seconds: int


class RoleCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    department: str = ""
    description: str = ""
    questions: list[QuestionCreate] = Field(default_factory=list, max_length=25)


class RoleUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=2, max_length=200)
    department: str | None = None
    description: str | None = None
    is_active: bool | None = None
    questions: list[QuestionCreate] | None = Field(default=None, max_length=25)


class RoleOut(ORMModel):
    id: int
    title: str
    department: str
    description: str
    is_active: bool
    created_at: datetime
    questions: list[QuestionOut] = []


class RoleSummary(ORMModel):
    id: int
    title: str
    department: str
    is_active: bool
    question_count: int = 0
    interview_count: int = 0


# ---------------------------------------------------------------- candidates
class CandidateOut(ORMModel):
    id: int
    full_name: str
    email: EmailStr
    phone: str
    resume_filename: str
    created_at: datetime


class InviteRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=200)
    email: EmailStr
    phone: str = ""
    role_id: int
    resume_text: str = ""
    send_email: bool = True
    expires_in_hours: int | None = Field(default=None, ge=1, le=24 * 60)


class InviteResponse(BaseModel):
    interview_id: int
    public_id: str
    invite_url: str
    expires_at: datetime | None
    email_sent: bool
    email_error: str | None = None


# ---------------------------------------------------------------- interviews
class AnswerOut(ORMModel):
    id: int
    question_id: int
    transcript: str
    scores: dict[str, float] | None = None
    skills: list[str] | None = None
    feedback: str | None = None
    duration_seconds: float | None = None


class InterviewSummary(ORMModel):
    id: int
    public_id: str
    status: InterviewStatus
    overall_score: float | None
    relevance_score: float | None
    verdict: str | None
    invited_at: datetime
    submitted_at: datetime | None
    completed_at: datetime | None
    candidate: CandidateOut
    role: RoleSummary


class InterviewReport(ORMModel):
    id: int
    public_id: str
    status: InterviewStatus
    overall_score: float | None
    relevance_score: float | None
    verdict: str | None
    dimension_scores: dict[str, float] | None = None
    skills_resume: list[str] | None = None
    skills_answer: list[str] | None = None
    summary: list[str] | None = None
    strengths: list[str] | None = None
    improvements: list[str] | None = None
    error_message: str | None = None
    invited_at: datetime
    submitted_at: datetime | None
    completed_at: datetime | None
    candidate_email_sent_at: datetime | None = None
    candidate: CandidateOut
    role: RoleOut
    answers: list[AnswerOut] = []


# ------------------------------------------------- candidate-facing schemas
class PublicQuestion(BaseModel):
    id: int
    text: str
    order_index: int
    time_limit_seconds: int


class InterviewSession(BaseModel):
    public_id: str
    status: InterviewStatus
    candidate_name: str
    candidate_email: EmailStr
    role_title: str
    role_description: str
    company_name: str
    questions: list[PublicQuestion]
    resume_on_file: bool
    expires_at: datetime | None


class SubmitResponse(BaseModel):
    public_id: str
    status: InterviewStatus
    message: str


class InterviewStatusOut(BaseModel):
    public_id: str
    status: InterviewStatus
    message: str
    overall_score: float | None = None
    email_sent: bool = False


# ----------------------------------------------------------------- dashboard
class DashboardStats(BaseModel):
    total_roles: int
    active_roles: int
    total_candidates: int
    total_interviews: int
    completed_interviews: int
    pending_interviews: int
    failed_interviews: int
    average_score: float | None
    shortlisted: int
    recent: list[InterviewSummary] = []


class HealthOut(BaseModel):
    status: str
    version: str
    environment: str
    database: str
    llm_provider: str
    llm_ready: bool
    transcription_provider: str
    transcription_ready: bool
    mail_configured: bool
