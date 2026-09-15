"""Application configuration, loaded from environment variables / .env."""
from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the AI Interview Analyzer."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    APP_NAME: str = "AI Interview Analyzer"
    # Shown to candidates in the UI and in every email they receive.
    COMPANY_NAME: str = "AI Interview Analyzer"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = "/api"

    # Public base URL of the deployed app. Used to build candidate invite links.
    PUBLIC_BASE_URL: str = "http://localhost:8000"

    # ----------------------------------------------------------------- auth
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12
    INVITE_TOKEN_EXPIRE_HOURS: int = 24 * 7

    # Bootstrap recruiter account, created on first startup if absent.
    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = "change-me-now"
    ADMIN_NAME: str = "Hiring Admin"
    # Create one starter role with sample questions when the database is empty.
    SEED_STARTER_ROLE: bool = True

    # ------------------------------------------------------------- database
    DATABASE_URL: str = "sqlite:///./data/interviews.db"

    # ------------------------------------------------------------------ cors
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:8000"]

    # ------------------------------------------------------------------- llm
    # "ollama" keeps everything local and free. "openai" / "groq" / "gemini"
    # talk to a hosted API and need the matching key below.
    LLM_PROVIDER: Literal["ollama", "openai", "groq", "gemini", "none"] = "ollama"
    LLM_MODEL: str = "llama3.2"
    LLM_TIMEOUT_SECONDS: int = 120
    LLM_MAX_RETRIES: int = 2

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    GROQ_API_KEY: str | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GEMINI_API_KEY: str | None = None
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # --------------------------------------------------------- transcription
    # "faster_whisper" runs locally (default). "openai" uses the Whisper API.
    # "google" uses the free SpeechRecognition endpoint (best-effort only).
    TRANSCRIPTION_PROVIDER: Literal["faster_whisper", "openai", "google", "none"] = "faster_whisper"
    WHISPER_MODEL: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_LANGUAGE: str | None = "en"

    # ------------------------------------------------------------ embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDINGS_ENABLED: bool = True

    # ---------------------------------------------------------------- uploads
    STORAGE_DIR: str = "./data/uploads"
    MAX_AUDIO_MB: int = 25
    MAX_RESUME_MB: int = 10
    ALLOWED_AUDIO_EXTENSIONS: list[str] = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"]
    ALLOWED_RESUME_EXTENSIONS: list[str] = [".pdf", ".txt", ".md", ".docx"]
    KEEP_AUDIO_FILES: bool = True

    # ------------------------------------------------------------------ mail
    MAIL_ENABLED: bool = True
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = "your-address@gmail.com"
    SMTP_PASSWORD: str | None = None  # Gmail App Password (16 chars, no spaces)
    SMTP_STARTTLS: bool = True
    SMTP_SSL: bool = False
    MAIL_FROM: str = "your-address@gmail.com"
    MAIL_FROM_NAME: str = "AI Interview Analyzer"
    MAIL_REPLY_TO: str | None = None
    # Recruiter address that receives a copy of every finished report.
    RECRUITER_NOTIFY_EMAIL: str | None = None
    SEND_SCORES_TO_CANDIDATE: bool = True
    ATTACH_PDF_REPORT: bool = True

    # ----------------------------------------------------------- rate limits
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_SUBMISSIONS_PER_HOUR: int = 20

    # ------------------------------------------------------------- analysis
    ANALYSIS_WORKERS: int = 2
    # Weights used to fold the per-dimension scores into one overall score.
    WEIGHT_COMMUNICATION: float = 0.2
    WEIGHT_TECHNICAL: float = 0.3
    WEIGHT_CONFIDENCE: float = 0.15
    WEIGHT_CLARITY: float = 0.15
    WEIGHT_OVERALL: float = 0.2
    # How much resume/answer skill relevance nudges the final score (0-1).
    RELEVANCE_WEIGHT: float = 0.15
    SHORTLIST_THRESHOLD: float = 7.0
    # Include full transcripts in the PDF the candidate receives.
    PDF_INCLUDE_TRANSCRIPTS: bool = True

    @field_validator("CORS_ORIGINS", "ALLOWED_AUDIO_EXTENSIONS", "ALLOWED_RESUME_EXTENSIONS", mode="before")
    @classmethod
    def _split_csv(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def mail_configured(self) -> bool:
        return bool(self.MAIL_ENABLED and self.SMTP_USERNAME and self.SMTP_PASSWORD)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
