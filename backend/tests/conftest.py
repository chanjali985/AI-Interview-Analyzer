"""Test fixtures. Everything external (LLM, Whisper, SMTP) is stubbed."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

_TMP = tempfile.mkdtemp(prefix="interview-tests-")

os.environ.update(
    {
        "ENVIRONMENT": "development",
        "DATABASE_URL": f"sqlite:///{_TMP}/test.db",
        "STORAGE_DIR": f"{_TMP}/uploads",
        "SECRET_KEY": "test-secret-key-not-for-production",
        "MAIL_ENABLED": "false",
        "LLM_PROVIDER": "none",
        "TRANSCRIPTION_PROVIDER": "none",
        "EMBEDDINGS_ENABLED": "false",
        "SEED_STARTER_ROLE": "false",
        "ADMIN_EMAIL": "admin@example.com",
        "ADMIN_PASSWORD": "supersecret123",
        "RATE_LIMIT_ENABLED": "false",
        "PUBLIC_BASE_URL": "http://testserver",
    }
)

from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.models import Question, Role  # noqa: E402
from app.seed import ensure_admin  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        ensure_admin(db)
    finally:
        db.close()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    with TestClient(fastapi_app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "supersecret123"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def role(db):
    """A role with two questions, created directly so tests stay independent."""
    role = Role(title="Backend Engineer", department="Engineering", description="Python and APIs")
    role.questions.append(Question(text="Tell us about your Python experience.", order_index=0))
    role.questions.append(Question(text="Describe a hard bug you fixed.", order_index=1))
    db.add(role)
    db.commit()
    db.refresh(role)
    yield role


@pytest.fixture
def fake_audio() -> bytes:
    """A minimal but structurally valid WAV file."""
    import struct

    sample_rate = 8000
    frames = b"\x00\x00" * sample_rate  # 1 second of silence, 16-bit mono
    header = b"RIFF" + struct.pack("<I", 36 + len(frames)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(frames))
    return header + frames
