"""The full candidate journey: invite -> record -> submit -> report."""
from __future__ import annotations

import pytest

from app.models import Interview, InterviewStatus

RESUME = (
    "Senior Python engineer with 6 years of experience building FastAPI and Django services. "
    "Comfortable with PostgreSQL, Docker, AWS and Redis. Led the migration of a monolith to services."
)


@pytest.fixture
def invited(client, auth_headers, role, monkeypatch):
    """Invite a candidate without touching the analysis worker."""
    queued: list[int] = []
    monkeypatch.setattr("app.routers.public.enqueue", queued.append)
    monkeypatch.setattr("app.routers.interviews.enqueue", queued.append)

    response = client.post(
        "/api/interviews/invite",
        headers=auth_headers,
        json={
            "full_name": "Asha Rao",
            "email": "asha.rao@example.com",
            "role_id": role.id,
            "send_email": False,
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    token = payload["invite_url"].rsplit("/", 1)[-1]
    return {"payload": payload, "token": token, "headers": {"X-Interview-Token": token}, "queued": queued}


def test_invite_creates_link(invited):
    assert invited["payload"]["invite_url"].startswith("http://testserver/interview/")
    assert invited["payload"]["email_sent"] is False


def test_duplicate_open_invite_is_rejected(client, auth_headers, role, invited):
    response = client.post(
        "/api/interviews/invite",
        headers=auth_headers,
        json={"full_name": "Asha Rao", "email": "asha.rao@example.com", "role_id": role.id, "send_email": False},
    )
    assert response.status_code == 409


def test_session_requires_valid_token(client):
    assert client.get("/api/public/session").status_code == 401
    assert client.get("/api/public/session", headers={"X-Interview-Token": "garbage"}).status_code == 401


def test_candidate_completes_interview(client, invited, role, fake_audio, db):
    headers = invited["headers"]

    session = client.get("/api/public/session", headers=headers)
    assert session.status_code == 200
    body = session.json()
    assert body["candidate_name"] == "Asha Rao"
    assert len(body["questions"]) == 2
    assert body["resume_on_file"] is False

    assert client.post("/api/public/session/start", headers=headers).json()["status"] == "in_progress"

    # Submitting early fails: no resume, no recordings.
    early = client.post("/api/public/submit", headers=headers)
    assert early.status_code == 422

    resume = client.post("/api/public/resume", headers=headers, data={"resume_text": RESUME})
    assert resume.status_code == 200, resume.text

    for question in body["questions"]:
        upload = client.post(
            f"/api/public/answers/{question['id']}",
            headers=headers,
            files={"audio": ("answer.wav", fake_audio, "audio/wav")},
            data={"duration_seconds": "42.5"},
        )
        assert upload.status_code == 200, upload.text

    submitted = client.post("/api/public/submit", headers=headers)
    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["status"] == "submitted"
    assert invited["queued"], "submission should have been queued for analysis"

    # A second submission is rejected.
    assert client.post("/api/public/submit", headers=headers).status_code == 409

    interview = db.query(Interview).filter(Interview.public_id == submitted.json()["public_id"]).one()
    db.refresh(interview)
    assert interview.status == InterviewStatus.SUBMITTED
    assert len(interview.answers) == 2
    assert interview.candidate.resume_text.startswith("Senior Python engineer")


def test_rejects_unsupported_audio_type(client, invited, role, fake_audio):
    question_id = client.get("/api/public/session", headers=invited["headers"]).json()["questions"][0]["id"]
    response = client.post(
        f"/api/public/answers/{question_id}",
        headers=invited["headers"],
        files={"audio": ("answer.exe", fake_audio, "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_rejects_answer_for_foreign_question(client, invited):
    response = client.post(
        "/api/public/answers/999999",
        headers=invited["headers"],
        files={"audio": ("a.wav", b"RIFF0000WAVE", "audio/wav")},
    )
    assert response.status_code == 404


def test_short_pasted_resume_is_rejected(client, invited):
    response = client.post("/api/public/resume", headers=invited["headers"], data={"resume_text": "python"})
    assert response.status_code == 422


def test_recruiter_sees_interview_in_dashboard(client, auth_headers, invited):
    dashboard = client.get("/api/dashboard", headers=auth_headers)
    assert dashboard.status_code == 200
    stats = dashboard.json()
    assert stats["total_interviews"] >= 1
    assert stats["total_candidates"] >= 1

    listing = client.get("/api/interviews", headers=auth_headers, params={"search": "asha"})
    assert listing.status_code == 200
    assert listing.json()[0]["candidate"]["email"] == "asha.rao@example.com"

    detail = client.get(f"/api/interviews/{invited['payload']['interview_id']}", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["role"]["title"]


def test_resend_report_requires_completed_interview(client, auth_headers, invited):
    response = client.post(
        f"/api/interviews/{invited['payload']['interview_id']}/resend-report", headers=auth_headers
    )
    assert response.status_code == 409
