"""Email rendering, PDF generation and the persist+notify pipeline."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from app.models import Answer, Candidate, Interview, InterviewStatus, Question, Role
from app.services import mailer, notifications
from app.services.report_pdf import build_report_pdf, pdf_available


@pytest.fixture
def completed_interview(db):
    # Candidate emails are unique, so each test gets its own record.
    unique = uuid.uuid4().hex[:8]
    role = Role(title="Platform Engineer", department="Infra", description="Kubernetes and CI")
    question = Question(text="How do you approach on-call?", order_index=0)
    role.questions.append(question)
    candidate = Candidate(
        full_name="Rahul Mehta",
        email=f"rahul.mehta+{unique}@example.com",
        resume_text="Platform engineer with Kubernetes, Terraform and Go experience.",
    )
    interview = Interview(
        candidate=candidate,
        role=role,
        status=InterviewStatus.COMPLETED,
        submitted_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        overall_score=8.4,
        relevance_score=0.72,
        verdict="strong_hire",
        dimension_scores={
            "communication": 8.0,
            "technical_relevance": 9.0,
            "confidence": 8.0,
            "clarity": 8.5,
            "overall_quality": 8.5,
        },
        skills_resume=["Kubernetes", "Terraform", "Go"],
        skills_answer=["Kubernetes", "Prometheus"],
        summary=["Deep operational experience.", "Clear incident-response instincts."],
        strengths=["Concrete on-call examples."],
        improvements=["Could speak more about prevention."],
    )
    interview.answers.append(
        Answer(
            question=question,
            transcript="I run a weekly on-call review and keep runbooks in the repo.",
            scores={"communication": 8.0, "technical_relevance": 9.0},
            skills=["Kubernetes"],
        )
    )
    db.add(interview)
    db.commit()
    db.refresh(interview)
    yield interview


def test_candidate_email_renders_scores(completed_interview):
    html = mailer.render(
        "emails/report_candidate.html",
        subject="Your results",
        company_name="Acme",
        candidate_name="Rahul",
        role_title="Platform Engineer",
        overall_score=8.4,
        relevance_score=0.72,
        verdict_label="Strong match",
        score_bg="#047857",
        dimensions=[("Communication", 8.0), ("Clarity", 8.5)],
        summary=["Deep operational experience."],
        strengths=[],
        improvements=[],
        skills_answer=["Kubernetes"],
        attached_pdf=True,
    )
    assert "8.4" in html
    assert "72%" in html
    assert "Strong match" in html
    assert "Deep operational experience." in html
    assert "<script" not in html.lower()


def test_invite_email_renders_link(completed_interview):
    html = mailer.render(
        "emails/invite.html",
        subject="Invite",
        company_name="Acme",
        candidate_name="Rahul",
        role_title="Platform Engineer",
        department="Infra",
        question_count=3,
        estimated_minutes=12,
        invite_url="https://hire.acme.com/interview/abc123",
        expires_at="10 Sep 2026, 09:00 UTC",
    )
    assert "https://hire.acme.com/interview/abc123" in html
    assert "3 questions" in html


def test_html_to_text_strips_markup():
    text = mailer.html_to_text("<p>Hello <strong>Rahul</strong></p><br><script>bad()</script>")
    assert "Hello" in text and "Rahul" in text
    assert "bad()" not in text
    assert "<" not in text


def test_email_escapes_untrusted_candidate_name():
    html = mailer.render(
        "emails/invite.html",
        subject="Invite",
        company_name="Acme",
        candidate_name="<img src=x onerror=alert(1)>",
        role_title="Engineer",
        department="",
        question_count=1,
        estimated_minutes=5,
        invite_url="https://example.com/i/1",
        expires_at="",
    )
    assert "<img src=x" not in html
    assert "&lt;img" in html


@pytest.mark.skipif(not pdf_available(), reason="reportlab is not installed")
def test_pdf_report_is_generated(completed_interview):
    attachment = notifications.build_pdf_attachment(completed_interview)
    assert attachment is not None
    assert attachment.content[:4] == b"%PDF"
    assert len(attachment.content) > 2000
    assert attachment.filename.endswith(".pdf")


@pytest.mark.skipif(not pdf_available(), reason="reportlab is not installed")
def test_pdf_handles_long_content_without_error():
    content = build_report_pdf(
        candidate_name="Very Long Name " * 5,
        candidate_email="long@example.com",
        role_title="Staff Engineer",
        company_name="Acme",
        overall_score=6.1,
        verdict_label="Needs review",
        relevance_score=0.33,
        dimensions=[("Communication", 6.0)],
        summary=["A very long summary point. " * 30],
        answers=[{"question": "Q? " * 40, "transcript": "Answer. " * 400, "scores": {"clarity": 6}}],
    )
    assert content and content[:4] == b"%PDF"


def test_send_email_is_skipped_when_disabled(completed_interview):
    # MAIL_ENABLED=false in the test environment.
    assert notifications.send_candidate_report(completed_interview) is False


def test_try_send_captures_mail_errors():
    def boom():
        raise mailer.MailError("SMTP is down")

    sent, error = notifications.try_send(boom, "test email")
    assert sent is False
    assert "SMTP is down" in error


def test_pipeline_persists_results_and_completes(db, completed_interview, monkeypatch):
    """process_interview should write scores back and mark the interview completed."""
    from app.services import analyzer, pipeline

    completed_interview.status = InterviewStatus.SUBMITTED
    completed_interview.overall_score = None
    completed_interview.completed_at = None
    db.commit()
    interview_id = completed_interview.id

    def fake_analysis(role_title, resume_text, answers, on_progress=None):
        result = analyzer.InterviewAnalysis()
        for item in answers:
            result.answers.append(
                analyzer.AnswerAnalysis(
                    question_id=item.question_id,
                    question_text=item.question_text,
                    transcript="A transcribed answer about Kubernetes.",
                    scores=dict.fromkeys(analyzer.DIMENSIONS, 8.0),
                    skills=["Kubernetes"],
                    feedback="Good.",
                    duration_seconds=30.0,
                )
            )
        result.dimension_scores = dict.fromkeys(analyzer.DIMENSIONS, 8.0)
        result.skills_resume = ["Kubernetes", "Go"]
        result.skills_answer = ["Kubernetes"]
        result.relevance_score = 0.8
        result.overall_score = 8.0
        result.verdict = "shortlist"
        result.summary = ["Solid platform experience."]
        result.meta = {"answers_analyzed": len(answers)}
        return result

    monkeypatch.setattr(pipeline, "analyze_interview", fake_analysis)
    pipeline.process_interview(interview_id)

    db.expire_all()
    refreshed = db.get(Interview, interview_id)
    assert refreshed.status == InterviewStatus.COMPLETED
    assert refreshed.overall_score == 8.0
    assert refreshed.verdict == "shortlist"
    assert refreshed.completed_at is not None
    assert refreshed.answers[0].transcript.startswith("A transcribed answer")
    # Mail is disabled in tests, so nothing was sent.
    assert refreshed.candidate_email_sent_at is None


def test_pipeline_marks_failure(db, completed_interview, monkeypatch):
    from app.services import analyzer, pipeline

    completed_interview.status = InterviewStatus.SUBMITTED
    db.commit()
    interview_id = completed_interview.id

    def boom(*args, **kwargs):
        raise analyzer.AnalysisError("None of the recordings could be transcribed")

    monkeypatch.setattr(pipeline, "analyze_interview", boom)
    pipeline.process_interview(interview_id)

    db.expire_all()
    refreshed = db.get(Interview, interview_id)
    assert refreshed.status == InterviewStatus.FAILED
    assert "None of the recordings could be transcribed" == refreshed.error_message
