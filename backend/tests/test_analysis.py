"""Analyzer internals and the full pipeline with stubbed models."""
from __future__ import annotations

import pytest

from app.services import analyzer, embeddings
from app.services.llm import extract_json
from app.services.transcription import Transcript


# ------------------------------------------------------------------ utilities
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 2}\n```', {"a": 2}),
        ('Sure! Here you go: {"a": 3} — hope that helps', {"a": 3}),
        ('{"nested": {"b": [1, 2]}, "c": "}"}', {"nested": {"b": [1, 2]}, "c": "}"}),
        ('["Python", "Django"]', ["Python", "Django"]),
        ("no json at all", None),
        ("", None),
    ],
)
def test_extract_json(raw, expected):
    assert extract_json(raw) == expected


def test_clamp_score_bounds():
    assert analyzer._clamp_score(15) == 10.0
    assert analyzer._clamp_score(-3) == 1.0
    assert analyzer._clamp_score("7.5") == 7.5
    assert analyzer._clamp_score(None) == 5.0
    assert analyzer._clamp_score("not a number") == 5.0


def test_clean_string_list_deduplicates_and_trims():
    result = analyzer._clean_string_list(["Python", " python ", "Django.", 42, None, {"name": "AWS"}])
    assert result == ["Python", "Django", "AWS"]


def test_weighted_overall_respects_relevance():
    dimensions = dict.fromkeys(analyzer.DIMENSIONS, 8.0)
    high = analyzer.weighted_overall(dimensions, relevance_score=1.0)
    low = analyzer.weighted_overall(dimensions, relevance_score=0.0)
    assert high > low
    assert 1.0 <= low <= high <= 10.0


def test_verdict_thresholds():
    assert analyzer.decide_verdict(9.5) == "strong_hire"
    assert analyzer.decide_verdict(7.2) == "shortlist"
    assert analyzer.decide_verdict(5.5) == "review"
    assert analyzer.decide_verdict(3.0) == "not_recommended"


def test_aggregate_dimensions_averages():
    aggregated = analyzer.aggregate_dimensions(
        [dict.fromkeys(analyzer.DIMENSIONS, 6.0), dict.fromkeys(analyzer.DIMENSIONS, 8.0)]
    )
    assert all(value == 7.0 for value in aggregated.values())


def test_lexical_similarity_fallback():
    # Embeddings are disabled in tests, so this exercises the fallback path.
    assert embeddings.skill_similarity(["Python", "Django"], ["Python", "Django"]) > 0.9
    assert embeddings.skill_similarity(["Python"], ["Photoshop"]) == 0.0
    assert embeddings.skill_similarity([], ["Python"]) == 0.0


def test_matched_skills_split():
    matched, extra = embeddings.matched_skills(["Python", "AWS"], ["Python", "Kubernetes"])
    assert matched == ["Python"]
    assert extra == ["Kubernetes"]


# ------------------------------------------------------------------- pipeline
@pytest.fixture
def stub_models(monkeypatch):
    """Deterministic transcription + LLM so the pipeline can be asserted on."""

    def fake_transcribe(path):
        return Transcript(text="I built REST APIs in Python with Django and deployed them on AWS.",
                          duration_seconds=31.0, provider="stub")

    def fake_chat_json(prompt, system=None, temperature=0.1):
        if "Grade this interview answer" in prompt:
            return {
                "communication": 8,
                "technical_relevance": 9,
                "confidence": 7,
                "clarity": 8,
                "overall_quality": 8,
                "feedback": "Clear and concrete, could add measurable outcomes.",
            }
        if "List the technical skills" in prompt:
            return {"skills": ["Python", "Django", "AWS"]}
        return {
            "summary": ["Strong hands-on Python background."],
            "strengths": ["Concrete API examples."],
            "improvements": ["Quantify the impact."],
        }

    monkeypatch.setattr(analyzer, "transcribe", fake_transcribe)
    monkeypatch.setattr(analyzer, "chat_json", fake_chat_json)


def test_analyze_interview_end_to_end(stub_models, tmp_path):
    audio = tmp_path / "answer.wav"
    audio.write_bytes(b"RIFF....WAVE")

    result = analyzer.analyze_interview(
        role_title="Backend Engineer",
        resume_text="Python developer with Django and AWS experience.",
        answers=[
            analyzer.AnswerInput(question_id=1, question_text="Tell us about your Python experience.",
                                 audio_path=str(audio)),
            analyzer.AnswerInput(question_id=2, question_text="Describe a hard bug.", audio_path=str(audio)),
        ],
    )

    assert len(result.answers) == 2
    assert result.answers[0].transcript.startswith("I built REST APIs")
    assert result.dimension_scores["technical_relevance"] == 9.0
    assert result.skills_answer == ["Python", "Django", "AWS"]
    assert result.relevance_score > 0.5
    assert 7.0 <= result.overall_score <= 10.0
    assert result.verdict in {"shortlist", "strong_hire"}
    assert result.summary
    assert result.meta["answers_analyzed"] == 2


def test_analysis_fails_when_nothing_transcribes(monkeypatch, tmp_path):
    from app.services.transcription import TranscriptionError

    def failing(path):
        raise TranscriptionError("no speech detected")

    monkeypatch.setattr(analyzer, "transcribe", failing)
    audio = tmp_path / "silent.wav"
    audio.write_bytes(b"RIFF....WAVE")

    with pytest.raises(analyzer.AnalysisError):
        analyzer.analyze_interview(
            role_title="Backend Engineer",
            resume_text="Python",
            answers=[analyzer.AnswerInput(1, "Question?", audio_path=str(audio))],
        )


def test_scoring_degrades_gracefully_without_llm(monkeypatch):
    """With no model reachable, we still produce bounded scores rather than crash."""
    from app.services.llm import LLMError

    def unavailable(*args, **kwargs):
        raise LLMError("no provider configured")

    monkeypatch.setattr(analyzer, "chat_json", unavailable)
    result = analyzer.score_answer("Tell us about Python", "I have used Python for five years " * 10)
    assert result["degraded"] is True
    assert set(result["scores"]) == set(analyzer.DIMENSIONS)
    assert all(1.0 <= value <= 10.0 for value in result["scores"].values())
