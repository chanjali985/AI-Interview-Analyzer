"""The interview analysis pipeline.

Given a set of (question, recorded answer) pairs plus the candidate's resume,
produce transcripts, per-dimension scores, extracted skills, a resume/answer
relevance score, an overall weighted score, a verdict, and a written summary.

Every LLM interaction is defensive: a malformed model response degrades to a
documented default rather than taking the whole interview down.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from . import embeddings
from .llm import LLMError, chat_json, get_llm
from .transcription import Transcript, TranscriptionError, transcribe

logger = logging.getLogger(__name__)

DIMENSIONS = ("communication", "technical_relevance", "confidence", "clarity", "overall_quality")

DIMENSION_LABELS = {
    "communication": "Communication",
    "technical_relevance": "Technical relevance",
    "confidence": "Confidence",
    "clarity": "Clarity",
    "overall_quality": "Overall quality",
}

SCORER_SYSTEM = (
    "You are a senior technical interviewer. You grade fairly and consistently, "
    "you never invent details that are not in the answer, and you reply with JSON only."
)


class AnalysisError(RuntimeError):
    """Raised when an interview cannot be analysed at all."""


@dataclass
class AnswerInput:
    """One question and the file holding the candidate's spoken answer."""

    question_id: int
    question_text: str
    audio_path: str | None = None
    transcript: str | None = None


@dataclass
class AnswerAnalysis:
    question_id: int
    question_text: str
    transcript: str
    scores: dict[str, float]
    skills: list[str]
    feedback: str = ""
    duration_seconds: float | None = None
    transcription_provider: str = ""


@dataclass
class InterviewAnalysis:
    answers: list[AnswerAnalysis] = field(default_factory=list)
    dimension_scores: dict[str, float] = field(default_factory=dict)
    skills_resume: list[str] = field(default_factory=list)
    skills_answer: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    overall_score: float = 0.0
    verdict: str = "review"
    summary: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- helpers
def _clamp_score(value: Any, default: float = 5.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return round(max(1.0, min(10.0, number)), 2)


def _clean_string_list(raw: Any, limit: int = 40) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, dict):
        for key in ("skills", "skill_list", "items", "result"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raw = list(raw.values())
    if not isinstance(raw, (list, tuple, set)):
        return []

    seen: dict[str, str] = {}
    for item in raw:
        if isinstance(item, dict):
            item = item.get("name") or item.get("skill") or ""
        if not isinstance(item, str):
            continue
        value = item.strip().strip(".,;:")
        if not value or len(value) > 60:
            continue
        key = value.lower()
        if key not in seen:
            seen[key] = value
        if len(seen) >= limit:
            break
    return list(seen.values())


def _heuristic_scores(question: str, answer: str) -> dict[str, float]:
    """Deterministic fallback so a model outage does not produce garbage."""
    words = answer.split()
    length_score = min(10.0, max(1.0, len(words) / 25.0))
    question_words = {w.lower().strip(".,?") for w in question.split() if len(w) > 4}
    answer_words = {w.lower().strip(".,?") for w in words if len(w) > 4}
    overlap = len(question_words & answer_words) / max(1, len(question_words))
    relevance = min(10.0, max(1.0, 3.0 + overlap * 7.0))
    return {
        "communication": round(length_score, 2),
        "technical_relevance": round(relevance, 2),
        "confidence": round(min(10.0, length_score * 0.9), 2),
        "clarity": round(length_score, 2),
        "overall_quality": round((length_score + relevance) / 2, 2),
    }


# ------------------------------------------------------------- LLM operations
def score_answer(question: str, answer: str) -> dict[str, Any]:
    """Score one answer on the five dimensions and return short written feedback."""
    if not answer.strip():
        return {"scores": dict.fromkeys(DIMENSIONS, 1.0), "feedback": "No answer was recorded."}

    prompt = f"""Grade this interview answer.

QUESTION:
{question}

CANDIDATE ANSWER (transcribed from audio, so ignore minor transcription noise):
{answer[:4000]}

Score each dimension from 1 to 10, where 1 is unacceptable, 5 is average and 10 is outstanding:
- communication: structure and articulation of the response
- technical_relevance: does the answer actually address the question with correct substance
- confidence: conviction and ownership shown in the response
- clarity: how easy the answer is to follow
- overall_quality: your holistic judgement

Also write 1-2 sentences of specific, actionable feedback the candidate could act on.

Reply with JSON only, exactly this shape:
{{"communication": 0, "technical_relevance": 0, "confidence": 0, "clarity": 0, "overall_quality": 0, "feedback": "..."}}"""

    try:
        parsed = chat_json(prompt, system=SCORER_SYSTEM)
    except LLMError as exc:
        logger.error("Scoring failed: %s", exc)
        parsed = None

    if not isinstance(parsed, dict):
        logger.warning("Falling back to heuristic scoring for one answer")
        return {"scores": _heuristic_scores(question, answer), "feedback": "", "degraded": True}

    scores = {dim: _clamp_score(parsed.get(dim)) for dim in DIMENSIONS}
    feedback = parsed.get("feedback") or ""
    if not isinstance(feedback, str):
        feedback = ""
    return {"scores": scores, "feedback": feedback.strip()[:600], "degraded": False}


def extract_skills(text: str, source: str = "answer") -> list[str]:
    """Pull technical skills, tools and technologies out of a block of text."""
    if not text or not text.strip():
        return []

    prompt = f"""List the technical skills, programming languages, frameworks, tools, platforms and
methodologies explicitly mentioned in this {source}.

{source.upper()}:
{text[:4000]}

Rules:
- Only include things that are actually mentioned. Do not infer or add related technologies.
- Use the canonical name (e.g. "PostgreSQL" not "postgres db").
- No duplicates, no soft skills, no job titles.

Reply with JSON only: {{"skills": ["Python", "Django", "AWS"]}}"""

    try:
        parsed = chat_json(prompt, system="You extract structured data and reply with JSON only.")
    except LLMError as exc:
        logger.error("Skill extraction failed: %s", exc)
        return []

    if isinstance(parsed, dict):
        parsed = parsed.get("skills", parsed)
    return _clean_string_list(parsed)


def generate_summary(
    role_title: str,
    answers: Sequence[AnswerAnalysis],
    dimension_scores: dict[str, float],
    overall_score: float,
    relevance_score: float,
    skills_resume: Sequence[str],
    skills_answer: Sequence[str],
) -> dict[str, list[str]]:
    """Write the recruiter-facing narrative for the whole interview."""
    transcript_digest = "\n\n".join(
        f"Q: {a.question_text}\nA: {a.transcript[:600]}" for a in answers[:6]
    )
    score_lines = "\n".join(f"- {DIMENSION_LABELS[k]}: {v}/10" for k, v in dimension_scores.items())

    prompt = f"""Write the evaluation summary for a candidate interviewing for: {role_title}.

AVERAGE SCORES
{score_lines}
Overall weighted score: {overall_score}/10
Resume-to-answer skill relevance: {relevance_score:.0%}
Skills on resume: {', '.join(list(skills_resume)[:15]) or 'none detected'}
Skills demonstrated in answers: {', '.join(list(skills_answer)[:15]) or 'none detected'}

TRANSCRIPTS
{transcript_digest[:6000]}

Produce JSON only:
{{
  "summary": ["3 to 5 bullet points a hiring manager can read in 20 seconds"],
  "strengths": ["2 to 4 concrete strengths evidenced by the answers"],
  "improvements": ["2 to 4 specific, kind, actionable areas to improve"]
}}

Be specific and evidence-based. Reference what the candidate actually said. No praise without evidence."""

    try:
        parsed = chat_json(prompt, system="You are a hiring analyst who replies with JSON only.", temperature=0.3)
    except LLMError as exc:
        logger.error("Summary generation failed: %s", exc)
        parsed = None

    if not isinstance(parsed, dict):
        return {
            "summary": [
                f"Overall weighted score: {overall_score}/10.",
                f"Resume-to-answer skill relevance: {relevance_score:.0%}.",
                f"Strongest dimension: {_best_dimension(dimension_scores)}.",
                f"Answered {len(answers)} question(s).",
            ],
            "strengths": [],
            "improvements": [],
        }

    return {
        "summary": _clean_string_list(parsed.get("summary"), limit=6) or [f"Overall score {overall_score}/10."],
        "strengths": _clean_string_list(parsed.get("strengths"), limit=5),
        "improvements": _clean_string_list(parsed.get("improvements"), limit=5),
    }


def _best_dimension(scores: dict[str, float]) -> str:
    if not scores:
        return "n/a"
    key = max(scores, key=scores.get)
    return f"{DIMENSION_LABELS.get(key, key)} ({scores[key]}/10)"


# ------------------------------------------------------------------ scoring
def aggregate_dimensions(per_answer: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_answer:
        return dict.fromkeys(DIMENSIONS, 0.0)
    return {
        dim: round(sum(scores.get(dim, 0.0) for scores in per_answer) / len(per_answer), 2)
        for dim in DIMENSIONS
    }


def weighted_overall(dimension_scores: dict[str, float], relevance_score: float) -> float:
    """Fold dimensions into one 1-10 score, nudged by resume/answer relevance."""
    weights = {
        "communication": settings.WEIGHT_COMMUNICATION,
        "technical_relevance": settings.WEIGHT_TECHNICAL,
        "confidence": settings.WEIGHT_CONFIDENCE,
        "clarity": settings.WEIGHT_CLARITY,
        "overall_quality": settings.WEIGHT_OVERALL,
    }
    total_weight = sum(weights.values()) or 1.0
    base = sum(dimension_scores.get(dim, 0.0) * weight for dim, weight in weights.items()) / total_weight

    relevance_weight = max(0.0, min(1.0, settings.RELEVANCE_WEIGHT))
    blended = base * (1 - relevance_weight) + (relevance_score * 10.0) * relevance_weight
    return round(max(1.0, min(10.0, blended)), 2)


def decide_verdict(overall_score: float) -> str:
    if overall_score >= settings.SHORTLIST_THRESHOLD + 1.5:
        return "strong_hire"
    if overall_score >= settings.SHORTLIST_THRESHOLD:
        return "shortlist"
    if overall_score >= settings.SHORTLIST_THRESHOLD - 2.0:
        return "review"
    return "not_recommended"


VERDICT_LABELS = {
    "strong_hire": "Strong match",
    "shortlist": "Shortlisted",
    "review": "Needs review",
    "not_recommended": "Not a match right now",
}


# ------------------------------------------------------------------ pipeline
def analyze_interview(
    role_title: str,
    resume_text: str,
    answers: Sequence[AnswerInput],
    on_progress=None,
) -> InterviewAnalysis:
    """Run the full pipeline. Raises AnalysisError only when nothing can be salvaged."""
    if not answers:
        raise AnalysisError("This interview has no recorded answers")

    def progress(message: str) -> None:
        logger.info(message)
        if on_progress:
            try:
                on_progress(message)
            except Exception:  # noqa: BLE001 - progress reporting must never break analysis
                pass

    result = InterviewAnalysis()
    degraded_answers = 0
    transcription_failures: list[str] = []

    # ---- 1. transcribe -----------------------------------------------------
    transcripts: list[Transcript] = []
    for index, item in enumerate(answers, start=1):
        progress(f"Transcribing answer {index}/{len(answers)}")
        if item.transcript:
            transcripts.append(Transcript(text=item.transcript, provider="provided"))
            continue
        if not item.audio_path:
            transcripts.append(Transcript(text="", provider="missing"))
            transcription_failures.append(f"Q{index}: no recording was submitted")
            continue
        try:
            transcripts.append(transcribe(item.audio_path))
        except TranscriptionError as exc:
            logger.warning("Transcription failed for question %s: %s", item.question_id, exc)
            transcripts.append(Transcript(text="", provider="failed"))
            transcription_failures.append(f"Q{index}: {exc}")

    if all(not t.text.strip() for t in transcripts):
        raise AnalysisError(
            "None of the recordings could be transcribed. " + "; ".join(transcription_failures)
        )

    # ---- 2. score each answer ---------------------------------------------
    per_answer_scores: list[dict[str, float]] = []
    for index, (item, transcript) in enumerate(zip(answers, transcripts, strict=True), start=1):
        progress(f"Scoring answer {index}/{len(answers)}")
        scored = score_answer(item.question_text, transcript.text)
        if scored.get("degraded"):
            degraded_answers += 1
        answer_skills = extract_skills(transcript.text, "answer") if transcript.text.strip() else []
        result.answers.append(
            AnswerAnalysis(
                question_id=item.question_id,
                question_text=item.question_text,
                transcript=transcript.text,
                scores=scored["scores"],
                skills=answer_skills,
                feedback=scored.get("feedback", ""),
                duration_seconds=transcript.duration_seconds,
                transcription_provider=transcript.provider,
            )
        )
        per_answer_scores.append(scored["scores"])

    # ---- 3. skills + relevance --------------------------------------------
    progress("Extracting resume skills")
    result.skills_resume = extract_skills(resume_text, "resume") if resume_text else []
    combined_answer_skills: list[str] = []
    for answer in result.answers:
        for skill in answer.skills:
            if skill.lower() not in {s.lower() for s in combined_answer_skills}:
                combined_answer_skills.append(skill)
    result.skills_answer = combined_answer_skills

    progress("Computing skill relevance")
    result.relevance_score = embeddings.skill_similarity(result.skills_resume, result.skills_answer)

    # ---- 4. aggregate ------------------------------------------------------
    result.dimension_scores = aggregate_dimensions(per_answer_scores)
    result.overall_score = weighted_overall(result.dimension_scores, result.relevance_score)
    result.verdict = decide_verdict(result.overall_score)

    # ---- 5. narrative ------------------------------------------------------
    progress("Writing summary")
    narrative = generate_summary(
        role_title,
        result.answers,
        result.dimension_scores,
        result.overall_score,
        result.relevance_score,
        result.skills_resume,
        result.skills_answer,
    )
    result.summary = narrative["summary"]
    result.strengths = narrative["strengths"]
    result.improvements = narrative["improvements"]

    result.meta = {
        "llm_provider": get_llm().name,
        "llm_model": get_llm().model,
        "transcription_provider": settings.TRANSCRIPTION_PROVIDER,
        "embeddings_used": embeddings.embeddings_ready(),
        "degraded_answers": degraded_answers,
        "transcription_failures": transcription_failures,
        "answers_analyzed": len(result.answers),
    }
    progress("Analysis complete")
    return result
