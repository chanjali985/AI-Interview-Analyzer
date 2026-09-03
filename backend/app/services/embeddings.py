"""Skill relevance scoring.

Primary path is Sentence-BERT embeddings + cosine similarity, exactly as the
original project did. If sentence-transformers/torch are not installed (or are
disabled to keep the image small) we fall back to a normalised token-overlap
score so the pipeline still produces a sensible number instead of crashing.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import lru_cache

from ..config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


@lru_cache
def _load_model():
    if not settings.EMBEDDINGS_ENABLED:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info("sentence-transformers not installed; using lexical skill matching")
        return None
    try:
        logger.info("Loading embedding model %s", settings.EMBEDDING_MODEL)
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load embedding model (%s); using lexical fallback", exc)
        return None


def embeddings_ready() -> bool:
    return _load_model() is not None


def _normalise(skill: str) -> str:
    return " ".join(_TOKEN_RE.findall(skill.lower()))


def _tokens(skills: Sequence[str]) -> set:
    bag: set = set()
    for skill in skills:
        bag.update(_TOKEN_RE.findall(skill.lower()))
    return bag


def _lexical_similarity(resume_skills: Sequence[str], answer_skills: Sequence[str]) -> float:
    left, right = _tokens(resume_skills), _tokens(answer_skills)
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    # Coverage of the answer by the resume matters more than raw Jaccard here:
    # a candidate who names three resume skills out of four is a strong match.
    coverage = intersection / len(right)
    jaccard = intersection / len(left | right)
    return round(min(1.0, 0.7 * coverage + 0.3 * jaccard), 4)


def skill_similarity(resume_skills: Sequence[str], answer_skills: Sequence[str]) -> float:
    """Return a 0..1 relevance score between two skill lists."""
    resume_skills = [s for s in (resume_skills or []) if s and s.strip()]
    answer_skills = [s for s in (answer_skills or []) if s and s.strip()]
    if not resume_skills or not answer_skills:
        return 0.0

    model = _load_model()
    if model is None:
        return _lexical_similarity(resume_skills, answer_skills)

    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        resume_vectors = model.encode([_normalise(s) for s in resume_skills])
        answer_vectors = model.encode([_normalise(s) for s in answer_skills])

        resume_mean = np.mean(resume_vectors, axis=0).reshape(1, -1)
        answer_mean = np.mean(answer_vectors, axis=0).reshape(1, -1)
        score = float(cosine_similarity(resume_mean, answer_mean)[0][0])
        return round(max(0.0, min(1.0, score)), 4)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding similarity failed (%s); using lexical fallback", exc)
        return _lexical_similarity(resume_skills, answer_skills)


def matched_skills(
    resume_skills: Sequence[str], answer_skills: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split answer skills into those backed by the resume and those that are not."""
    resume_norm = {_normalise(s): s for s in resume_skills or []}
    matched: list[str] = []
    extra: list[str] = []
    for skill in answer_skills or []:
        key = _normalise(skill)
        if key in resume_norm or any(key and key in existing for existing in resume_norm):
            matched.append(skill)
        else:
            extra.append(skill)
    return matched, extra


def warm_up() -> bool | None:
    """Preload the embedding model at startup so the first request is not slow."""
    return embeddings_ready()
