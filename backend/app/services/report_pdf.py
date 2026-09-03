"""PDF interview report, attached to the candidate and recruiter emails."""
from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

INK = (0.07, 0.09, 0.15)
MUTED = (0.42, 0.45, 0.52)
ACCENT = (0.15, 0.39, 0.92)
TEAL = (0.05, 0.65, 0.64)
RULE = (0.89, 0.90, 0.94)


def _score_color(score: float) -> tuple[float, float, float]:
    if score >= 8.0:
        return (0.08, 0.55, 0.35)
    if score >= 6.5:
        return (0.15, 0.39, 0.92)
    if score >= 5.0:
        return (0.79, 0.51, 0.05)
    return (0.78, 0.18, 0.18)


def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def build_report_pdf(
    *,
    candidate_name: str,
    candidate_email: str,
    role_title: str,
    company_name: str,
    overall_score: float,
    verdict_label: str,
    relevance_score: float,
    dimensions: Sequence[tuple[str, float]],
    summary: Sequence[str],
    strengths: Sequence[str] = (),
    improvements: Sequence[str] = (),
    skills_resume: Sequence[str] = (),
    skills_answer: Sequence[str] = (),
    answers: Sequence[dict[str, Any]] | None = None,
    submitted_at: str = "",
    include_transcripts: bool = True,
) -> bytes | None:
    """Render the report to PDF bytes. Returns None if reportlab is unavailable."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as pdf_canvas
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("reportlab is not installed; skipping PDF report")
        return None

    buffer = io.BytesIO()
    width, height = A4
    pdf = pdf_canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle(f"Interview report - {candidate_name}")

    left = 20 * mm
    right = width - 20 * mm
    y = height - 22 * mm

    def new_page_if_needed(space: float) -> None:
        nonlocal y
        if y - space < 22 * mm:
            pdf.showPage()
            y = height - 22 * mm

    def heading(text: str) -> None:
        nonlocal y
        new_page_if_needed(16 * mm)
        pdf.setFillColorRGB(*MUTED)
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(left, y, text.upper())
        y -= 3.2 * mm
        pdf.setStrokeColorRGB(*RULE)
        pdf.setLineWidth(0.6)
        pdf.line(left, y, right, y)
        y -= 6 * mm

    def bullets(items: Sequence[str]) -> None:
        nonlocal y
        pdf.setFont("Helvetica", 9.8)
        pdf.setFillColorRGB(*INK)
        for item in items:
            for line_index, line in enumerate(_wrap(pdf, str(item), right - left - 6 * mm, "Helvetica", 9.8)):
                new_page_if_needed(7 * mm)
                if line_index == 0:
                    pdf.setFillColorRGB(*ACCENT)
                    pdf.drawString(left, y, "•")
                    pdf.setFillColorRGB(*INK)
                pdf.drawString(left + 5 * mm, y, line)
                y -= 5 * mm
            y -= 1 * mm

    # ------------------------------------------------------------ header band
    pdf.setFillColorRGB(0.07, 0.09, 0.15)
    pdf.rect(0, height - 34 * mm, width, 34 * mm, stroke=0, fill=1)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(left, height - 17 * mm, "Interview evaluation report")
    pdf.setFont("Helvetica", 9.5)
    pdf.setFillColorRGB(0.62, 0.66, 0.74)
    pdf.drawString(left, height - 24 * mm, f"{company_name}  •  {role_title}")
    if submitted_at:
        pdf.drawRightString(right, height - 24 * mm, f"Submitted {submitted_at}")

    y = height - 46 * mm

    # ------------------------------------------------------------ candidate
    pdf.setFillColorRGB(*INK)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(left, y, candidate_name)
    y -= 5.5 * mm
    pdf.setFont("Helvetica", 9.5)
    pdf.setFillColorRGB(*MUTED)
    pdf.drawString(left, y, candidate_email)
    y -= 12 * mm

    # ------------------------------------------------------------ score card
    card_height = 26 * mm
    pdf.setFillColorRGB(0.97, 0.975, 0.985)
    pdf.setStrokeColorRGB(*RULE)
    pdf.roundRect(left, y - card_height, right - left, card_height, 4, stroke=1, fill=1)

    pdf.setFillColorRGB(*MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(left + 8 * mm, y - 8 * mm, "OVERALL SCORE")
    pdf.setFillColorRGB(*_score_color(overall_score))
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(left + 8 * mm, y - 18 * mm, f"{overall_score:.1f}")
    pdf.setFont("Helvetica", 10)
    pdf.setFillColorRGB(*INK)
    pdf.drawString(left + 26 * mm, y - 18 * mm, "/ 10")

    mid = left + (right - left) / 2
    pdf.setFillColorRGB(*MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(mid, y - 8 * mm, "SKILL MATCH")
    pdf.setFillColorRGB(*TEAL)
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(mid, y - 18 * mm, f"{relevance_score * 100:.0f}%")

    pdf.setFillColorRGB(*MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(right - 8 * mm, y - 8 * mm, "RECOMMENDATION")
    pdf.setFillColorRGB(*INK)
    pdf.setFont("Helvetica-Bold", 11.5)
    pdf.drawRightString(right - 8 * mm, y - 17 * mm, verdict_label)

    y -= card_height + 12 * mm

    # ------------------------------------------------------------ dimensions
    heading("Score breakdown")
    bar_left = left + 55 * mm
    bar_width = right - bar_left - 16 * mm
    for label, value in dimensions:
        new_page_if_needed(9 * mm)
        pdf.setFillColorRGB(*INK)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left, y, label)
        pdf.setFillColorRGB(0.91, 0.92, 0.95)
        pdf.roundRect(bar_left, y - 0.6 * mm, bar_width, 3.2 * mm, 1.6, stroke=0, fill=1)
        pdf.setFillColorRGB(*_score_color(float(value)))
        filled = max(0.01, min(1.0, float(value) / 10.0)) * bar_width
        pdf.roundRect(bar_left, y - 0.6 * mm, filled, 3.2 * mm, 1.6, stroke=0, fill=1)
        pdf.setFillColorRGB(*INK)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawRightString(right, y, f"{float(value):.1f}")
        y -= 8 * mm
    y -= 2 * mm

    if summary:
        heading("Assessment summary")
        bullets(summary)
        y -= 2 * mm
    if strengths:
        heading("Strengths")
        bullets(strengths)
        y -= 2 * mm
    if improvements:
        heading("Areas to improve")
        bullets(improvements)
        y -= 2 * mm

    if skills_resume or skills_answer:
        heading("Skills")
        pdf.setFont("Helvetica-Bold", 9.5)
        pdf.setFillColorRGB(*INK)
        for title, values in (("From resume", skills_resume), ("Demonstrated in answers", skills_answer)):
            if not values:
                continue
            new_page_if_needed(12 * mm)
            pdf.setFont("Helvetica-Bold", 9.5)
            pdf.drawString(left, y, title)
            y -= 5 * mm
            pdf.setFont("Helvetica", 9.5)
            for line in _wrap(pdf, ", ".join(values), right - left, "Helvetica", 9.5):
                new_page_if_needed(6 * mm)
                pdf.drawString(left, y, line)
                y -= 5 * mm
            y -= 3 * mm

    if include_transcripts and answers:
        heading("Question by question")
        for index, answer in enumerate(answers, start=1):
            new_page_if_needed(28 * mm)
            pdf.setFont("Helvetica-Bold", 10)
            pdf.setFillColorRGB(*INK)
            for line in _wrap(pdf, f"Q{index}. {answer.get('question', '')}", right - left, "Helvetica-Bold", 10):
                new_page_if_needed(6 * mm)
                pdf.drawString(left, y, line)
                y -= 5.2 * mm
            y -= 1 * mm

            scores = answer.get("scores") or {}
            if scores:
                pdf.setFont("Helvetica", 8.6)
                pdf.setFillColorRGB(*MUTED)
                chips = "   ".join(f"{k.replace('_', ' ')}: {v}" for k, v in scores.items())
                for line in _wrap(pdf, chips, right - left, "Helvetica", 8.6):
                    new_page_if_needed(5 * mm)
                    pdf.drawString(left, y, line)
                    y -= 4.4 * mm
                y -= 1 * mm

            transcript = (answer.get("transcript") or "").strip() or "(no speech detected)"
            pdf.setFont("Helvetica-Oblique", 9.3)
            pdf.setFillColorRGB(0.28, 0.31, 0.38)
            for line in _wrap(pdf, transcript, right - left - 4 * mm, "Helvetica-Oblique", 9.3):
                new_page_if_needed(6 * mm)
                pdf.drawString(left + 2 * mm, y, line)
                y -= 4.8 * mm
            y -= 6 * mm

    # footer on every page
    total_pages = pdf.getPageNumber()
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColorRGB(*MUTED)
    pdf.drawString(left, 12 * mm, f"{company_name} — automated interview analysis. Reviewed by a person before any decision.")
    pdf.drawRightString(right, 12 * mm, f"Page {total_pages}")

    pdf.save()
    return buffer.getvalue()


def _wrap(pdf, text: str, max_width: float, font: str, size: float) -> list[str]:
    """Greedy word wrap using the PDF canvas' real font metrics."""
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdf.stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
