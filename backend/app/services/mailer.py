"""Transactional email over SMTP (Gmail by default).

Gmail needs an *App Password*, not the account password:
  Google Account -> Security -> 2-Step Verification -> App passwords
Put the 16-character value in SMTP_PASSWORD (spaces removed).
"""
from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
import ssl
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import settings

logger = logging.getLogger(__name__)

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")


class MailError(RuntimeError):
    """Raised when an email could not be handed to the SMTP server."""


@dataclass
class Attachment:
    filename: str
    content: bytes
    mimetype: str | None = None


@lru_cache
def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda value: f"{float(value or 0) * 100:.0f}%"
    env.filters["score"] = lambda value: f"{float(value or 0):.1f}"
    return env


def render(template_name: str, **context) -> str:
    context.setdefault("app_name", settings.APP_NAME)
    context.setdefault("base_url", settings.PUBLIC_BASE_URL.rstrip("/"))
    return _environment().get_template(template_name).render(**context)


def html_to_text(html: str) -> str:
    """Very small HTML -> text reduction for the plain-text alternative part."""
    import re

    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&mdash;", "—")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_message(
    to: Sequence[str],
    subject: str,
    html: str,
    text: str | None = None,
    cc: Sequence[str] | None = None,
    reply_to: str | None = None,
    attachments: Iterable[Attachment] | None = None,
) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr((settings.MAIL_FROM_NAME, settings.MAIL_FROM))
    message["To"] = ", ".join(to)
    if cc:
        message["Cc"] = ", ".join(cc)
    reply = reply_to or settings.MAIL_REPLY_TO
    if reply:
        message["Reply-To"] = reply
    message["Message-ID"] = make_msgid(domain=settings.MAIL_FROM.split("@")[-1])

    message.set_content(text or html_to_text(html))
    message.add_alternative(html, subtype="html")

    for attachment in attachments or []:
        mimetype = attachment.mimetype or mimetypes.guess_type(attachment.filename)[0] or "application/octet-stream"
        maintype, _, subtype = mimetype.partition("/")
        message.add_attachment(
            attachment.content,
            maintype=maintype,
            subtype=subtype or "octet-stream",
            filename=attachment.filename,
        )
    return message


def send_email(
    to: Sequence[str] | str,
    subject: str,
    html: str,
    text: str | None = None,
    cc: Sequence[str] | None = None,
    reply_to: str | None = None,
    attachments: Iterable[Attachment] | None = None,
) -> bool:
    """Send one email. Returns True on success, raises MailError on failure."""
    recipients: list[str] = [to] if isinstance(to, str) else [address for address in to if address]
    if not recipients:
        raise MailError("No recipient address given")

    if not settings.MAIL_ENABLED:
        logger.info("MAIL_ENABLED=false — skipping email '%s' to %s", subject, recipients)
        return False

    if not settings.SMTP_PASSWORD:
        raise MailError(
            "SMTP_PASSWORD is not set. Generate a Gmail App Password and put it in your .env file."
        )

    message = _build_message(recipients, subject, html, text, cc, reply_to, attachments)
    all_recipients = recipients + [address for address in (cc or []) if address]

    context = ssl.create_default_context()
    try:
        if settings.SMTP_SSL:
            with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, context=context, timeout=30) as server:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(message, to_addrs=all_recipients)
        else:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=30) as server:
                server.ehlo()
                if settings.SMTP_STARTTLS:
                    server.starttls(context=context)
                    server.ehlo()
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(message, to_addrs=all_recipients)
    except smtplib.SMTPAuthenticationError as exc:
        raise MailError(
            "SMTP authentication failed. For Gmail you must use a 16-character App Password "
            "with 2-Step Verification enabled, not your normal password."
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise MailError(f"Could not send email: {exc}") from exc

    logger.info("Sent '%s' to %s", subject, all_recipients)
    return True


def verify_connection() -> dict[str, object]:
    """Check SMTP credentials without sending anything. Used by /api/health/email."""
    if not settings.SMTP_PASSWORD:
        return {"ok": False, "detail": "SMTP_PASSWORD is not configured"}
    try:
        context = ssl.create_default_context()
        if settings.SMTP_SSL:
            with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, context=context, timeout=15) as server:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        else:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
                server.ehlo()
                if settings.SMTP_STARTTLS:
                    server.starttls(context=context)
                    server.ehlo()
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        return {"ok": True, "detail": f"Authenticated as {settings.SMTP_USERNAME}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)}
