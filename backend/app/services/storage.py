"""Validated file storage for uploaded audio and resumes."""
from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from collections.abc import Sequence

from fastapi import UploadFile

from ..config import settings

logger = logging.getLogger(__name__)

CHUNK = 1024 * 1024
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


class StorageError(ValueError):
    """Raised when an upload is rejected."""


def safe_name(filename: str) -> str:
    base = os.path.basename(filename or "")
    cleaned = _SAFE.sub("_", base).strip("._") or "upload"
    return cleaned[:120]


def _root() -> str:
    path = os.path.abspath(settings.STORAGE_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _subdir_path(subdir: str) -> str:
    """Sanitise each path segment so nesting survives but traversal cannot."""
    parts = [_SAFE.sub("_", part) for part in str(subdir).replace("\\", "/").split("/") if part not in ("", ".", "..")]
    return os.path.join(_root(), *parts) if parts else _root()


def save_upload(
    upload: UploadFile,
    subdir: str,
    allowed_extensions: Sequence[str],
    max_mb: int,
    stem: str | None = None,
) -> str:
    """Stream an upload to disk with extension and size checks. Returns the path."""
    original = safe_name(upload.filename or "")
    extension = os.path.splitext(original)[1].lower()
    if extension not in allowed_extensions:
        raise StorageError(
            f"Unsupported file type '{extension or 'unknown'}'. Allowed: {', '.join(allowed_extensions)}"
        )

    directory = _subdir_path(subdir)
    os.makedirs(directory, exist_ok=True)
    name = f"{stem or uuid.uuid4().hex}{extension}"
    destination = os.path.join(directory, name)

    limit = max_mb * 1024 * 1024
    written = 0
    try:
        with open(destination, "wb") as handle:
            while True:
                chunk = upload.file.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise StorageError(f"File is larger than the {max_mb} MB limit")
                handle.write(chunk)
    except StorageError:
        _remove(destination)
        raise
    except Exception as exc:  # noqa: BLE001
        _remove(destination)
        raise StorageError(f"Could not save upload: {exc}") from exc
    finally:
        try:
            upload.file.close()
        except Exception:  # noqa: BLE001
            pass

    if written == 0:
        _remove(destination)
        raise StorageError("The uploaded file is empty")

    logger.debug("Stored %s (%s bytes)", destination, written)
    return destination


def save_audio(upload: UploadFile, interview_public_id: str, question_id: int) -> str:
    return save_upload(
        upload,
        subdir=os.path.join("interviews", interview_public_id),
        allowed_extensions=settings.ALLOWED_AUDIO_EXTENSIONS,
        max_mb=settings.MAX_AUDIO_MB,
        stem=f"q{question_id}",
    )


def save_resume(upload: UploadFile, interview_public_id: str) -> str:
    return save_upload(
        upload,
        subdir=os.path.join("interviews", interview_public_id),
        allowed_extensions=settings.ALLOWED_RESUME_EXTENSIONS,
        max_mb=settings.MAX_RESUME_MB,
        stem="resume",
    )


def delete_interview_files(interview_public_id: str) -> None:
    directory = _subdir_path(os.path.join("interviews", interview_public_id))
    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)


def _remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
