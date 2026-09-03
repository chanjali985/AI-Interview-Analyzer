"""Speech-to-text with a local-first, pluggable backend.

Default is faster-whisper running on CPU inside the container: no API key, no
per-minute cost, and it handles the webm/opus blobs the browser recorder
produces. Hosted Whisper and the free Google endpoint are available as
alternatives via TRANSCRIPTION_PROVIDER.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when audio could not be turned into text."""


@dataclass
class Transcript:
    text: str
    duration_seconds: float | None = None
    language: str | None = None
    provider: str = ""


# ------------------------------------------------------------------- utilities
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_duration(audio_path: str) -> float | None:
    """Best-effort audio duration in seconds via ffprobe."""
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return round(float(result.stdout.strip()), 2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ffprobe failed for %s: %s", audio_path, exc)
    return None


def convert_to_wav(audio_path: str) -> str:
    """Transcode any input to 16 kHz mono WAV. Returns a temp file path."""
    if not ffmpeg_available():
        raise TranscriptionError("ffmpeg is required to convert audio but was not found on PATH")

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    handle.close()
    command = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ac", "1", "-ar", "16000", "-vn",
        "-loglevel", "error",
        handle.name,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        os.unlink(handle.name)
        raise TranscriptionError(f"ffmpeg conversion failed: {result.stderr[:300]}")
    return handle.name


# ------------------------------------------------------------- faster-whisper
@lru_cache
def _load_whisper():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TranscriptionError(
            "faster-whisper is not installed. Run `pip install faster-whisper` "
            "or set TRANSCRIPTION_PROVIDER to another backend."
        ) from exc

    logger.info("Loading faster-whisper model '%s' on %s", settings.WHISPER_MODEL, settings.WHISPER_DEVICE)
    return WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )


def _transcribe_faster_whisper(audio_path: str) -> Transcript:
    model = _load_whisper()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        language=settings.WHISPER_LANGUAGE or None,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return Transcript(
        text=text,
        duration_seconds=round(getattr(info, "duration", 0) or 0, 2) or None,
        language=getattr(info, "language", None),
        provider="faster_whisper",
    )


# ------------------------------------------------------------- OpenAI Whisper
def _transcribe_openai(audio_path: str) -> Transcript:
    if not settings.OPENAI_API_KEY:
        raise TranscriptionError("TRANSCRIPTION_PROVIDER=openai but OPENAI_API_KEY is not set")

    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/audio/transcriptions"
    with open(audio_path, "rb") as handle:
        files = {"file": (os.path.basename(audio_path), handle, "application/octet-stream")}
        data = {"model": "whisper-1"}
        if settings.WHISPER_LANGUAGE:
            data["language"] = settings.WHISPER_LANGUAGE
        with httpx.Client(timeout=180) as client:
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                files=files,
                data=data,
            )
    if response.status_code >= 400:
        raise TranscriptionError(f"Whisper API error {response.status_code}: {response.text[:300]}")
    return Transcript(
        text=response.json().get("text", "").strip(),
        duration_seconds=probe_duration(audio_path),
        provider="openai",
    )


# ------------------------------------------------------- Google (best effort)
def _transcribe_google(audio_path: str) -> Transcript:
    try:
        import speech_recognition as sr
    except ImportError as exc:  # pragma: no cover
        raise TranscriptionError("SpeechRecognition is not installed") from exc

    wav_path = audio_path
    temp_path = None
    if not audio_path.lower().endswith(".wav"):
        temp_path = convert_to_wav(audio_path)
        wav_path = temp_path

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return Transcript(text=text.strip(), duration_seconds=probe_duration(audio_path), provider="google")
    except sr.UnknownValueError as exc:
        raise TranscriptionError("Speech was not intelligible in the recording") from exc
    except sr.RequestError as exc:
        raise TranscriptionError(f"Google speech endpoint unavailable: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ----------------------------------------------------------------- public API
def transcribe(audio_path: str) -> Transcript:
    """Transcribe an audio file using the configured provider."""
    if not os.path.exists(audio_path):
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    if os.path.getsize(audio_path) == 0:
        raise TranscriptionError("Audio file is empty")

    provider = settings.TRANSCRIPTION_PROVIDER
    logger.info("Transcribing %s with %s", os.path.basename(audio_path), provider)

    if provider == "faster_whisper":
        result = _transcribe_faster_whisper(audio_path)
    elif provider == "openai":
        result = _transcribe_openai(audio_path)
    elif provider == "google":
        result = _transcribe_google(audio_path)
    else:
        raise TranscriptionError("TRANSCRIPTION_PROVIDER=none — transcription is disabled")

    if not result.text:
        raise TranscriptionError(
            "No speech was detected in the recording. Ask the candidate to re-record in a quieter place."
        )
    if result.duration_seconds is None:
        result.duration_seconds = probe_duration(audio_path)
    return result


def transcription_ready() -> bool:
    """Cheap readiness probe used by /health."""
    provider = settings.TRANSCRIPTION_PROVIDER
    if provider == "faster_whisper":
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False
    if provider == "openai":
        return bool(settings.OPENAI_API_KEY)
    if provider == "google":
        try:
            import speech_recognition  # noqa: F401
            return ffmpeg_available()
        except ImportError:
            return False
    return False
