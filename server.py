"""
FieldOps Manager — Railway STT Service
Faster-Whisper CPU transcription endpoint.

Accepts the same request shape as the RunPod serverless handler so that
speech-router can point STT_URL here with no other code changes.

POST /transcribe
  Body: { "audio_base64": "...", "language": "en" }
  Returns: { "transcript": "...", "language": "en", "duration_s": 3.1, "processing_ms": 280 }

GET /health
  Returns: { "ok": true, "model": "distil-small.en", ... }
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("railway-stt")
logging.basicConfig(level=logging.INFO)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "distil-small.en")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "120"))
MAX_AUDIO_B64_MB = float(os.environ.get("MAX_AUDIO_B64_MB", "25"))

# Optional bearer token auth — set STT_API_KEY on Railway to enable
_STT_API_KEY = os.environ.get("STT_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
logger.info("Loading Faster-Whisper model: %s on %s/%s", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
try:
    from faster_whisper import WhisperModel
    _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    logger.info("Model loaded successfully: %s", WHISPER_MODEL)
except Exception as exc:
    logger.error("FATAL: Failed to load model: %s", exc)
    _model = None

app = FastAPI(title="FieldOps Railway STT", version="1.0.0")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TranscribeRequest(BaseModel):
    # speech-router sends "audio_base64"; legacy RunPod callers send "audio_b64"
    audio_base64: str | None = Field(default=None)
    audio_b64: str | None = Field(default=None)
    language: str = Field(default="en")
    org_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)

    @field_validator("audio_base64", "audio_b64", mode="before")
    @classmethod
    def _allow_none(cls, v: Any) -> Any:
        return v


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    duration_s: float
    processing_ms: int


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _check_auth(authorization: str | None) -> None:
    if not _STT_API_KEY:
        return  # no key configured — open (dev mode)
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token != _STT_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": _model is not None,
        "service": "railway-stt",
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute": WHISPER_COMPUTE,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    payload: TranscribeRequest,
    authorization: str | None = Header(default=None),
) -> TranscribeResponse:
    _check_auth(authorization)
    return await _do_transcribe(payload)


# The speech-router posts directly to STT_URL with no path appended, so expose
# /transcribe AND handle a root POST for callers that send to the bare URL.
@app.post("/", response_model=TranscribeResponse, include_in_schema=False)
async def transcribe_root(
    payload: TranscribeRequest,
    authorization: str | None = Header(default=None),
) -> TranscribeResponse:
    _check_auth(authorization)
    return await _do_transcribe(payload)


async def _do_transcribe(payload: TranscribeRequest) -> TranscribeResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_b64 = (payload.audio_base64 or payload.audio_b64 or "").strip()
    if not audio_b64:
        raise HTTPException(status_code=422, detail="audio_base64 is required")

    b64_size_mb = len(audio_b64) / (1024 * 1024)
    if b64_size_mb > MAX_AUDIO_B64_MB:
        raise HTTPException(
            status_code=413,
            detail=f"audio_base64 payload too large ({b64_size_mb:.1f} MB, max {MAX_AUDIO_B64_MB} MB)",
        )

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid base64: {exc}") from exc

    language = (payload.language or "en").strip() or "en"

    t_start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = _model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        transcript_parts = [seg.text for seg in segments]
    except Exception as exc:
        logger.error("Transcription error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    transcript = " ".join(transcript_parts).strip()
    duration_s = round(getattr(info, "duration", 0.0), 2)
    processing_ms = int((time.perf_counter() - t_start) * 1000)

    logger.info(
        "Transcribed %.1fs audio in %dms | language=%s | model=%s",
        duration_s, processing_ms, language, WHISPER_MODEL,
    )

    return TranscribeResponse(
        transcript=transcript,
        language=language,
        duration_s=duration_s,
        processing_ms=processing_ms,
    )
