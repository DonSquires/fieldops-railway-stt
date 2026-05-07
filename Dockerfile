# FieldOps Manager — Railway STT Service (Faster-Whisper CPU)
# No CUDA required — runs on any Railway instance tier.
# Model: distil-small.en (fast on CPU, ~1s per 5-second clip)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ffmpeg is required by faster-whisper for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model at build time so cold starts are fast.
# Using cpu/int8 — Railway instances have no GPU.
ARG WHISPER_MODEL=distil-small.en
ENV WHISPER_MODEL=${WHISPER_MODEL} \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE=int8 \
    MAX_AUDIO_SECONDS=120 \
    MAX_AUDIO_B64_MB=25 \
    PORT=8080

RUN python3 -c "\
from faster_whisper import WhisperModel; \
WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8'); \
print('Model pre-cached:', '${WHISPER_MODEL}');"

COPY server.py .

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
