# FieldOps Railway STT

Standalone Faster-Whisper CPU STT service for FieldOps Manager.

## Run locally

```bash
docker build -t fieldops-stt .
docker run --rm -p 8080:8080 -e STT_API_KEY=change-me fieldops-stt
```

Health check:

```bash
curl http://localhost:8080/health
```

Deployed on Railway from the `main` branch.
