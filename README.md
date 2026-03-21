# Qwen ASR Server

FastAPI-powered Voice Activity Detection and target-speaker Transcription service using Qwen3-ASR.

## Configuration
Application logic settings (like speech duration triggers, gaps, and confidence thresholds) now live in `src/config.yml`. Verify or edit these parameters as necessary before booting up the server.

## Build the env

CPU ONLY
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.qwen-asr \
  -t quanghung20gg/qwen-asr:cpu \
  --push .
```

## Run the script

```bash
docker run -it --rm --name qwen-asr \
  -p 8000:8000 \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quanghung20gg/qwen-asr:cpu \
  python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
```

## TODOs
1. **Manage Processing State:** Create a service/database to manage files, versions, and log currently running tasks.
2. **Audio Similarity Reference:** Create a service/database to effectively manage users' audio samples for similarity comparisons.