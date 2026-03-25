# Qwen ASR Server

FastAPI-powered Voice Activity Detection and target-speaker Transcription service using Qwen3-ASR.

## Configuration
Application logic settings (like speech duration triggers, gaps, and confidence thresholds) now live in `src/config.yml`.
Set `model.serving` to `transformers` or `vllm` before booting up the server.

## Build the env

CPU ONLY
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.qwen-asr \
  -t quanghung20gg/qwen-asr:cpu \
  --push .
```

GPU (transformers)
```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.qwen-asr-gpu \
  -t quanghung20gg/qwen-asr:gpu \
  --push .
```

GPU (vLLM)
```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.qwen-asr-vllm \
  -t quanghung20gg/qwen-asr:vllm-gpu \
  --push .
```

## Run the script

```bash
docker run -it --rm --name qwen-asr \
  -p 8000:8000 \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quanghung20gg/qwen-asr:cpu \
  python -m src.start --host 0.0.0.0 --port 8000
```

### GPU (transformers)

Set `model.serving: "transformers"` in `src/config.yml`:

```bash
docker run -it --rm --name qwen-asr-gpu \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quanghung20gg/qwen-asr:gpu \
  python3 -m src.start --host 0.0.0.0 --port 8000 --gpus 0,1
```

### vLLM backend

Use the GPU image and set `model.serving: "vllm"` in `src/config.yml`:

```bash
docker run -it --rm --name qwen-asr-vllm \
  --gpus all \
  -p 8001:8001 \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quanghung20gg/qwen-asr:vllm-gpu \
  python -m src.start --host 0.0.0.0 --port 8001 --gpus 0,1

You can limit GPU visibility with `--gpus`. Examples: `--gpus 0` (single GPU) or `--gpus 0,1` (multi-GPU).
```

## Development mode:

Install dependencies and run the server with hot-reloading for development:

```bash
./install.sh
```

Start the server in development mode:

```bash
uvicorn src.start:app --host 0.0.0.0 --port 8000
```

## TODOs
1. **Manage Processing State:** Create a service/database to manage files, versions, and log currently running tasks.
2. **Audio Similarity Reference:** Create a service/database to effectively manage users' audio samples for similarity comparisons.