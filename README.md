```bash
docker run -it \
  -p 8000:8000 \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quanghung20gg/qwen-asr:cpu \
  python -m uvicorn server:app --host 0.0.0.0 --port 8000
  
```