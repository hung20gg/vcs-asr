#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR" && pwd)"
AUDIO_PATH="${AUDIO_PATH:-$WORKSPACE_DIR/files/test_audio.m4a}"
WAV_PATH="$(mktemp --suffix=.wav)"

ffmpeg -y -i "$AUDIO_PATH" -ar 16000 -ac 1 -f wav "$WAV_PATH" >/dev/null 2>&1

PAYLOAD_FILE="$(mktemp)"
python3 - <<'PY' "$WAV_PATH" "$PAYLOAD_FILE"
import base64
import json
import sys

audio_path = sys.argv[1]
payload_path = sys.argv[2]

with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    },
                }
            ],
        }
    ]
}

with open(payload_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
    --data-binary "@$PAYLOAD_FILE"

rm -f "$PAYLOAD_FILE" "$WAV_PATH"