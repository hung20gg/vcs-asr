#!/usr/bin/env bash
set -euo pipefail

START_SECONDS=$SECONDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR" && pwd)"

SERVER_URL="${SERVER_URL:-http://localhost:8000}"
AUDIO_PATH="${AUDIO_PATH:-$WORKSPACE_DIR/files/long_audio/demo_agent.m4a}"
AGGRESSIVENESS="${AGGRESSIVENESS:-2}"
LANGUAGE="${LANGUAGE:-Vietnamese}"
OUT_FILE="${OUT_FILE:-$WORKSPACE_DIR/log/transcribe_stream_2.jsonl}"


if [[ ! -f "$AUDIO_PATH" ]]; then
	echo "Audio file not found: $AUDIO_PATH" >&2
	exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")"

curl -sS --no-buffer -X POST \
	"$SERVER_URL/transcribe_stream?aggressiveness=$AGGRESSIVENESS&dump=true" \
	-F "file=@${AUDIO_PATH}" \
	-o "$OUT_FILE"

ELAPSED_SECONDS=$((SECONDS - START_SECONDS))
echo "Saved stream output to: $OUT_FILE"
echo "Duration: ${ELAPSED_SECONDS}s"


