#!/usr/bin/env bash
set -euo pipefail

START_SECONDS=$SECONDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR" && pwd)"

SERVER_URL="${SERVER_URL:-http://localhost:8000}"
AUDIO_PATH="${AUDIO_PATH:-$WORKSPACE_DIR/files/test_audio.m4a}"
AGGRESSIVENESS="${AGGRESSIVENESS:-2}"
LANGUAGE="${LANGUAGE:-Vietnamese}"

if [[ ! -f "$AUDIO_PATH" ]]; then
	echo "Audio file not found: $AUDIO_PATH" >&2
	exit 1
fi

curl -sS -X POST \
	"$SERVER_URL/transcribe_stream?aggressiveness=$AGGRESSIVENESS&dump=true" \
	-F "file=@${AUDIO_PATH}"

ELAPSED_SECONDS=$((SECONDS - START_SECONDS))
echo "Duration: ${ELAPSED_SECONDS}s"


