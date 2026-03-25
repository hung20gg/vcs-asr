#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./install.sh [--cpu|--gpu|--vllm]
#   ./install.sh [cpu|gpu|vllm]
# Default mode is cpu when no argument is provided.

MODE="cpu"

if [[ $# -gt 1 ]]; then
	echo "Too many arguments."
	echo "Usage: $0 [--cpu|--gpu|--vllm]"
	echo "   or: $0 [cpu|gpu|vllm]"
	exit 1
fi

if [[ $# -eq 1 ]]; then
	case "$1" in
		--cpu|cpu)
			MODE="cpu"
			;;
		--gpu|gpu)
			MODE="gpu"
			;;
		--vllm|vllm)
			MODE="vllm"
			;;
		*)
			echo "Invalid mode: $1"
			echo "Usage: $0 [--cpu|--gpu|--vllm]"
			echo "   or: $0 [cpu|gpu|vllm]"
			exit 1
			;;
	esac
fi

if command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
	PYTHON_BIN="python"
else
	echo "Python is not installed. Please install Python 3 first."
	exit 1
fi

PIP_CMD=("$PYTHON_BIN" -m pip)

echo "[1/4] Installing Ubuntu system dependencies..."
if ! command -v apt-get >/dev/null 2>&1; then
	echo "This installer expects Ubuntu/Debian (apt-get not found)."
	exit 1
fi

SUDO=""
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
	SUDO="sudo"
fi

$SUDO apt-get update
$SUDO apt-get install -y \
	python3 \
	python3-pip \
	python3-venv \
	ffmpeg \
	libsndfile1 \
	git \
	build-essential

echo "[2/4] Upgrading pip toolchain..."
"${PIP_CMD[@]}" install -U pip setuptools wheel

echo "[3/4] Installing PyTorch and qwen-asr for mode: $MODE"
if [[ "$MODE" == "cpu" ]]; then
	"${PIP_CMD[@]}" install torch==2.8.0 torchaudio==2.8.0 \
		--extra-index-url https://download.pytorch.org/whl/cpu
	"${PIP_CMD[@]}" install --no-cache-dir qwen-asr
elif [[ "$MODE" == "gpu" ]]; then
	"${PIP_CMD[@]}" install torch==2.8.0 torchaudio==2.8.0 \
		--extra-index-url https://download.pytorch.org/whl/cu121
	"${PIP_CMD[@]}" install --no-cache-dir qwen-asr
elif [[ "$MODE" == "vllm" ]]; then
	"${PIP_CMD[@]}" install -U "qwen-asr[vllm]"
	"${PIP_CMD[@]}" install -U --ignore-installed qwen-asr
fi

echo "[4/4] Installing Python requirements..."
"${PIP_CMD[@]}" install --no-cache-dir -r requirements.txt

echo "Installation complete for mode: $MODE"