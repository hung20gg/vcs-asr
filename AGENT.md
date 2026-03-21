# Agent Instructions for ASR Server

This repository contains an Automatic Speech Recognition (ASR) server powered by Qwen3-ASR, WebRTCVAD, and FastAPI.

## Code Architecture

The codebase was recently refactored to prioritize modularity:
- `src/server.py`: The main FastAPI server application exposing the `/transcribe` and `/transcribe_stream` endpoints.
- `src/config.yml`: Central configuration file for audio parameters, VAD thresholds, transcript batch sizes, and related settings.
- `src/config.py`: Parses `config.yml` and exposes variables across the application, handling runtime environment variable injections (like `SPEAKER_ENROLL_DIR`).
- `src/audio_utils.py`: Houses VAD operations, raw audio preprocessing pipelines, and segmentation formatting.
- `src/model_utils.py`: Manages device selection, PyTorch / HuggingFace model loading (`Qwen3-ASR-0.6B`, `speechbrain/spkrec-ecapa-voxceleb`), speaker embedding processes, transcript post-processing, and ChromaDB caching logic.

## Developing and Maintaining
When editing parameters: Check `src/config.yml` first before diving into Python classes.
When modifying the AI inference tools: Refer to `src/model_utils.py`.
When troubleshooting segmenting issues: Refer to `src/audio_utils.py` and tweak the minimum VAD thresholds stored in `src/config.yml`.

## Future Development / TODOs
We have identified the following key areas for future improvement:
1. **Asset & Log Management**: Create a service/database to manage processed files, handle versioning, and log the current running tasks.
2. **Audio Similarity Service**: Create a service/database specifically tailored to management of users' audio samples for similarity matching, expanding on our ChromaDB embedding caching approach.
