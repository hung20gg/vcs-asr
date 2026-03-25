import os
import torch
import torchaudio
import numpy as np
import chromadb

# HOTFIX for speechbrain + torchaudio 2.1+
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []

from speechbrain.inference.speaker import EncoderClassifier

from qwen_asr import Qwen3ASRModel

from .config import (
    SPEAKER_ENROLL_DIR,
    SPEAKER_MIN_SCORE,
    ASR_MODEL,
    SPEAKER_MODEL,
    MODEL_SERVING,
    BATCH_SIZE,
)

# ===== Load model once =====
_cuda_available = torch.cuda.is_available()
_cuda_device_count = torch.cuda.device_count() if _cuda_available else 0

device = "cuda:0" if _cuda_available else "cpu"
dtype = torch.bfloat16 if _cuda_available else torch.bfloat16

def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def _as_text(value, default):
    if value is None:
        return default
    return str(value).strip() or default

def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

serving = _as_text(os.getenv("QWEN_ASR_SERVING"), MODEL_SERVING).lower()
use_vllm = serving == "vllm" or _as_bool(os.getenv("QWEN_ASR_USE_VLLM"))

if use_vllm:
    vllm_model = os.getenv("QWEN_ASR_VLLM_MODEL", "Qwen/Qwen3-ASR-1.7B")
    gpu_mem_util = _as_float(os.getenv("QWEN_ASR_VLLM_GPU_UTIL"), 0.7)
    max_batch_size = _as_int(os.getenv("QWEN_ASR_VLLM_BATCH_SIZE"), 128)
    max_new_tokens = _as_int(os.getenv("QWEN_ASR_VLLM_MAX_NEW_TOKENS"), 4096)

    aligner_device_map = "auto" if _cuda_device_count > 1 else device

    llm_kwargs = {}
    if _cuda_device_count > 1:
        llm_kwargs["tensor_parallel_size"] = _cuda_device_count

    model = Qwen3ASRModel.LLM(
        model=vllm_model,
        gpu_memory_utilization=gpu_mem_util,
        max_inference_batch_size=max_batch_size,
        max_new_tokens=max_new_tokens,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map=aligner_device_map,
        ),
        **llm_kwargs,
    )
else:
    device_map = "auto" if _cuda_device_count > 1 else device
    model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=BATCH_SIZE,
    )

speaker_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)

speaker_client = chromadb.Client()
speaker_collection = speaker_client.get_or_create_collection(
    name="voice_auth_db",
    metadata={"hnsw:space": "cosine"},
)


def _get_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path, backend="soundfile" if hasattr(torchaudio, "info") else None)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)

    with torch.no_grad():
        embeddings = speaker_classifier.encode_batch(signal)
    return embeddings.squeeze().cpu().numpy()


def _enroll_speakers(base_path):
    if not os.path.isdir(base_path):
        return

    user_folders = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
    ]

    for user_id in user_folders:
        user_path = os.path.join(base_path, user_id)
        audio_files = [f for f in os.listdir(user_path) if f.endswith(".wav")]
        if not audio_files:
            continue

        user_embeddings = []
        for file_name in audio_files:
            file_path = os.path.join(user_path, file_name)
            user_embeddings.append(_get_embedding(file_path))

        final_embedding = np.mean(user_embeddings, axis=0).tolist()
        speaker_collection.add(
            embeddings=[final_embedding],
            ids=[user_id],
            metadatas=[{"name": user_id, "sample_count": len(audio_files)}],
        )


_enroll_speakers(SPEAKER_ENROLL_DIR)


def extract_text(result):
    if isinstance(result, list) and result:
        result = result[0]
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, dict):
        return result.get("text", "")
    return ""


def transcribe_batch(audio_paths, language):
    try:
        res = model.transcribe(audio=audio_paths, language=language)
    except Exception:
        return None

    if isinstance(audio_paths, (list, tuple)):
        if isinstance(res, list) and len(res) == len(audio_paths):
            return [extract_text(item) for item in res]
        if len(audio_paths) == 1:
            return [extract_text(res)]
        return None

    return [extract_text(res)]


def identify_speaker(audio_path):
    if speaker_collection.count() == 0:
        return "unk", 0.0

    test_vec = _get_embedding(audio_path).tolist()
    results = speaker_collection.query(query_embeddings=[test_vec], n_results=1)

    if not results.get("ids") or not results["ids"][0]:
        return "unk", 0.0

    dist = results["distances"][0][0]
    speaker = results["ids"][0][0]
    score = 1.0 - dist

    if score < SPEAKER_MIN_SCORE:
        return "unk", float(score)

    return speaker, float(score)
