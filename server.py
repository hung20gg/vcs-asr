import json
import os
import tempfile
import wave

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from dotenv import load_dotenv
import chromadb
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
import tempfile
import webrtcvad
import torch
import numpy as np
import wave

from qwen_asr import Qwen3ASRModel

MIN_SPEECH_MS = 300
MAX_SILENCE_MS = 200
ENERGY_THRESHOLD = 500
MIN_WINDOW_SEC = 5
MAX_GROUP_GAP_SEC = 1
TRANSCRIBE_BATCH_SIZE = 2
SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), "tmp")
SPEAKER_ENROLL_DIR = os.getenv("SPEAKER_ENROLL_DIR", "test/files/users")
SPEAKER_MIN_SCORE = 0.5

load_dotenv()

app = FastAPI()

# ===== Load model once =====
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.bfloat16

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=dtype,
    device_map=device,
    max_inference_batch_size=16,
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
    signal, fs = torchaudio.load(audio_path)
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


def _extract_text(result):
    if isinstance(result, list) and result:
        result = result[0]
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, dict):
        return result.get("text", "")
    return ""


def _transcribe_batch(audio_paths, language):
    try:
        res = model.transcribe(audio=audio_paths, language=language)
    except Exception:
        return None

    if isinstance(audio_paths, (list, tuple)):
        if isinstance(res, list) and len(res) == len(audio_paths):
            return [_extract_text(item) for item in res]
        if len(audio_paths) == 1:
            return [_extract_text(res)]
        return None

    return [_extract_text(res)]


def _identify_speaker(audio_path):
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

def post_process_segments(segments, audio):
    MIN_SPEECH_MS = 300
    MAX_SILENCE_MS = 200

    # 1. remove short segments
    segments = [
        s for s in segments
        if (s["end"] - s["start"]) * 1000 >= MIN_SPEECH_MS
    ]

    # 2. merge close segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]
        gap = (seg["start"] - last["end"]) * 1000

        if gap < MAX_SILENCE_MS:
            last["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged

# ===== VAD helper =====
# def split_audio_vad(wav_path, aggressiveness=2, frame_ms=30):
#     vad = webrtcvad.Vad(aggressiveness)

#     with wave.open(wav_path, "rb") as wf:
#         sample_rate = wf.getframerate()
#         audio = wf.readframes(wf.getnframes())

#     frame_bytes = int(sample_rate * frame_ms / 1000) * 2
#     segments = []
#     current = bytearray()

#     for i in range(0, len(audio), frame_bytes):
#         frame = audio[i:i+frame_bytes]
#         if len(frame) < frame_bytes:
#             break

#         is_speech = vad.is_speech(frame, sample_rate)

#         if is_speech:
#             current.extend(frame)
#         else:
#             if len(current) > 0:
#                 segments.append(bytes(current))
#                 current = bytearray()

#     if len(current) > 0:
#         segments.append(bytes(current))

#     return segments, sample_rate
def split_audio_vad(wav_path, aggressiveness=1, frame_ms=30):
    vad = webrtcvad.Vad(aggressiveness)

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        audio = wf.readframes(wf.getnframes())

    frame_bytes = int(sample_rate * frame_ms / 1000) * 2
    frame_duration = frame_ms / 1000.0

    segments = []
    current = bytearray()
    start_time = None

    # -----------------------------
    # STEP 1: RAW VAD SEGMENTS
    # -----------------------------
    for i in range(0, len(audio), frame_bytes):
        frame = audio[i:i+frame_bytes]
        if len(frame) < frame_bytes:
            break

        t = i / (2 * sample_rate)  # seconds
        is_speech = vad.is_speech(frame, sample_rate)

        if is_speech:
            if start_time is None:
                start_time = t
            current.extend(frame)
        else:
            if len(current) > 0:
                segments.append({
                    "start": start_time,
                    "end": t,
                    "audio": bytes(current)
                })
                current = bytearray()
                start_time = None

    if len(current) > 0:
        segments.append({
            "start": start_time,
            "end": start_time + len(current)/(2*sample_rate),
            "audio": bytes(current)
        })

    # -----------------------------
    # STEP 2: POST-PROCESS (simulate 1.5)
    # -----------------------------
    

    def compute_energy(raw_bytes):
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        return np.abs(samples).mean()

    # 2.1 remove short + low-energy
    filtered = []
    for s in segments:
        duration_ms = (s["end"] - s["start"]) * 1000

        if duration_ms < MIN_SPEECH_MS:
            continue

        if compute_energy(s["audio"]) < ENERGY_THRESHOLD:
            continue

        filtered.append(s)

    # 2.2 merge close segments
    merged = []
    for seg in filtered:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]
        gap_ms = (seg["start"] - last["end"]) * 1000

        if gap_ms < MAX_SILENCE_MS:
            # merge
            last["end"] = seg["end"]
            last["audio"] += seg["audio"]
        else:
            merged.append(seg)

    # 2.3 group into minimum window unless gap is too large
    grouped = []
    current = None
    for seg in merged:
        if current is None:
            current = dict(seg)
            continue

        gap_sec = seg["start"] - current["end"]
        current_window = current["end"] - current["start"]

        if gap_sec > MAX_GROUP_GAP_SEC and current_window < MIN_WINDOW_SEC:
            grouped.append(current)
            current = dict(seg)
            continue

        if gap_sec > MAX_GROUP_GAP_SEC and current_window >= MIN_WINDOW_SEC:
            grouped.append(current)
            current = dict(seg)
            continue

        current["end"] = seg["end"]
        current["audio"] += seg["audio"]

        if (current["end"] - current["start"]) >= MIN_WINDOW_SEC:
            grouped.append(current)
            current = None

    if current is not None:
        grouped.append(current)

    return grouped, sample_rate


# ===== Convert to wav (16k mono required) =====
def preprocess_audio(input_path: str) -> str:
    """
    Convert any audio format to:
    - WAV
    - mono
    - 16kHz
    """

    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as e:
        raise ValueError(f"Unsupported or corrupted audio file: {e}")

    # Normalize format
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Optional: normalize volume
    audio = audio.normalize()

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp_wav.name, format="wav")

    return tmp_wav.name


# ===== API =====
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    aggressiveness: int = Query(1, ge=0, le=3),
    language: str = Query("Vietnamese"),
    dump: bool = Query(False)
):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    # Convert to proper format
    wav_path = preprocess_audio(input_path)

    segments, sample_rate = split_audio_vad(
        wav_path,
        aggressiveness=aggressiveness,
    )
    print(f"Detected {len(segments)} speech segments with aggressiveness {aggressiveness}")

    if not segments:
        raise HTTPException(status_code=400, detail="No speech detected in audio")

    results = []
    items = []

    for index, segment in enumerate(segments):
        if dump:
            os.makedirs(SEGMENTS_DIR, exist_ok=True)
            chunk_path = os.path.join(SEGMENTS_DIR, f"segment_{index:04d}.wav")
        else:
            chunk_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        audio_bytes = segment["audio"]

        if not isinstance(audio_bytes, (bytes, bytearray)):
            continue

        with wave.open(chunk_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        items.append({"index": index, "path": chunk_path})

    for i in range(0, len(items), TRANSCRIBE_BATCH_SIZE):
        batch = items[i:i + TRANSCRIBE_BATCH_SIZE]
        batch_paths = [item["path"] for item in batch]
        batch_texts = _transcribe_batch(batch_paths, language)

        if batch_texts is None:
            batch_texts = []
            for item in batch:
                try:
                    res = model.transcribe(audio=item["path"], language=language)
                    text = _extract_text(res)
                except Exception:
                    text = ""
                batch_texts.append(text)

        for item, text in zip(batch, batch_texts):
            speaker, score = _identify_speaker(item["path"])
            if text.strip():
                results.append({
                    "speaker": speaker,
                    "confidence": score,
                    "text": text,
                })
            if not dump:
                os.remove(item["path"])

    return {
        "segments": results,
    }


@app.post("/transcribe_stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    aggressiveness: int = Query(1, ge=0, le=3),
    language: str = Query("Vietnamese"),
    dump: bool = Query(False)
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    wav_path = preprocess_audio(input_path)

    segments, sample_rate = split_audio_vad(
        wav_path,
        aggressiveness=aggressiveness
    )
    print(f"Detected {len(segments)} speech segments with aggressiveness {aggressiveness}")

    if not segments:
        raise HTTPException(status_code=400, detail="No speech detected in audio")

    def generate():
        items = []

        for index, segment in enumerate(segments):
            if dump:
                os.makedirs(SEGMENTS_DIR, exist_ok=True)
                chunk_path = os.path.join(SEGMENTS_DIR, f"segment_{index:04d}.wav")
            else:
                chunk_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            audio_bytes = segment["audio"]

            if not isinstance(audio_bytes, (bytes, bytearray)):
                continue

            with wave.open(chunk_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

            items.append({"index": index, "segment": segment, "path": chunk_path})

        for i in range(0, len(items), TRANSCRIBE_BATCH_SIZE):
            batch = items[i:i + TRANSCRIBE_BATCH_SIZE]
            batch_paths = [item["path"] for item in batch]
            batch_texts = _transcribe_batch(batch_paths, language)

            if batch_texts is None:
                batch_texts = []
                for item in batch:
                    try:
                        res = model.transcribe(audio=item["path"], language=language)
                        text = _extract_text(res)
                    except Exception:
                        text = ""
                    batch_texts.append(text)

            for item, text in zip(batch, batch_texts):
                if text.strip():
                    payload = {
                        "index": item["index"],
                        "start": item["segment"].get("start"),
                        "end": item["segment"].get("end"),
                        "text": text,
                    }
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
                if not dump:
                    os.remove(item["path"])

    return StreamingResponse(generate(), media_type="application/x-ndjson")