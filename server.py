import json
import os
import tempfile
import wave

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
import tempfile
import webrtcvad
import torch
import numpy as np
import wave

from qwen_asr import Qwen3ASRModel

MIN_SPEECH_MS = 300
MAX_SILENCE_MS = 200
ENERGY_THRESHOLD = 500

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

    return merged, sample_rate


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
    language: str = Query("Vietnamese")
):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    # Convert to proper format
    wav_path = preprocess_audio(input_path)

    # ✅ Use param here
    segments, sample_rate = split_audio_vad(
        wav_path,
        aggressiveness=aggressiveness
    )
    print(f"Detected {len(segments)} speech segments with aggressiveness {aggressiveness}")

    if not segments:
        raise HTTPException(status_code=400, detail="No speech detected in audio")

    results = []

    for segment in segments:
        chunk_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        audio_bytes = segment["audio"]

        if not isinstance(audio_bytes, (bytes, bytearray)):
            continue  # safety check

        with wave.open(chunk_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)

        try:
            res = model.transcribe(audio=chunk_path, language=language)
            text = res[0].text
        except Exception:
            text = ""

        if not text.strip():
            continue

        results.append(text)
        os.remove(chunk_path)

    full_text = " ".join(results)

    return {
        "aggressiveness": aggressiveness,
        "segments": results,
        "full_text": full_text
    }


@app.post("/transcribe_stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    aggressiveness: int = Query(1, ge=0, le=3),
    language: str = Query("Vietnamese")
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
        for index, segment in enumerate(segments):
            chunk_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            audio_bytes = segment["audio"]

            if not isinstance(audio_bytes, (bytes, bytearray)):
                continue

            with wave.open(chunk_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

            try:
                res = model.transcribe(audio=chunk_path, language=language)
                text = res[0].text
            except Exception:
                text = ""

            if not text.strip():
                continue

            payload = {
                "index": index,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
            }
            yield json.dumps(payload, ensure_ascii=False) + "\n"
            os.remove(chunk_path)

    return StreamingResponse(generate(), media_type="application/x-ndjson")