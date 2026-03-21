import wave
import tempfile
import numpy as np
import webrtcvad
from pydub import AudioSegment

from .config import (
    MIN_SPEECH_MS,
    MAX_SILENCE_MS,
    ENERGY_THRESHOLD,
    MIN_WINDOW_SEC,
    MAX_GROUP_GAP_SEC,
)

def post_process_segments(segments, audio):
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
