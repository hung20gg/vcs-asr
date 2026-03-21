import json
import os
import tempfile
import wave

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse

from .config import SEGMENTS_DIR, TRANSCRIBE_BATCH_SIZE
from .audio_utils import split_audio_vad, preprocess_audio
from .model_utils import transcribe_batch, identify_speaker, extract_text, model

app = FastAPI()

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
        batch_texts = transcribe_batch(batch_paths, language)

        if batch_texts is None:
            batch_texts = []
            for item in batch:
                try:
                    res = model.transcribe(audio=item["path"], language=language)
                    text = extract_text(res)
                except Exception:
                    text = ""
                batch_texts.append(text)

        for item, text in zip(batch, batch_texts):
            speaker, score = identify_speaker(item["path"])
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
            batch_texts = transcribe_batch(batch_paths, language)

            if batch_texts is None:
                batch_texts = []
                for item in batch:
                    try:
                        res = model.transcribe(audio=item["path"], language=language)
                        text = extract_text(res)
                    except Exception:
                        text = ""
                    batch_texts.append(text)

            for item, text in zip(batch, batch_texts):
                speaker, score = identify_speaker(item["path"])
                if text.strip():
                    payload = {
                        "index": item["index"],
                        "start": item["segment"].get("start"),
                        "end": item["segment"].get("end"),
                        "speaker": speaker,
                        "confidence": score,
                        "text": text,
                    }
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
                if not dump:
                    os.remove(item["path"])

    return StreamingResponse(generate(), media_type="application/x-ndjson")