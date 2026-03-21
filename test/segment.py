from pyannote.audio import Pipeline
import pyannote.audio
from pydub import AudioSegment
import os
from dotenv import load_dotenv

print(f"Pyannote version: {pyannote.audio.__version__}")
load_dotenv()  # Load environment variables from .env file

INPUT_AUDIO = "tmp/segment_0033_multi.wav"
OUTPUT_DIR = "tmp"

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-community-1",
  token=os.getenv("HF_TOKEN"),
)

output = pipeline(INPUT_AUDIO)
audio = AudioSegment.from_file(INPUT_AUDIO)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for index, (turn, speaker) in enumerate(output.speaker_diarization):
  start_ms = int(turn.start * 1000)
  end_ms = int(turn.end * 1000)
  if end_ms <= start_ms:
    continue

  segment = audio[start_ms:end_ms]
  filename = f"{speaker}_{index:04d}_{start_ms}ms_{end_ms}ms.wav"
  segment.export(os.path.join(OUTPUT_DIR, filename), format="wav")
  print(f"Wrote {filename}")