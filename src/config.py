import os
import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")

with open(CONFIG_PATH, "r") as f:
    config_data = yaml.safe_load(f)

MODEL_PARAMS = config_data.get("model", {})
ASR_MODEL = MODEL_PARAMS.get("asr_model", "QwenLM/Qwen3-ASR-0.6B")
SPEAKER_MODEL = MODEL_PARAMS.get("speaker_model", "speechbrain/spkrec-ecapa-voxceleb")

AUDIO_PARAMS = config_data.get("audio", {})
MIN_SPEECH_MS = AUDIO_PARAMS.get("min_speech_ms", 300)
MAX_SILENCE_MS = AUDIO_PARAMS.get("max_silence_ms", 200)
ENERGY_THRESHOLD = AUDIO_PARAMS.get("energy_threshold", 500)
MIN_WINDOW_SEC = AUDIO_PARAMS.get("min_window_sec", 5)
MAX_GROUP_GAP_SEC = AUDIO_PARAMS.get("max_group_gap_sec", 1)

TRANSCRIBE_BATCH_SIZE = config_data.get("transcribe", {}).get("batch_size", 2)

SPEAKER_PARAMS = config_data.get("speaker", {})
SPEAKER_ENROLL_DIR = os.getenv("SPEAKER_ENROLL_DIR", SPEAKER_PARAMS.get("enroll_dir", "test/files/users"))
SPEAKER_MIN_SCORE = SPEAKER_PARAMS.get("min_score", 0.5)

SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), "tmp")
