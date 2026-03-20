import tempfile
import numpy as np
from pydub import AudioSegment
import noisereduce as nr


def preprocess_audio(input_path: str, noise_level: str = "off") -> str:
    audio = AudioSegment.from_file(input_path)

    # base normalization
    audio = audio.set_channels(1).set_frame_rate(16000)

    # always safe
    audio = audio.high_pass_filter(80)
    audio = audio.normalize()

    # convert to numpy for advanced processing
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate

    # ===== noise control =====
    if noise_level == "low":
        samples = nr.reduce_noise(y=samples, sr=sr, prop_decrease=0.3)

    elif noise_level == "medium":
        samples = nr.reduce_noise(y=samples, sr=sr, prop_decrease=0.6)

    elif noise_level == "high":
        samples = nr.reduce_noise(y=samples, sr=sr, prop_decrease=0.9)

    # back to AudioSegment
    processed_audio = AudioSegment(
        samples.astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    # optional silence trim (only for higher levels)
    if noise_level in ["medium", "high"]:
        processed_audio = processed_audio.strip_silence(silence_len=500)

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    processed_audio.export(tmp_wav.name, format="wav")

    return tmp_wav.name