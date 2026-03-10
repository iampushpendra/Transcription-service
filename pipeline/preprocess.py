"""
Audio preprocessing: convert → 16kHz mono → bandpass → normalize.
"""

import os

import soundfile as sf
from pydub import AudioSegment

from .config import PipelineConfig
from .utils import bandpass_fft, load_audio, norm_loudness


def preprocess(
    path: str,
    output: str = "preprocessed.wav",
    cfg: PipelineConfig | None = None,
) -> tuple[str, float]:
    """
    Full preprocessing chain.

    Returns:
        (output_path, duration_seconds)
    """
    if cfg is None:
        cfg = PipelineConfig()

    sr = cfg.sample_rate
    print("🔧 Preprocessing...")

    # Step 1: Convert to WAV via pydub (handles any format via ffmpeg)
    seg = AudioSegment.from_file(path)
    seg = seg.set_channels(1).set_frame_rate(sr).set_sample_width(2)
    tmp = "_tmp_convert.wav"
    seg.export(tmp, format="wav")

    # Step 2: Load as numpy
    audio, _ = load_audio(tmp, sr=sr)
    duration = len(audio) / sr
    print(f"   Duration: {duration:.1f}s ({duration / 60:.1f} min)")

    # Step 3: Bandpass filter
    print(f"   Bandpass {cfg.bandpass_lo}–{cfg.bandpass_hi} Hz")
    audio = bandpass_fft(audio, sr, cfg.bandpass_lo, cfg.bandpass_hi)

    # Step 4: Loudness normalization
    print(f"   Normalizing to {cfg.target_dbfs} dBFS")
    audio = norm_loudness(audio, cfg.target_dbfs)

    # Save
    sf.write(output, audio, sr)
    if os.path.exists(tmp):
        os.remove(tmp)

    print(f"✅ Preprocessed → {output}")
    return output, duration
