"""
Voice Activity Detection using Silero VAD.
"""

import torch

from .config import PipelineConfig


def run_vad(
    path: str,
    cfg: PipelineConfig | None = None,
) -> list[dict]:
    """
    Detect speech segments using Silero VAD.

    Returns:
        List of {'start': float, 'end': float} segments in seconds.
    """
    if cfg is None:
        cfg = PipelineConfig()

    sr = cfg.sample_rate
    print("🔍 Running Silero VAD...")

    model, utils = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    read_audio = utils[2]

    wav = read_audio(path, sampling_rate=sr)
    total_dur = len(wav) / sr

    segments = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        min_silence_duration_ms=cfg.vad_silence_ms,
        speech_pad_ms=cfg.vad_speech_pad_ms,
        return_seconds=True,
        threshold=cfg.vad_threshold,
    )

    speech_dur = sum(s["end"] - s["start"] for s in segments)
    silence_pct = (1 - speech_dur / total_dur) * 100
    print(
        f"   {len(segments)} segments | {speech_dur:.1f}s speech "
        f"/ {total_dur:.1f}s total | {silence_pct:.1f}% silence"
    )
    print("✅ VAD done")
    return segments
