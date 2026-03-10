"""
Shared audio utility functions.
Pure numpy/torchaudio — no scipy, no librosa.
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load any audio file → mono float32 numpy at target sample rate."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    # Mix to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    # Resample if needed
    if file_sr != sr:
        tensor = torch.from_numpy(data).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, file_sr, sr)
        data = tensor.squeeze(0).numpy()
    return data.astype(np.float32), sr


def bandpass_fft(
    audio: np.ndarray, sr: int, lo: int = 300, hi: int = 3400
) -> np.ndarray:
    """FFT bandpass filter with raised-cosine taper. Pure numpy."""
    n = len(audio)
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    tw = 50.0  # transition width Hz
    mask = np.where(
        freqs < lo - tw,
        0.0,
        np.where(
            freqs < lo,
            0.5 * (1 + np.cos(np.pi * (lo - freqs) / tw)),
            np.where(
                freqs <= hi,
                1.0,
                np.where(
                    freqs <= hi + tw,
                    0.5 * (1 + np.cos(np.pi * (freqs - hi) / tw)),
                    0.0,
                ),
            ),
        ),
    ).astype(np.float32)
    return np.fft.irfft(fft * mask, n).astype(np.float32)


def norm_loudness(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Normalize audio to target dBFS."""
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-10:
        return audio
    gain = 10 ** ((target_dbfs - 20 * np.log10(rms + 1e-10)) / 20)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def compute_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    """MFCC features via torchaudio (fallback diarization)."""
    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )
    return transform(waveform).squeeze(0).numpy()


def fmt(sec: float) -> str:
    """Seconds → HH:MM:SS.ss"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"
