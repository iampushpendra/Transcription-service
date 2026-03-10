"""
Speaker embedding backends for advanced diarization.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import PipelineConfig
from ..utils import load_audio

_ECAPA_ENCODER = None
_PYANNOTE_EMBED_INFERENCE = None


def _resolve_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(x) + 1e-9)
    return (x / denom).astype(np.float32)


def _load_pretrained(loader, model_id: str, token: str | None):
    if token:
        try:
            return loader(model_id, token=token)
        except TypeError:
            return loader(model_id, use_auth_token=token)
    return loader(model_id)


def _generate_with_ecapa(
    path: str,
    segments: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    global _ECAPA_ENCODER

    from speechbrain.inference.speaker import EncoderClassifier

    if _ECAPA_ENCODER is None:
        auth = cfg.hf_token if cfg.hf_token else False
        _ECAPA_ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": _resolve_device_str()},
            use_auth_token=auth,
        )

    audio, sr = load_audio(path, sr=cfg.sample_rate)
    min_samples = int(max(0.25, cfg.advanced_min_segment_s) * sr)

    out: list[dict] = []
    for seg in segments:
        start = int(float(seg["start"]) * sr)
        end = int(float(seg["end"]) * sr)
        if end <= start:
            continue

        chunk = audio[start:end]
        if chunk.size < min_samples:
            pad = min_samples - chunk.size
            chunk = np.pad(chunk, (0, pad), mode="edge")

        wave = torch.from_numpy(chunk).float().unsqueeze(0)
        with torch.no_grad():
            emb = _ECAPA_ENCODER.encode_batch(wave)
        emb_np = emb.squeeze().detach().cpu().numpy().astype(np.float32).reshape(-1)
        seg_out = seg.copy()
        seg_out["embedding"] = _l2_normalize(emb_np)
        out.append(seg_out)

    if not out:
        raise RuntimeError("ECAPA produced no embeddings")
    return out


def _generate_with_pyannote(
    path: str,
    segments: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    global _PYANNOTE_EMBED_INFERENCE

    from pyannote.audio import Inference, Model
    from pyannote.core import Segment

    if _PYANNOTE_EMBED_INFERENCE is None:
        model = _load_pretrained(Model.from_pretrained, "pyannote/embedding", cfg.hf_token)
        device = torch.device(_resolve_device_str())
        model.to(device)
        _PYANNOTE_EMBED_INFERENCE = Inference(model, window="whole")

    out: list[dict] = []
    for seg in segments:
        segment = Segment(float(seg["start"]), float(seg["end"]))
        if segment.end <= segment.start:
            continue
        emb = _PYANNOTE_EMBED_INFERENCE.crop(path, segment)
        if hasattr(emb, "detach"):
            emb_np = emb.detach().cpu().numpy()
        else:
            emb_np = np.asarray(emb)
        emb_np = emb_np.astype(np.float32).reshape(-1)
        if emb_np.size == 0:
            continue

        seg_out = seg.copy()
        seg_out["embedding"] = _l2_normalize(emb_np)
        out.append(seg_out)

    if not out:
        raise RuntimeError("pyannote embedding produced no embeddings")
    return out


def generate_embeddings(
    path: str,
    segments: list[dict],
    cfg: PipelineConfig,
) -> tuple[list[dict], str]:
    """
    Preferred backend: SpeechBrain ECAPA
    Alternative: pyannote embedding
    """
    errors: list[str] = []

    try:
        out = _generate_with_ecapa(path, segments, cfg)
        print(f"🧠 Embeddings: speechbrain/spkrec-ecapa-voxceleb ({len(out)} segments)")
        return out, "ecapa"
    except Exception as exc:
        errors.append(f"ECAPA failed: {exc}")

    try:
        out = _generate_with_pyannote(path, segments, cfg)
        print(f"🧠 Embeddings: pyannote/embedding ({len(out)} segments)")
        return out, "pyannote_embedding"
    except Exception as exc:
        errors.append(f"pyannote embedding failed: {exc}")

    raise RuntimeError(" | ".join(errors))
