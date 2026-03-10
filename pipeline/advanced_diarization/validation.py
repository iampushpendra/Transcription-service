"""
Validation metrics for comparing legacy vs advanced diarization outputs.

All accuracy metrics here are proxy metrics unless explicit ground-truth labels exist.
"""

from __future__ import annotations

import numpy as np

from ..config import PipelineConfig
from ..utils import compute_mfcc, load_audio


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def _boundary_times(segments: list[dict]) -> list[float]:
    if len(segments) < 2:
        return []
    return [float(s["end"]) for s in segments[:-1]]


def _switch_times(segments: list[dict]) -> list[float]:
    switches = []
    for i in range(len(segments) - 1):
        if segments[i]["speaker"] != segments[i + 1]["speaker"]:
            switches.append(float(segments[i]["end"]))
    return switches


def speaker_switch_accuracy_proxy(
    segments: list[dict],
    change_points: list[float],
    tolerance_s: float = 0.35,
) -> float | None:
    switches = _switch_times(segments)
    if not switches:
        return 1.0
    if not change_points:
        return None
    supported = 0
    for s in switches:
        nearest = min(change_points, key=lambda cp: abs(cp - s))
        if abs(nearest - s) <= tolerance_s:
            supported += 1
    return supported / max(len(switches), 1)


def segment_boundary_accuracy_proxy(
    segments: list[dict],
    change_points: list[float],
    cap_s: float = 0.7,
) -> float | None:
    boundaries = _boundary_times(segments)
    if not boundaries:
        return 1.0
    if not change_points:
        return None

    dists = []
    for b in boundaries:
        nearest = min(change_points, key=lambda cp: abs(cp - b))
        dists.append(min(abs(nearest - b), cap_s))
    return 1.0 - (float(np.mean(dists)) / cap_s)


def average_segment_length(segments: list[dict]) -> float:
    if not segments:
        return 0.0
    durations = [max(0.0, float(s["end"]) - float(s["start"])) for s in segments]
    return float(np.mean(durations)) if durations else 0.0


def speaker_consistency_score(
    audio_path: str,
    segments: list[dict],
    cfg: PipelineConfig,
) -> float | None:
    """
    Consistency proxy:
      mean cosine similarity(within-speaker pairs) - mean cosine similarity(cross-speaker pairs)
    Uses MFCC summary vectors for speed and dependency safety.
    """
    if len(segments) < 2:
        return None

    audio, sr = load_audio(audio_path, sr=cfg.sample_rate)
    emb = []
    labels = []
    for seg in segments:
        s = int(float(seg["start"]) * sr)
        e = int(float(seg["end"]) * sr)
        if e - s < int(0.25 * sr):
            continue
        chunk = audio[s:e]
        mfcc = compute_mfcc(chunk, sr=sr)
        vec = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        emb.append(vec)
        labels.append(seg["speaker"])

    if len(emb) < 2:
        return None

    emb = np.stack(emb, axis=0)
    within = []
    between = []
    for i in range(len(emb)):
        for j in range(i + 1, len(emb)):
            sim = _cosine(emb[i], emb[j])
            if labels[i] == labels[j]:
                within.append(sim)
            else:
                between.append(sim)

    if not within or not between:
        return None

    return float(np.mean(within) - np.mean(between))


def compute_validation_metrics(
    audio_path: str,
    segments: list[dict],
    change_points: list[float],
    cfg: PipelineConfig,
) -> dict:
    return {
        "speaker_switch_accuracy_proxy": speaker_switch_accuracy_proxy(segments, change_points),
        "segment_boundary_accuracy_proxy": segment_boundary_accuracy_proxy(segments, change_points),
        "average_segment_length_s": average_segment_length(segments),
        "speaker_consistency_proxy": speaker_consistency_score(audio_path, segments, cfg),
        "num_segments": len(segments),
        "num_switches": len(_switch_times(segments)),
    }
