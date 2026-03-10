"""
Speaker change segmentation + overlap detection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import PipelineConfig

_SEGMENTATION_INFERENCE = None
_OVERLAP_PIPELINE = None
_OVERLAP_INFERENCE = None


@dataclass
class SegmentationArtifacts:
    segments: list[dict]
    change_points: list[float]
    overlap_regions: list[dict]


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_pretrained(loader, model_id: str, token: str | None):
    # Handle both pyannote>=4 (`token`) and older APIs (`use_auth_token`)
    if token:
        try:
            return loader(model_id, token=token)
        except TypeError:
            return loader(model_id, use_auth_token=token)
    return loader(model_id)


def _ensure_segmentation_inference(cfg: PipelineConfig):
    global _SEGMENTATION_INFERENCE
    if _SEGMENTATION_INFERENCE is not None:
        return _SEGMENTATION_INFERENCE

    from pyannote.audio import Inference, Model

    model = _load_pretrained(Model.from_pretrained, "pyannote/segmentation", cfg.hf_token)
    model.to(_resolve_device())
    _SEGMENTATION_INFERENCE = Inference(model, duration=5.0, step=0.25, batch_size=16)
    return _SEGMENTATION_INFERENCE


def _ensure_overlap_pipeline(cfg: PipelineConfig):
    global _OVERLAP_PIPELINE
    if _OVERLAP_PIPELINE is not None:
        return _OVERLAP_PIPELINE, "pipeline"

    from pyannote.audio import Inference, Model, Pipeline

    try:
        pipeline = _load_pretrained(
            Pipeline.from_pretrained,
            "pyannote/overlapped-speech-detection",
            cfg.hf_token,
        )
        pipeline.to(_resolve_device())
        _OVERLAP_PIPELINE = pipeline
        return _OVERLAP_PIPELINE, "pipeline"
    except Exception:
        # Compatibility fallback for environments where this model card cannot be
        # instantiated as a Pipeline in the installed pyannote version.
        global _OVERLAP_INFERENCE
        if _OVERLAP_INFERENCE is None:
            model = _load_pretrained(
                Model.from_pretrained,
                "pyannote/overlapped-speech-detection",
                cfg.hf_token,
            )
            model.to(_resolve_device())
            _OVERLAP_INFERENCE = Inference(model, duration=5.0, step=0.25, batch_size=16)
        return _OVERLAP_INFERENCE, "inference"


def _is_inside_vad(t: float, vad_segs: list[dict], pad: float = 0.02) -> bool:
    for seg in vad_segs:
        if (seg["start"] - pad) <= t <= (seg["end"] + pad):
            return True
    return False


def _extract_frame_times(sliding_window, n_frames: int) -> list[float]:
    times = []
    for i in range(n_frames):
        try:
            times.append(float(sliding_window[i].middle))
        except Exception:
            start = float(getattr(sliding_window, "start", 0.0))
            step = float(getattr(sliding_window, "step", 0.02))
            dur = float(getattr(sliding_window, "duration", step))
            times.append(start + i * step + dur / 2.0)
    return times


def detect_change_points(path: str, vad_segs: list[dict], cfg: PipelineConfig) -> list[float]:
    """Use pyannote segmentation scores to infer speaker change candidates."""
    inference = _ensure_segmentation_inference(cfg)
    scores = inference(path)

    if not hasattr(scores, "data"):
        raise RuntimeError("segmentation output missing score data")

    data = np.asarray(scores.data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    if data.shape[0] < 2:
        return []

    frame_times = _extract_frame_times(scores.sliding_window, data.shape[0])
    winners = np.argmax(data, axis=1)
    activity = np.max(data, axis=1)
    frame_delta = np.linalg.norm(np.diff(data, axis=0), axis=1)
    delta_thr = max(float(np.quantile(frame_delta, 0.75)), cfg.advanced_change_point_threshold)

    cps: list[float] = []
    for i in range(1, len(frame_times)):
        changed_channel = winners[i] != winners[i - 1]
        strong_jump = frame_delta[i - 1] >= delta_thr
        enough_speech = min(activity[i], activity[i - 1]) >= 0.1
        if not enough_speech or not (changed_channel or strong_jump):
            continue

        t = float(frame_times[i])
        if not _is_inside_vad(t, vad_segs):
            continue
        if cps and (t - cps[-1]) < 0.2:
            continue
        cps.append(round(t, 3))

    return cps


def _merge_regions(regions: list[dict], max_gap: float = 0.1) -> list[dict]:
    if not regions:
        return []
    merged = [regions[0].copy()]
    for seg in regions[1:]:
        prev = merged[-1]
        if seg["start"] <= prev["end"] + max_gap:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())
    return merged


def _detect_overlap_from_segmentation(path: str, cfg: PipelineConfig) -> list[dict]:
    """
    Fallback overlap detector derived from segmentation activity.
    Marks frames as overlap when >=2 segmentation channels are strongly active.
    """
    inference = _ensure_segmentation_inference(cfg)
    scores = inference(path)
    if not hasattr(scores, "data"):
        return []

    data = np.asarray(scores.data)
    if data.ndim == 1:
        return []
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    frame_times = _extract_frame_times(scores.sliding_window, data.shape[0])
    active_counts = (data >= 0.35).sum(axis=1)
    is_overlap = active_counts >= 2

    regions = []
    start = None
    for i, on in enumerate(is_overlap.tolist()):
        t = float(frame_times[i])
        if on and start is None:
            start = t
        elif not on and start is not None:
            regions.append({"start": round(start, 3), "end": round(t, 3)})
            start = None
    if start is not None:
        regions.append({"start": round(start, 3), "end": round(float(frame_times[-1]), 3)})

    regions = [r for r in regions if (r["end"] - r["start"]) >= 0.1]
    regions.sort(key=lambda x: x["start"])
    return _merge_regions(regions)


def detect_overlap_regions(path: str, cfg: PipelineConfig) -> list[dict]:
    """Detect overlapped speech regions using pyannote OSD."""
    try:
        detector, mode = _ensure_overlap_pipeline(cfg)
        output = detector(path)
    except Exception as exc:
        print(f"⚠️  OSD model unavailable, using segmentation-derived overlap fallback: {exc}")
        return _detect_overlap_from_segmentation(path, cfg)

    regions: list[dict] = []

    if mode == "pipeline" and hasattr(output, "get_timeline"):
        timeline = output.get_timeline().support()
        for seg in timeline:
            regions.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3)})
    elif mode == "pipeline" and hasattr(output, "itertracks"):
        for turn, _, _ in output.itertracks(yield_label=True):
            regions.append({"start": round(float(turn.start), 3), "end": round(float(turn.end), 3)})
    elif mode == "inference" and hasattr(output, "data"):
        data = np.asarray(output.data)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        scores = np.max(data, axis=1)
        frame_times = _extract_frame_times(output.sliding_window, data.shape[0])

        active = scores >= max(cfg.advanced_overlap_threshold, float(np.quantile(scores, 0.8)))
        start = None
        for i, on in enumerate(active.tolist()):
            t = float(frame_times[i])
            if on and start is None:
                start = t
            elif not on and start is not None:
                regions.append({"start": round(start, 3), "end": round(t, 3)})
                start = None
        if start is not None:
            regions.append({"start": round(start, 3), "end": round(float(frame_times[-1]), 3)})
    else:
        raise RuntimeError("overlap detector returned unsupported output format")

    regions = [r for r in regions if (r["end"] - r["start"]) >= 0.1]
    regions.sort(key=lambda x: x["start"])
    return _merge_regions(regions)


def _split_vad_with_change_points(
    vad_segs: list[dict],
    change_points: list[float],
    min_duration: float,
) -> list[dict]:
    segments: list[dict] = []

    for vad in sorted(vad_segs, key=lambda x: x["start"]):
        v_start = float(vad["start"])
        v_end = float(vad["end"])
        if v_end <= v_start:
            continue

        cuts = [v_start]
        for cp in change_points:
            if (v_start + min_duration) <= cp <= (v_end - min_duration):
                cuts.append(cp)
        cuts.append(v_end)
        cuts = sorted(set(round(c, 3) for c in cuts))

        for start, end in zip(cuts, cuts[1:]):
            if (end - start) >= min_duration:
                segments.append({"start": round(start, 3), "end": round(end, 3)})

    return segments


def _mark_overlap(segments: list[dict], overlap_regions: list[dict]) -> list[dict]:
    marked = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        is_overlap = False
        for ov in overlap_regions:
            ov_start = ov["start"]
            ov_end = ov["end"]
            inter = max(0.0, min(end, ov_end) - max(start, ov_start))
            if inter > 0.05:
                is_overlap = True
                break
        seg_out = seg.copy()
        seg_out["is_overlap"] = is_overlap
        marked.append(seg_out)
    return marked


def build_initial_segments(
    path: str,
    vad_segs: list[dict],
    cfg: PipelineConfig,
) -> SegmentationArtifacts:
    """
    Build initial candidate segments:
      1) pyannote segmentation -> change points
      2) pyannote OSD -> overlap windows
      3) split VAD segments at change points
      4) annotate overlap flags
    """
    if not vad_segs:
        return SegmentationArtifacts(segments=[], change_points=[], overlap_regions=[])

    change_points = detect_change_points(path, vad_segs, cfg)
    overlap_regions = detect_overlap_regions(path, cfg)
    segments = _split_vad_with_change_points(
        vad_segs=vad_segs,
        change_points=change_points,
        min_duration=cfg.advanced_min_segment_s,
    )
    segments = _mark_overlap(segments, overlap_regions)

    print(
        "🧩 Advanced segmentation: "
        f"{len(vad_segs)} VAD -> {len(segments)} candidate segments | "
        f"{len(change_points)} change points | {len(overlap_regions)} overlap regions"
    )

    return SegmentationArtifacts(
        segments=segments,
        change_points=change_points,
        overlap_regions=overlap_regions,
    )
