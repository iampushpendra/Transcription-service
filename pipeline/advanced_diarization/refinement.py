"""
Post-clustering refinement and smoothing for diarization segments.
"""

from __future__ import annotations

import numpy as np

from ..config import PipelineConfig


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def _avg_embedding(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    vec = (a + b) / 2.0
    norm = np.linalg.norm(vec) + 1e-9
    return (vec / norm).astype(np.float32)


def _merge_adjacent_same_speaker(segments: list[dict], gap_tol: float = 0.2) -> list[dict]:
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        same_spk = seg["speaker"] == prev["speaker"]
        close_enough = (float(seg["start"]) - float(prev["end"])) <= gap_tol
        if same_spk and close_enough:
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
            if "embedding" in prev and "embedding" in seg:
                prev["embedding"] = _avg_embedding(
                    np.asarray(prev["embedding"], dtype=np.float32),
                    np.asarray(seg["embedding"], dtype=np.float32),
                )
            prev["is_overlap"] = bool(prev.get("is_overlap", False) or seg.get("is_overlap", False))
        else:
            merged.append(seg.copy())
    return merged


def _split_long_segments_on_change_points(
    segments: list[dict],
    change_points: list[float],
    cfg: PipelineConfig,
) -> list[dict]:
    if not change_points:
        return segments

    out: list[dict] = []
    min_span = max(0.3, cfg.advanced_min_segment_s)
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if (end - start) < (2.5 * min_span):
            out.append(seg.copy())
            continue

        cps = [cp for cp in change_points if (start + min_span) < cp < (end - min_span)]
        if not cps:
            out.append(seg.copy())
            continue

        cuts = [start] + cps + [end]
        for s0, s1 in zip(cuts, cuts[1:]):
            if (s1 - s0) < min_span:
                continue
            child = seg.copy()
            child["start"] = round(float(s0), 3)
            child["end"] = round(float(s1), 3)
            out.append(child)
    return out


def _snap_switch_boundaries_to_change_points(
    segments: list[dict],
    change_points: list[float],
    max_delta: float = 0.35,
) -> list[dict]:
    if not change_points or len(segments) < 2:
        return segments

    out = [s.copy() for s in segments]
    for i in range(len(out) - 1):
        left = out[i]
        right = out[i + 1]
        if left["speaker"] == right["speaker"]:
            continue

        boundary = (float(left["end"]) + float(right["start"])) / 2.0
        nearest = min(change_points, key=lambda cp: abs(cp - boundary))
        if abs(nearest - boundary) > max_delta:
            continue

        new_boundary = round(float(nearest), 3)
        if new_boundary <= float(left["start"]) or new_boundary >= float(right["end"]):
            continue

        left["end"] = new_boundary
        right["start"] = new_boundary

    return out


def refine_clustered_segments(
    segments: list[dict],
    change_points: list[float],
    cfg: PipelineConfig,
) -> list[dict]:
    """
    Post-clustering refinement:
    - split long segments at probable speaker-change points
    - snap speaker-switch boundaries near detected change points
    - merge adjacent same-speaker spans
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (float(s["start"]), float(s["end"])))
    split = _split_long_segments_on_change_points(ordered, change_points, cfg)
    snapped = _snap_switch_boundaries_to_change_points(split, change_points)

    # Merge redundant switches where embeddings strongly agree.
    merged = []
    for seg in snapped:
        if not merged:
            merged.append(seg.copy())
            continue

        prev = merged[-1]
        if prev["speaker"] == seg["speaker"]:
            merged = _merge_adjacent_same_speaker(merged + [seg], gap_tol=0.25)
            continue

        prev_emb = np.asarray(prev.get("embedding", []), dtype=np.float32)
        curr_emb = np.asarray(seg.get("embedding", []), dtype=np.float32)
        if prev_emb.size and curr_emb.size:
            sim = _cosine(prev_emb, curr_emb)
            if sim >= cfg.advanced_merge_similarity and min(
                float(prev["end"]) - float(prev["start"]),
                float(seg["end"]) - float(seg["start"]),
            ) < 0.8:
                # If two neighboring labels look acoustically identical, keep longer segment's label.
                if (float(prev["end"]) - float(prev["start"])) >= (float(seg["end"]) - float(seg["start"])):
                    seg = seg.copy()
                    seg["speaker"] = prev["speaker"]
                else:
                    prev["speaker"] = seg["speaker"]
                    merged[-1] = prev

        merged.append(seg.copy())

    refined = _merge_adjacent_same_speaker(merged, gap_tol=0.35)
    print(f"🛠️  Refinement: {len(segments)} -> {len(refined)} segments")
    return refined


def smooth_conversation_structure(
    segments: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    """
    Phone-call prior smoothing:
    If a very short segment (<0.5s default) is sandwiched between the same speaker,
    relabel it when similarity indicates likely same speaker.
    """
    if len(segments) < 3:
        return segments

    out = [s.copy() for s in segments]
    changed = False

    for i in range(1, len(out) - 1):
        prev_seg = out[i - 1]
        curr_seg = out[i]
        next_seg = out[i + 1]

        curr_dur = float(curr_seg["end"]) - float(curr_seg["start"])
        if curr_dur >= cfg.advanced_short_segment_s:
            continue
        if prev_seg["speaker"] != next_seg["speaker"]:
            continue
        if curr_seg["speaker"] == prev_seg["speaker"]:
            continue

        prev_emb = np.asarray(prev_seg.get("embedding", []), dtype=np.float32)
        curr_emb = np.asarray(curr_seg.get("embedding", []), dtype=np.float32)
        next_emb = np.asarray(next_seg.get("embedding", []), dtype=np.float32)
        if not (prev_emb.size and curr_emb.size and next_emb.size):
            continue

        sim_prev = _cosine(curr_emb, prev_emb)
        sim_next = _cosine(curr_emb, next_emb)
        if max(sim_prev, sim_next) >= (cfg.advanced_merge_similarity - cfg.advanced_relabel_margin):
            curr_seg["speaker"] = prev_seg["speaker"]
            out[i] = curr_seg
            changed = True

    if changed:
        out = _merge_adjacent_same_speaker(out, gap_tol=0.35)
    print(f"🧽 Conversation smoothing: {len(segments)} -> {len(out)} segments")
    return out


def align_to_asr_frames(
    segments: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    """
    Align segment boundaries to ASR frame grid (20ms default).
    """
    if not segments:
        return []

    frame = max(0.01, float(cfg.advanced_asr_frame_s))
    out = []
    prev_end = 0.0

    for seg in sorted(segments, key=lambda s: (float(s["start"]), float(s["end"]))):
        start = round(round(float(seg["start"]) / frame) * frame, 3)
        end = round(round(float(seg["end"]) / frame) * frame, 3)
        start = max(start, prev_end)
        if end <= start:
            end = round(start + frame, 3)

        aligned = seg.copy()
        aligned["start"] = start
        aligned["end"] = end
        out.append(aligned)
        prev_end = end

    out = _merge_adjacent_same_speaker(out, gap_tol=max(frame, 0.02))
    print(f"🕒 ASR frame alignment: {len(segments)} -> {len(out)} segments")
    return out
