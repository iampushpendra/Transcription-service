"""
Advanced diarization subsystem.

This package is additive and safe-by-default:
- It runs only when explicitly invoked by pipeline.diarize.run_diarization.
- Any failure in advanced stages should be handled by caller fallback logic.
"""

from __future__ import annotations

from .clustering import cluster_segments
from .embedding import generate_embeddings
from .refinement import (
    align_to_asr_frames,
    refine_clustered_segments,
    smooth_conversation_structure,
)
from .segmenter import build_initial_segments


def run_advanced_diarization(
    path: str,
    vad_segs: list[dict],
    cfg,
) -> tuple[list[dict], str]:
    """
    Run the advanced diarization flow:
      VAD -> pyannote segmentation -> overlap detection -> embeddings ->
      spectral clustering -> refinement -> smoothing -> ASR frame alignment.

    Returns:
        (segments, method_name)
    """
    if not vad_segs:
        raise RuntimeError("advanced diarization requires non-empty VAD segments")

    artifacts = build_initial_segments(path, vad_segs, cfg)
    if not artifacts.segments:
        raise RuntimeError("advanced segmentation returned no usable segments")

    embedded, emb_backend = generate_embeddings(path, artifacts.segments, cfg)
    if not embedded:
        raise RuntimeError("advanced embedding stage produced no usable embeddings")

    clustered, cluster_meta = cluster_segments(embedded, cfg)
    refined = refine_clustered_segments(clustered, artifacts.change_points, cfg)
    smoothed = smooth_conversation_structure(refined, cfg)
    aligned = align_to_asr_frames(smoothed, cfg)

    output = []
    for seg in aligned:
        start = round(float(seg["start"]), 3)
        end = round(float(seg["end"]), 3)
        if end - start < 0.1:
            continue
        output.append(
            {
                "speaker": str(seg["speaker"]),
                "start": start,
                "end": end,
            }
        )

    if not output:
        raise RuntimeError("advanced diarization produced empty output after refinement")

    method = (
        f"advanced_pyannote_seg+osd_{emb_backend}_spectral_"
        f"{cluster_meta.get('num_speakers', cfg.num_speakers)}spk"
    )
    return output, method
