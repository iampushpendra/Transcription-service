"""
Spectral clustering for advanced diarization.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from ..config import PipelineConfig


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def _affinity_matrix(x: np.ndarray) -> np.ndarray:
    xn = _normalize_rows(x)
    aff = np.matmul(xn, xn.T)
    aff = (aff + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    np.fill_diagonal(aff, 1.0)
    return aff


def _spectral_labels(x: np.ndarray, n_clusters: int) -> np.ndarray:
    if n_clusters <= 1:
        return np.zeros(x.shape[0], dtype=int)
    aff = _affinity_matrix(x)
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
        n_init=10,
    )
    return model.fit_predict(aff)


def _choose_cluster_count(
    x: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[int, dict[int, float]]:
    n_samples = x.shape[0]
    max_k = min(max(2, cfg.advanced_max_speakers), n_samples)
    default_k = min(max(1, cfg.num_speakers), max_k)
    if n_samples < 4 or max_k <= 2:
        return default_k, {}

    scores: dict[int, float] = {}
    for k in range(2, max_k + 1):
        try:
            labels = _spectral_labels(x, k)
            if len(set(labels.tolist())) < 2:
                continue
            score = float(silhouette_score(x, labels, metric="cosine"))
            scores[k] = score
        except Exception:
            continue

    if not scores:
        return default_k, {}

    best_k = max(scores, key=scores.get)
    default_score = scores.get(default_k, -1.0)
    best_score = scores[best_k]

    if (
        best_k != default_k
        and best_score >= 0.2
        and (best_score - default_score) >= 0.08
    ):
        return best_k, scores

    return default_k, scores


def _label_to_speaker_ids(labels: np.ndarray, starts: list[float]) -> dict[int, str]:
    first_start = {}
    for idx, lab in enumerate(labels.tolist()):
        if lab not in first_start:
            first_start[lab] = starts[idx]
    ordered = [k for k, _ in sorted(first_start.items(), key=lambda kv: kv[1])]
    return {lab: f"speaker_{i}" for i, lab in enumerate(ordered)}


def cluster_segments(
    segments: list[dict],
    cfg: PipelineConfig,
) -> tuple[list[dict], dict]:
    """
    Cluster embeddings with spectral clustering.
    - Fits on non-overlap segments first.
    - Assigns overlap segments by nearest centroid.
    """
    if not segments:
        raise RuntimeError("cannot cluster empty segment list")

    embeddings = np.stack([np.asarray(s["embedding"], dtype=np.float32) for s in segments], axis=0)
    non_overlap_idx = [i for i, s in enumerate(segments) if not s.get("is_overlap", False)]
    fit_idx = non_overlap_idx if len(non_overlap_idx) >= 2 else list(range(len(segments)))

    x_fit = embeddings[fit_idx]
    n_clusters, sil_scores = _choose_cluster_count(x_fit, cfg)
    labels_fit = _spectral_labels(x_fit, n_clusters)

    labels_all = np.full(len(segments), -1, dtype=int)
    for local_i, global_i in enumerate(fit_idx):
        labels_all[global_i] = int(labels_fit[local_i])

    # Build centroids from fitted segments
    centroids: dict[int, np.ndarray] = {}
    grouped = defaultdict(list)
    for idx in fit_idx:
        grouped[int(labels_all[idx])].append(embeddings[idx])
    for lab, vecs in grouped.items():
        centroids[lab] = _normalize_rows(np.stack(vecs, axis=0)).mean(axis=0)

    # Assign remaining segments (typically overlap) by nearest centroid
    for i in range(len(segments)):
        if labels_all[i] != -1:
            continue
        best_label = None
        best_sim = -1.0
        for lab, centroid in centroids.items():
            sim = _cosine(embeddings[i], centroid)
            if sim > best_sim:
                best_sim = sim
                best_label = lab
        labels_all[i] = int(best_label if best_label is not None else 0)

    speaker_map = _label_to_speaker_ids(labels_all, [float(s["start"]) for s in segments])

    clustered = []
    for seg, lab in zip(segments, labels_all.tolist()):
        out = seg.copy()
        out["speaker"] = speaker_map[int(lab)]
        clustered.append(out)

    print(
        "🧮 Spectral clustering: "
        f"{len(segments)} segments -> {len(set(labels_all.tolist()))} speakers "
        f"(default={cfg.num_speakers}, chosen={n_clusters}, "
        f"silhouette={sil_scores.get(n_clusters, 'n/a'):.3f})"
        if sil_scores else
        f"🧮 Spectral clustering: {len(segments)} segments -> {len(set(labels_all.tolist()))} speakers "
        f"(default={cfg.num_speakers}, chosen={n_clusters})"
    )

    return clustered, {"num_speakers": int(len(set(labels_all.tolist()))), "silhouette": sil_scores}
