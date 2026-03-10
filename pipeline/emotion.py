"""
Emotion / Pitch / Heated Conversation Analysis — Acoustic-grounded.

Extracts acoustic features using openSMILE eGeMAPS, detects heated windows
via rule-based thresholds, performs speaker attribution and escalation
detection, and aligns with transcript segments.

This module is fully CPU-based, deterministic, and fails gracefully.
All outputs are appended under the NEW 'emotion_analysis' key.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Acoustic Feature Extraction (openSMILE eGeMAPS)
# ─────────────────────────────────────────────────────────────

def extract_acoustic_features(
    audio_path: str,
    speaker_segments: list[dict],
    sample_rate: int = 16000,
    window_sec: float = 0.5,
    overlap: float = 0.5,
) -> list[dict]:
    """Extract per-window acoustic features from diarized audio using openSMILE eGeMAPS.

    Args:
        audio_path: Path to preprocessed WAV (16kHz mono).
        speaker_segments: Diarized segments [{speaker, start, end}].
        sample_rate: Audio sample rate.
        window_sec: Sliding window duration in seconds.
        overlap: Window overlap fraction (0.5 = 50%).

    Returns:
        List of feature dicts, one per window:
        [{speaker, start, end, pitch_mean, pitch_max, rms_energy,
          energy_delta, jitter, shimmer, mfcc_mean, pitch_z, energy_z}]
    """
    import opensmile
    import soundfile as sf

    print("🎭 Extracting acoustic features (openSMILE eGeMAPS)...")

    try:
        audio, file_sr = sf.read(audio_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        logger.error(f"Failed to load audio for emotion analysis: {e}")
        return []

    # Initialize openSMILE with eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set)
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    except Exception as e:
        logger.error(f"Failed to initialize openSMILE: {e}")
        return []

    hop_sec = window_sec * (1.0 - overlap)
    all_windows = []

    for seg in speaker_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        speaker = seg["speaker"]

        # Skip very short segments
        if seg_end - seg_start < window_sec:
            continue

        # Extract audio slice
        i_start = int(seg_start * file_sr)
        i_end = int(seg_end * file_sr)
        seg_audio = audio[i_start:i_end]

        if len(seg_audio) < int(window_sec * file_sr * 0.5):
            continue

        # Slide windows across this segment
        win_samples = int(window_sec * file_sr)
        hop_samples = int(hop_sec * file_sr)

        for win_start in range(0, len(seg_audio) - win_samples + 1, hop_samples):
            win_end = win_start + win_samples
            window_audio = seg_audio[win_start:win_end]

            # Absolute timestamps
            abs_start = seg_start + win_start / file_sr
            abs_end = seg_start + win_end / file_sr

            # Extract eGeMAPS features for this window
            try:
                features_df = smile.process_signal(window_audio, file_sr)

                if features_df.empty:
                    continue

                row = features_df.iloc[0]

                # Core features from eGeMAPS (column names from openSMILE)
                pitch_mean = float(row.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0.0))
                pitch_max = float(row.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0.0))  # LLD level
                rms_energy = float(row.get("loudness_sma3_amean", 0.0))
                jitter = float(row.get("jitterLocal_sma3nz_amean", 0.0))
                shimmer = float(row.get("shimmerLocaldB_sma3nz_amean", 0.0))

                # Compute RMS energy from raw signal as backup
                raw_rms = float(np.sqrt(np.mean(window_audio ** 2)))

                # MFCC statistics (from the audio directly using simple computation)
                mfcc_mean = _compute_mfcc_mean(window_audio, file_sr)

                all_windows.append({
                    "speaker": speaker,
                    "start": round(abs_start, 3),
                    "end": round(abs_end, 3),
                    "pitch_mean": pitch_mean,
                    "pitch_max": pitch_max,
                    "rms_energy": rms_energy if rms_energy > 0 else raw_rms,
                    "energy_delta": 0.0,  # computed below
                    "jitter": jitter,
                    "shimmer": shimmer,
                    "mfcc_mean": mfcc_mean,
                    "pitch_z": 0.0,  # computed after normalization
                    "energy_z": 0.0,
                })

            except Exception as e:
                logger.debug(f"openSMILE failed for window {abs_start:.1f}-{abs_end:.1f}: {e}")
                continue

    # Compute energy deltas (rate of change between consecutive same-speaker windows)
    _compute_energy_deltas(all_windows)

    # Z-score normalize per speaker
    _zscore_normalize(all_windows)

    print(f"   🎭 Extracted {len(all_windows)} acoustic feature windows "
          f"from {len(speaker_segments)} speaker segments")

    return all_windows


def _compute_mfcc_mean(audio: np.ndarray, sr: int) -> float:
    """Compute mean of first 3 MFCCs as a simple spectral summary."""
    try:
        import torch
        import torchaudio
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        transform = torchaudio.transforms.MFCC(
            sample_rate=sr, n_mfcc=4,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23},
        )
        mfcc = transform(waveform).squeeze(0).numpy()
        # Return mean of coefficients 1-3 (skip coefficient 0 which is energy)
        if mfcc.shape[0] > 1:
            return float(np.mean(mfcc[1:4]))
        return 0.0
    except Exception:
        return 0.0


def _compute_energy_deltas(windows: list[dict]) -> None:
    """Compute energy delta (change rate) between consecutive same-speaker windows."""
    # Group by speaker
    by_speaker: dict[str, list[int]] = {}
    for i, w in enumerate(windows):
        by_speaker.setdefault(w["speaker"], []).append(i)

    for indices in by_speaker.values():
        for k in range(1, len(indices)):
            prev_idx = indices[k - 1]
            curr_idx = indices[k]
            dt = windows[curr_idx]["start"] - windows[prev_idx]["start"]
            if dt > 0:
                windows[curr_idx]["energy_delta"] = (
                    (windows[curr_idx]["rms_energy"] - windows[prev_idx]["rms_energy"]) / dt
                )


def _zscore_normalize(windows: list[dict]) -> None:
    """Z-score normalize pitch and energy per speaker."""
    by_speaker: dict[str, list[int]] = {}
    for i, w in enumerate(windows):
        by_speaker.setdefault(w["speaker"], []).append(i)

    for indices in by_speaker.values():
        if len(indices) < 2:
            continue

        pitches = np.array([windows[i]["pitch_mean"] for i in indices])
        energies = np.array([windows[i]["rms_energy"] for i in indices])

        p_mean, p_std = np.mean(pitches), np.std(pitches)
        e_mean, e_std = np.mean(energies), np.std(energies)

        for idx in indices:
            if p_std > 1e-8:
                windows[idx]["pitch_z"] = round((windows[idx]["pitch_mean"] - p_mean) / p_std, 3)
            if e_std > 1e-8:
                windows[idx]["energy_z"] = round((windows[idx]["rms_energy"] - e_mean) / e_std, 3)


# ─────────────────────────────────────────────────────────────
# STEP 2 — Heated Window Detection (Rule-Based)
# ─────────────────────────────────────────────────────────────

def detect_heated_windows(
    features: list[dict],
    pitch_threshold: float = 2.5,
    energy_threshold: float = 2.0,
) -> list[dict]:
    """Detect heated windows using rule-based pitch + energy thresholds.

    A window is HEATED if:
        pitch_z_score > pitch_threshold AND energy_z_score > energy_threshold

    Returns:
        List of heated windows: [{speaker, start, end, pitch_z, energy_z}]
    """
    heated = []
    for w in features:
        if w["pitch_z"] > pitch_threshold and w["energy_z"] > energy_threshold:
            heated.append({
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "pitch_z": w["pitch_z"],
                "energy_z": w["energy_z"],
            })

    return heated


# ─────────────────────────────────────────────────────────────
# STEP 3 — Speaker Attribution
# ─────────────────────────────────────────────────────────────

def attribute_heated_segments(heated_windows: list[dict]) -> dict:
    """Classify heated windows by speaker.

    Returns:
        {
            "agent_heated": [{start, end, pitch_z, energy_z}],
            "customer_heated": [{start, end, pitch_z, energy_z}],
            "agent_raised_voice": bool,
            "customer_raised_voice": bool,
        }
    """
    agent_heated = [w for w in heated_windows if w["speaker"] == "agent"]
    customer_heated = [w for w in heated_windows if w["speaker"] == "customer"]

    return {
        "agent_heated": agent_heated,
        "customer_heated": customer_heated,
        "agent_raised_voice": len(agent_heated) > 0,
        "customer_raised_voice": len(customer_heated) > 0,
    }


# ─────────────────────────────────────────────────────────────
# STEP 4 — Escalation Detection
# ─────────────────────────────────────────────────────────────

def detect_escalation(heated_windows: list[dict], min_streak: int = 3) -> dict:
    """Detect escalation patterns (3+ consecutive heated windows).

    Returns:
        {
            "escalation_detected": bool,
            "first_escalator": "agent" | "customer" | "none",
            "total_heated_duration_seconds": float,
            "longest_escalation_streak": int,
            "escalation_events": [{start, end, duration, speaker_sequence}],
        }
    """
    if not heated_windows:
        return {
            "escalation_detected": False,
            "first_escalator": "none",
            "total_heated_duration_seconds": 0.0,
            "longest_escalation_streak": 0,
            "escalation_events": [],
        }

    # Sort by time
    sorted_windows = sorted(heated_windows, key=lambda w: w["start"])

    # Total heated duration
    total_duration = sum(w["end"] - w["start"] for w in sorted_windows)

    # Find streaks (consecutive windows within 1.5s of each other)
    streaks = []
    current_streak = [sorted_windows[0]]

    for i in range(1, len(sorted_windows)):
        gap = sorted_windows[i]["start"] - sorted_windows[i - 1]["end"]
        if gap < 1.5:  # consecutive if gap < 1.5s
            current_streak.append(sorted_windows[i])
        else:
            streaks.append(current_streak)
            current_streak = [sorted_windows[i]]
    streaks.append(current_streak)

    # Find escalation events (streaks with >= min_streak windows)
    escalation_events = []
    longest_streak = 0
    for streak in streaks:
        if len(streak) >= min_streak:
            escalation_events.append({
                "start": streak[0]["start"],
                "end": streak[-1]["end"],
                "duration": round(streak[-1]["end"] - streak[0]["start"], 2),
                "window_count": len(streak),
                "speaker_sequence": [w["speaker"] for w in streak],
            })
        longest_streak = max(longest_streak, len(streak))

    # First escalator: who has the earliest heated window?
    first_escalator = "none"
    if sorted_windows:
        first_escalator = sorted_windows[0]["speaker"]

    return {
        "escalation_detected": len(escalation_events) > 0,
        "first_escalator": first_escalator,
        "total_heated_duration_seconds": round(total_duration, 2),
        "longest_escalation_streak": longest_streak,
        "escalation_events": escalation_events,
    }


# ─────────────────────────────────────────────────────────────
# STEP 5 — Transcript Alignment
# ─────────────────────────────────────────────────────────────

def align_with_transcript(
    heated_windows: list[dict],
    transcript_segments: list[dict],
    window_tolerance: float = 2.0,
) -> list[dict]:
    """Align heated windows with transcript text (±2 seconds).

    Returns:
        List of [{start_time, end_time, speaker, text, pitch_z, energy_z}]
    """
    aligned = []
    for hw in heated_windows:
        # Find transcript segments that overlap within ±tolerance
        matching_text = []
        for seg in transcript_segments:
            seg_start = seg.get("t0", 0)
            seg_end = seg.get("t1", 0)

            # Check overlap with tolerance
            if (seg_start <= hw["end"] + window_tolerance and
                    seg_end >= hw["start"] - window_tolerance):
                matching_text.append(seg.get("text", ""))

        text = " ".join(matching_text).strip() if matching_text else ""

        aligned.append({
            "start_time": _format_timestamp(hw["start"]),
            "end_time": _format_timestamp(hw["end"]),
            "start_seconds": hw["start"],
            "end_seconds": hw["end"],
            "speaker": hw["speaker"],
            "text": text,
            "pitch_z": hw.get("pitch_z", 0.0),
            "energy_z": hw.get("energy_z", 0.0),
        })

    return aligned


def _format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────
# STEP 6 — Expanded Feature Enrichment (Phase 30)
# ─────────────────────────────────────────────────────────────

def _enrich_features(features: list[dict], audio_path: str, speaker_segments: list[dict]) -> None:
    """Enrich existing feature windows with additional acoustic dimensions.

    Adds IN-PLACE to each window dict:
        hnr, spectral_flux, speech_rate, pause_before, pause_duration,
        pitch_variance, pitch_slope, jitter_z, shimmer_z, hnr_z,
        speech_rate_z, pause_freq_z
    
    Existing keys are NEVER modified. Only new keys are added.
    """
    import soundfile as sf

    try:
        audio, file_sr = sf.read(audio_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        logger.warning(f"Could not load audio for feature enrichment: {e}")
        # Set defaults so downstream classifiers still work
        for w in features:
            w.setdefault("hnr", 0.0)
            w.setdefault("spectral_flux", 0.0)
            w.setdefault("speech_rate", 0.0)
            w.setdefault("pause_before", 0.0)
            w.setdefault("pitch_variance", 0.0)
            w.setdefault("pitch_slope", 0.0)
        _zscore_normalize_extended(features)
        return

    # --- HNR: Extract from openSMILE if available, else estimate ---
    try:
        import opensmile
        smile_hnr = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    except Exception:
        smile_hnr = None

    # --- Per-window enrichment ---
    for w in features:
        i_start = int(w["start"] * file_sr)
        i_end = int(w["end"] * file_sr)
        window_audio = audio[i_start:i_end]

        # HNR from openSMILE
        hnr_val = 0.0
        if smile_hnr is not None and len(window_audio) > 0:
            try:
                df = smile_hnr.process_signal(window_audio, file_sr)
                if not df.empty:
                    hnr_val = float(df.iloc[0].get("HNRdBACF_sma3nz_amean", 0.0))
            except Exception:
                pass
        w["hnr"] = hnr_val

        # Spectral flux (mean frame-level spectral change)
        w["spectral_flux"] = _compute_spectral_flux(window_audio, file_sr)

        # Speech rate estimate (zero-crossing rate as proxy for syllable rate)
        w["speech_rate"] = _estimate_speech_rate(window_audio, file_sr)

        # Pitch variance and slope (within this window, from raw signal)
        pv, ps = _compute_pitch_dynamics(window_audio, file_sr)
        w["pitch_variance"] = pv
        w["pitch_slope"] = ps

    # --- Pause computation (gap between consecutive same-speaker windows) ---
    by_speaker: dict[str, list[int]] = {}
    for i, w in enumerate(features):
        by_speaker.setdefault(w["speaker"], []).append(i)

    for indices in by_speaker.values():
        for k in range(len(indices)):
            idx = indices[k]
            if k == 0:
                features[idx]["pause_before"] = 0.0
            else:
                prev_idx = indices[k - 1]
                gap = features[idx]["start"] - features[prev_idx]["end"]
                features[idx]["pause_before"] = max(gap, 0.0)

    # --- Z-score normalize new features ---
    _zscore_normalize_extended(features)


def _compute_spectral_flux(audio: np.ndarray, sr: int) -> float:
    """Compute mean spectral flux (frame-level spectral change)."""
    try:
        frame_len = int(0.025 * sr)  # 25ms frames
        hop_len = int(0.010 * sr)    # 10ms hop
        if len(audio) < frame_len * 2:
            return 0.0

        # Compute short-time FFT magnitudes
        n_frames = (len(audio) - frame_len) // hop_len + 1
        if n_frames < 2:
            return 0.0

        prev_spec = None
        flux_values = []
        for i in range(n_frames):
            start = i * hop_len
            frame = audio[start:start + frame_len]
            spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            if prev_spec is not None and len(spec) == len(prev_spec):
                diff = np.sum((spec - prev_spec) ** 2)
                flux_values.append(diff)
            prev_spec = spec

        return float(np.mean(flux_values)) if flux_values else 0.0
    except Exception:
        return 0.0


def _estimate_speech_rate(audio: np.ndarray, sr: int) -> float:
    """Estimate speech rate using zero-crossing rate as syllable proxy."""
    try:
        if len(audio) < sr * 0.1:  # Less than 100ms
            return 0.0

        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        # Scale to approximate syllables per second (empirical mapping)
        duration = len(audio) / sr
        rate = zcr * sr / duration if duration > 0 else 0.0
        return float(rate)
    except Exception:
        return 0.0


def _compute_pitch_dynamics(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Compute pitch variance and pitch slope within a window.

    Returns (pitch_variance, pitch_slope).
    """
    try:
        # Simple autocorrelation-based pitch tracking in sub-frames
        sub_frame_len = int(0.03 * sr)  # 30ms sub-frames
        hop = int(0.015 * sr)           # 15ms hop
        pitches = []

        for start in range(0, len(audio) - sub_frame_len, hop):
            frame = audio[start:start + sub_frame_len]
            # Autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]

            # Find first peak after the initial decay
            min_lag = int(sr / 500)  # Max 500 Hz
            max_lag = int(sr / 50)   # Min 50 Hz
            if max_lag > len(corr):
                continue

            search_region = corr[min_lag:max_lag]
            if len(search_region) == 0:
                continue

            peak_idx = np.argmax(search_region) + min_lag
            if corr[peak_idx] > 0.3 * corr[0]:  # Voicing threshold
                pitch = sr / peak_idx
                pitches.append(pitch)

        if len(pitches) < 2:
            return 0.0, 0.0

        variance = float(np.var(pitches))
        # Slope: linear regression of pitch over time
        x = np.arange(len(pitches), dtype=float)
        slope = float(np.polyfit(x, pitches, 1)[0])

        return variance, slope
    except Exception:
        return 0.0, 0.0


def _zscore_normalize_extended(features: list[dict]) -> None:
    """Z-score normalize the new enriched features per speaker.

    Adds: jitter_z, shimmer_z, hnr_z, speech_rate_z, pause_freq_z.
    Does NOT modify existing pitch_z or energy_z.
    """
    by_speaker: dict[str, list[int]] = {}
    for i, w in enumerate(features):
        by_speaker.setdefault(w["speaker"], []).append(i)

    new_features_to_norm = [
        ("jitter", "jitter_z"),
        ("shimmer", "shimmer_z"),
        ("hnr", "hnr_z"),
        ("speech_rate", "speech_rate_z"),
        ("pause_before", "pause_freq_z"),
        ("spectral_flux", "spectral_flux_z"),
        ("pitch_variance", "pitch_variance_z"),
    ]

    for indices in by_speaker.values():
        if len(indices) < 2:
            for idx in indices:
                for raw_key, z_key in new_features_to_norm:
                    features[idx][z_key] = 0.0
            continue

        for raw_key, z_key in new_features_to_norm:
            values = np.array([features[i].get(raw_key, 0.0) for i in indices])
            mean_val, std_val = float(np.mean(values)), float(np.std(values))
            for idx in indices:
                if std_val > 1e-8:
                    features[idx][z_key] = round(
                        (features[idx].get(raw_key, 0.0) - mean_val) / std_val, 3
                    )
                else:
                    features[idx][z_key] = 0.0


# ─────────────────────────────────────────────────────────────
# STEP 7 — Multi-Emotion Classification (Phase 30)
# ─────────────────────────────────────────────────────────────

# Emotion labels and their emoji for frontend display
EMOTION_LABELS = {
    "angry": "😡 Angry",
    "frustrated": "😤 Frustrated",
    "hesitant": "😟 Hesitant",
    "engaged": "🟢 Engaged",
    "confused": "😕 Confused",
    "calm": "😌 Calm",
}


def classify_emotion_window(w: dict) -> dict:
    """Rule-based per-window emotion classifier.

    Uses z-score normalized acoustic features to determine emotion state.
    Returns {emotion: str, confidence: float (0-1)}.

    Classification rules (checked in priority order):
        angry:      pitch_z > 2   AND energy_z > 2
        frustrated: pitch_z > 1   AND jitter_z > 1 AND pause_freq_z > 0.5
        hesitant:   speech_rate_z < -1 AND pause_freq_z > 1
        engaged:    pitch_z > 1   AND energy_z > 1 AND speech_rate_z > 0.5
        confused:   speech_rate_z < -0.5 AND pause_freq_z > 1
        calm:       (default)
    """
    pitch_z = w.get("pitch_z", 0.0)
    energy_z = w.get("energy_z", 0.0)
    jitter_z = w.get("jitter_z", 0.0)
    speech_rate_z = w.get("speech_rate_z", 0.0)
    pause_freq_z = w.get("pause_freq_z", 0.0)

    # ANGRY: high pitch + high energy
    if pitch_z > 2.0 and energy_z > 2.0:
        signals = [abs(pitch_z), abs(energy_z)]
        confidence = min(float(np.mean(signals)) / 4.0, 1.0)
        return {"emotion": "angry", "confidence": round(confidence, 3)}

    # FRUSTRATED: elevated pitch + voice instability + pausing
    if pitch_z > 1.0 and jitter_z > 1.0 and pause_freq_z > 0.5:
        signals = [abs(pitch_z), abs(jitter_z), abs(pause_freq_z)]
        confidence = min(float(np.mean(signals)) / 3.0, 1.0)
        return {"emotion": "frustrated", "confidence": round(confidence, 3)}

    # HESITANT: slow speech + long pauses
    if speech_rate_z < -1.0 and pause_freq_z > 1.0:
        signals = [abs(speech_rate_z), abs(pause_freq_z)]
        confidence = min(float(np.mean(signals)) / 3.0, 1.0)
        return {"emotion": "hesitant", "confidence": round(confidence, 3)}

    # ENGAGED: elevated pitch + energy + fast speech
    if pitch_z > 1.0 and energy_z > 1.0 and speech_rate_z > 0.5:
        signals = [abs(pitch_z), abs(energy_z), abs(speech_rate_z)]
        confidence = min(float(np.mean(signals)) / 3.0, 1.0)
        return {"emotion": "engaged", "confidence": round(confidence, 3)}

    # CONFUSED: slow speech + repeated pauses
    if speech_rate_z < -0.5 and pause_freq_z > 1.0:
        signals = [abs(speech_rate_z), abs(pause_freq_z)]
        confidence = min(float(np.mean(signals)) / 3.0, 1.0)
        return {"emotion": "confused", "confidence": round(confidence, 3)}

    # CALM: default state
    # Confidence inversely related to how extreme any z-score is
    max_z = max(abs(pitch_z), abs(energy_z), abs(jitter_z))
    calm_confidence = max(1.0 - max_z / 3.0, 0.2)
    return {"emotion": "calm", "confidence": round(calm_confidence, 3)}


# ─────────────────────────────────────────────────────────────
# STEP 8 — Emotion Timeline Generation (Phase 30)
# ─────────────────────────────────────────────────────────────

def build_emotion_timeline(
    features: list[dict],
    transcript_segments: list[dict],
) -> list[dict]:
    """Generate chronological emotion timeline from classified windows.

    Adjacent windows with the same speaker + emotion are merged to reduce noise.

    Returns:
        [{timestamp, timestamp_seconds, speaker, emotion, confidence, emotion_label}]
    """
    if not features:
        return []

    # Classify each window
    raw_entries = []
    for w in features:
        classification = classify_emotion_window(w)
        raw_entries.append({
            "start": w["start"],
            "end": w["end"],
            "speaker": w["speaker"],
            "emotion": classification["emotion"],
            "confidence": classification["confidence"],
        })

    # Sort by time
    raw_entries.sort(key=lambda e: e["start"])

    # Merge adjacent windows with same speaker + emotion
    merged = []
    for entry in raw_entries:
        if (merged and
                merged[-1]["speaker"] == entry["speaker"] and
                merged[-1]["emotion"] == entry["emotion"] and
                entry["start"] - merged[-1]["end"] < 1.0):
            # Extend the previous entry
            merged[-1]["end"] = entry["end"]
            # Average confidence
            merged[-1]["confidence"] = round(
                (merged[-1]["confidence"] + entry["confidence"]) / 2, 3
            )
        else:
            merged.append(dict(entry))

    # Format for output
    timeline = []
    for entry in merged:
        emotion = entry["emotion"]
        timeline.append({
            "timestamp": _format_timestamp(entry["start"]),
            "timestamp_seconds": round(entry["start"], 2),
            "end_timestamp": _format_timestamp(entry["end"]),
            "end_seconds": round(entry["end"], 2),
            "speaker": entry["speaker"],
            "emotion": emotion,
            "emotion_label": EMOTION_LABELS.get(emotion, emotion),
            "confidence": entry["confidence"],
        })

    return timeline


# ─────────────────────────────────────────────────────────────
# STEP 9 — Call-Level Emotion Metrics (Phase 30)
# ─────────────────────────────────────────────────────────────

def compute_emotion_metrics(timeline: list[dict], features: list[dict]) -> dict:
    """Compute call-level emotional metrics from the emotion timeline.

    Returns:
        {
            emotion_switch_count, highest_emotional_intensity_moment,
            dominant_customer_emotion, dominant_agent_emotion,
            emotional_volatility_score, agent_calming_effectiveness
        }
    """
    if not timeline:
        return {
            "emotion_switch_count": 0,
            "highest_emotional_intensity_moment": None,
            "dominant_customer_emotion": "calm",
            "dominant_agent_emotion": "calm",
            "emotional_volatility_score": 0.0,
            "agent_calming_effectiveness": None,
        }

    # 1. Emotion switch count (transitions between different emotions)
    switches = 0
    for i in range(1, len(timeline)):
        if timeline[i]["emotion"] != timeline[i - 1]["emotion"]:
            switches += 1

    # 2. Highest emotional intensity moment
    non_calm = [e for e in timeline if e["emotion"] != "calm"]
    if non_calm:
        peak = max(non_calm, key=lambda e: e["confidence"])
        highest_moment = {
            "timestamp": peak["timestamp"],
            "timestamp_seconds": peak["timestamp_seconds"],
            "speaker": peak["speaker"],
            "emotion": peak["emotion"],
            "emotion_label": peak["emotion_label"],
            "confidence": peak["confidence"],
        }
    else:
        highest_moment = None

    # 3. Dominant emotions per speaker
    from collections import Counter
    customer_emotions = [e["emotion"] for e in timeline if e["speaker"] == "customer"]
    agent_emotions = [e["emotion"] for e in timeline if e["speaker"] == "agent"]

    dominant_customer = Counter(customer_emotions).most_common(1)[0][0] if customer_emotions else "calm"
    dominant_agent = Counter(agent_emotions).most_common(1)[0][0] if agent_emotions else "calm"

    # 4. Emotional volatility score (ratio of switches to total entries, 0-1)
    volatility = round(switches / max(len(timeline) - 1, 1), 3) if len(timeline) > 1 else 0.0

    # 5. Agent calming effectiveness
    # If customer was angry/frustrated and agent was calm, did customer emotion improve?
    calming_score = _compute_calming_effectiveness(timeline)

    return {
        "emotion_switch_count": switches,
        "highest_emotional_intensity_moment": highest_moment,
        "dominant_customer_emotion": dominant_customer,
        "dominant_agent_emotion": dominant_agent,
        "emotional_volatility_score": volatility,
        "agent_calming_effectiveness": calming_score,
    }


def _compute_calming_effectiveness(timeline: list[dict]) -> float | None:
    """Score how effective the agent was at calming a distressed customer.

    Looks for patterns where:
    1. Customer is angry/frustrated
    2. Agent responds calmly
    3. Customer emotion subsequently improves

    Returns 0-1 score, or None if no calming opportunity existed.
    """
    negative_emotions = {"angry", "frustrated"}
    positive_emotions = {"calm", "engaged"}

    calming_opportunities = 0
    calming_successes = 0

    for i in range(len(timeline) - 2):
        # Pattern: customer negative → agent calm → customer improved
        if (timeline[i]["speaker"] == "customer" and
                timeline[i]["emotion"] in negative_emotions):
            # Look for agent response
            for j in range(i + 1, min(i + 4, len(timeline))):
                if timeline[j]["speaker"] == "agent" and timeline[j]["emotion"] in positive_emotions:
                    calming_opportunities += 1
                    # Check if customer improved after agent's calm response
                    for k in range(j + 1, min(j + 4, len(timeline))):
                        if timeline[k]["speaker"] == "customer":
                            if timeline[k]["emotion"] in positive_emotions:
                                calming_successes += 1
                            break
                    break

    if calming_opportunities == 0:
        return None

    return round(calming_successes / calming_opportunities, 3)


# ─────────────────────────────────────────────────────────────
# STEP 10 — Main Orchestrator
# ─────────────────────────────────────────────────────────────

def analyze_emotion(
    audio_path: str,
    speaker_segments: list[dict],
    transcript_segments: list[dict],
) -> dict:
    """Full emotion analysis pipeline. Graceful failure guaranteed.

    Args:
        audio_path: Path to preprocessed WAV.
        speaker_segments: Diarized segments [{speaker, start, end}].
        transcript_segments: ASR output [{speaker, t0, t1, text, role}].

    Returns:
        Complete emotion_analysis dict matching the required output schema.
        Returns empty/default structure on any error.
    """
    empty_result = {
        "agent_raised_voice": False,
        "customer_raised_voice": False,
        "agent_heated_segments": [],
        "customer_heated_segments": [],
        "escalation_detected": False,
        "first_escalator": "none",
        "total_heated_duration_seconds": 0.0,
        "longest_escalation_streak": 0,
        "call_emotional_intensity_score": 0.0,
        "escalation_events": [],
    }

    try:
        print("🎭 Running Emotion & Heated Conversation Analysis...")

        # Step 1: Extract acoustic features
        features = extract_acoustic_features(audio_path, speaker_segments)
        if not features:
            print("   ⚠️  No acoustic features extracted. Returning empty emotion analysis.")
            return empty_result

        # Step 2: Detect heated windows
        heated = detect_heated_windows(features)

        # Step 3: Speaker attribution
        attribution = attribute_heated_segments(heated)

        # Step 4: Escalation detection
        escalation = detect_escalation(heated)

        # Step 5: Transcript alignment
        agent_aligned = align_with_transcript(attribution["agent_heated"], transcript_segments)
        customer_aligned = align_with_transcript(attribution["customer_heated"], transcript_segments)

        # Compute call emotional intensity score (0–1 normalized)
        # Based on: % of windows that are heated, weighted by z-score magnitude
        total_windows = len(features)
        heated_count = len(heated)
        if total_windows > 0:
            heated_ratio = heated_count / total_windows
            # Average z-score magnitude of heated windows (normalized to 0-1 range)
            avg_z = 0.0
            if heated:
                avg_z = np.mean([max(h["pitch_z"], h["energy_z"]) for h in heated])
            # Intensity: 50% heated ratio + 50% z-score magnitude (capped at z=5)
            intensity = 0.5 * min(heated_ratio * 5, 1.0) + 0.5 * min(avg_z / 5.0, 1.0)
            intensity = round(min(max(intensity, 0.0), 1.0), 3)
        else:
            intensity = 0.0

        result = {
            "agent_raised_voice": attribution["agent_raised_voice"],
            "customer_raised_voice": attribution["customer_raised_voice"],
            "agent_heated_segments": agent_aligned,
            "customer_heated_segments": customer_aligned,
            "escalation_detected": escalation["escalation_detected"],
            "first_escalator": escalation["first_escalator"],
            "total_heated_duration_seconds": escalation["total_heated_duration_seconds"],
            "longest_escalation_streak": escalation["longest_escalation_streak"],
            "call_emotional_intensity_score": intensity,
            "escalation_events": escalation["escalation_events"],
        }

        # Print stats
        print(f"   🔴 Agent heated: {len(agent_aligned)} segments | "
              f"Customer heated: {len(customer_aligned)} segments")
        print(f"   📊 Intensity score: {intensity:.3f} | "
              f"Escalation: {'YES' if escalation['escalation_detected'] else 'NO'}")

        # ── Phase 30: Expanded Emotion Analysis (additive only) ──
        try:
            # Step 6: Enrich features with HNR, spectral flux, speech rate, pauses
            print("   🧠 Enriching acoustic features for multi-emotion analysis...")
            _enrich_features(features, audio_path, speaker_segments)

            # Step 7+8: Build emotion timeline (classifies each window + merges)
            timeline = build_emotion_timeline(features, transcript_segments)

            # Step 9: Compute call-level emotion metrics
            metrics = compute_emotion_metrics(timeline, features)

            # Append NEW keys to result — existing keys are untouched
            result["emotion_timeline"] = timeline
            result.update(metrics)

            # Print expanded stats
            print(f"   🧠 Emotion timeline: {len(timeline)} entries | "
                  f"Switches: {metrics['emotion_switch_count']} | "
                  f"Volatility: {metrics['emotional_volatility_score']:.3f}")
            print(f"   🧠 Dominant: Agent={metrics['dominant_agent_emotion']} | "
                  f"Customer={metrics['dominant_customer_emotion']}")

        except Exception as e:
            logger.warning(f"Expanded emotion analysis failed (non-fatal): {e}")
            print(f"   ⚠️  Expanded emotion analysis error: {e}. Continuing with base results.")
            # Ensure new keys exist even on failure (backward compat for frontend)
            result.setdefault("emotion_timeline", [])
            result.setdefault("emotion_switch_count", 0)
            result.setdefault("dominant_customer_emotion", "calm")
            result.setdefault("dominant_agent_emotion", "calm")
            result.setdefault("emotional_volatility_score", 0.0)
            result.setdefault("highest_emotional_intensity_moment", None)
            result.setdefault("agent_calming_effectiveness", None)

        return result

    except Exception as e:
        logger.error(f"Emotion analysis failed gracefully: {e}", exc_info=True)
        print(f"   ⚠️  Emotion analysis error: {e}. Returning empty result.")
        return empty_result
