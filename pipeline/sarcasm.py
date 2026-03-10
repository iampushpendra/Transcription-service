"""
Sarcasm Detection Module — Multimodal signal analysis.

Detects potential sarcasm by combining three independent signals:
1. Text sentiment (keyword-based, no external deps)
2. Acoustic emotion (from enriched feature windows)
3. Previous conversational context

Sarcasm candidate: positive text + negative acoustic + negative context.

This module is fully deterministic and fails gracefully.
All outputs are appended under the NEW 'sarcasm_analysis' key.
"""

import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Sentiment Keyword Lexicons (Hinglish + English)
# ─────────────────────────────────────────────────────────────

_POSITIVE_KEYWORDS = {
    # English
    "great", "good", "perfect", "wonderful", "excellent", "amazing",
    "fantastic", "awesome", "sure", "absolutely", "definitely", "agree",
    "happy", "glad", "thanks", "thank you", "nice", "fine", "okay",
    "of course", "no problem", "sounds good", "love", "best",
    # Hindi / Hinglish
    "accha", "bahut accha", "sahi", "bilkul", "zaroor", "theek hai",
    "theek", "mast", "shandar", "badhiya", "achha", "haan bilkul",
    "shukriya", "dhanyavaad", "khushi", "samajh gaya",
}

_NEGATIVE_KEYWORDS = {
    # English
    "problem", "issue", "bad", "terrible", "horrible", "worse", "worst",
    "frustrated", "angry", "annoyed", "disappointed", "unfair", "wrong",
    "never", "failure", "useless", "waste", "complain", "scam", "fraud",
    "cheat", "liar", "harassment", "threat",
    # Hindi / Hinglish
    "nahi", "nahin", "galat", "bura", "kharab", "mushkil", "dikkat",
    "pareshani", "paresaan", "gussa", "pareshan", "dhamki", "loot",
    "dhokha", "jhooth", "barbaad", "takleef", "dard", "thak", "tang",
}


# ─────────────────────────────────────────────────────────────
# STEP 1 — Text Sentiment Estimation
# ─────────────────────────────────────────────────────────────

def _estimate_text_sentiment(text: str) -> str:
    """Lightweight keyword-based sentiment scorer.

    Returns "positive", "negative", or "neutral".
    No external dependencies — simple token matching.
    """
    if not text:
        return "neutral"

    text_lower = text.lower().strip()
    words = set(text_lower.split())

    pos_count = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_lower)
    neg_count = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)

    if pos_count > neg_count and pos_count > 0:
        return "positive"
    elif neg_count > pos_count and neg_count > 0:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────────────────────
# STEP 2 — Acoustic Emotion Lookup
# ─────────────────────────────────────────────────────────────

def _get_acoustic_emotion(
    emotion_timeline: list[dict],
    t0: float,
    t1: float,
    speaker: str,
) -> str:
    """Look up acoustic emotion for a transcript segment from the emotion timeline.

    Finds the timeline entry whose time range overlaps the segment.
    Returns the emotion label, or "neutral" if none found.
    """
    best_match = "calm"
    best_overlap = 0.0

    for entry in emotion_timeline:
        entry_start = entry.get("timestamp_seconds", 0.0)
        entry_end = entry.get("end_seconds", entry_start + 0.5)

        # Check overlap
        overlap_start = max(t0, entry_start)
        overlap_end = min(t1, entry_end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap and entry.get("speaker", "") == speaker:
            best_overlap = overlap
            best_match = entry.get("emotion", "calm")

    return best_match


# ─────────────────────────────────────────────────────────────
# STEP 3 — Sarcasm Detection Engine
# ─────────────────────────────────────────────────────────────

_NEGATIVE_ACOUSTIC_EMOTIONS = {"angry", "frustrated"}
_FORMAT_TS = lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"


def detect_sarcasm(
    transcript_segments: list[dict],
    features: list[dict],
    emotion_timeline: list[dict],
) -> list[dict]:
    """Detect potential sarcasm using multimodal signal analysis.

    Sarcasm candidate criteria:
    1. Text sentiment is POSITIVE
    2. Acoustic emotion is FRUSTRATED or ANGRY
    3. Previous segment's text sentiment was NEGATIVE

    Returns:
        [{timestamp, timestamp_seconds, speaker, phrase, confidence,
          signals: {text_sentiment, acoustic_emotion, previous_context}}]
    """
    candidates = []

    for i, seg in enumerate(transcript_segments):
        text = seg.get("text", "")
        speaker = seg.get("speaker", seg.get("role", ""))
        t0 = seg.get("t0", 0.0)
        t1 = seg.get("t1", t0 + 1.0)

        # Signal 1: Text sentiment
        text_sentiment = _estimate_text_sentiment(text)
        if text_sentiment != "positive":
            continue

        # Signal 2: Acoustic emotion
        acoustic_emotion = _get_acoustic_emotion(emotion_timeline, t0, t1, speaker)
        if acoustic_emotion not in _NEGATIVE_ACOUSTIC_EMOTIONS:
            continue

        # Signal 3: Previous context sentiment
        previous_context = "neutral"
        if i > 0:
            prev_text = transcript_segments[i - 1].get("text", "")
            previous_context = _estimate_text_sentiment(prev_text)

        # All three must align for a sarcasm candidate
        if previous_context != "negative":
            # Relax: if acoustic is strongly negative, allow neutral context
            if acoustic_emotion == "angry":
                # Strong acoustic signal — allow even without negative context
                confidence_base = 0.4
            else:
                continue
        else:
            confidence_base = 0.6

        # Compute confidence (higher if all signals strongly agree)
        confidence = confidence_base
        # Boost if text has strong positive markers
        strong_positive = any(kw in text.lower() for kw in
                             ["great", "perfect", "wonderful", "bilkul", "bahut accha", "shandar"])
        if strong_positive:
            confidence += 0.15

        # Boost if acoustic is angry (stronger than just frustrated)
        if acoustic_emotion == "angry":
            confidence += 0.1

        confidence = round(min(confidence, 0.95), 2)

        candidates.append({
            "timestamp": f"[{_FORMAT_TS(t0)}]",
            "timestamp_seconds": round(t0, 2),
            "speaker": speaker,
            "phrase": text,
            "confidence": confidence,
            "signals": {
                "text_sentiment": text_sentiment,
                "acoustic_emotion": acoustic_emotion,
                "previous_context": previous_context,
            },
        })

    return candidates


# ─────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────

def analyze_sarcasm(
    transcript_segments: list[dict],
    features: list[dict],
    emotion_timeline: list[dict],
) -> dict:
    """Full sarcasm detection pipeline. Graceful failure guaranteed.

    Args:
        transcript_segments: ASR output [{speaker, t0, t1, text, role}].
        features: Acoustic feature windows from emotion.py.
        emotion_timeline: Emotion timeline from emotion.py.

    Returns:
        {possible_sarcasm_segments: [{timestamp, speaker, phrase, confidence, signals}]}
        Returns empty array on error.
    """
    empty_result = {"possible_sarcasm_segments": []}

    try:
        print("🎭 Running Sarcasm Detection (multimodal)...")

        candidates = detect_sarcasm(transcript_segments, features, emotion_timeline)

        print(f"   🎭 Sarcasm candidates: {len(candidates)}")

        return {"possible_sarcasm_segments": candidates}

    except Exception as e:
        logger.error(f"Sarcasm detection failed gracefully: {e}", exc_info=True)
        print(f"   ⚠️  Sarcasm detection error: {e}. Returning empty result.")
        return empty_result
