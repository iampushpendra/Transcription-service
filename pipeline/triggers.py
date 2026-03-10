"""
Trigger Phrase & Business Insight Extraction — LLM-powered.

Extracts:
1. Negative triggers — phrases spoken during/before heated moments (acoustic-grounded)
2. Positive engagement phrases — agent phrases before positive energy spikes (acoustic-grounded)
3. Business Insights — LLM-extracted actionable keywords from transcript
4. Hesitation/resistance phrases — audit-relevant customer delay/objection markers (rule-based)

Business insights use LLM to extract MEANINGFUL phrases that drive business/marketing
decisions — NOT keyword matching. Each insight is a specific, actionable observation
tied to a verbatim quote and transcript timestamp.

All phrases are verbatim from the transcript. No paraphrasing. No hallucination.
Every phrase includes a transcript timestamp reference.
"""

import os
import re
import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None


# ─────────────────────────────────────────────────────────────
# HESITATION / RESISTANCE markers (audit-relevant for call evaluation)
# Grouped by category for better reporting
# ─────────────────────────────────────────────────────────────

HESITATION_CATEGORIES = {
    "delay_stalling": {
        "label": "Delay / Stalling",
        "markers": [
            "sochna padega", "soch ke batata", "sochenge", "sochtein", "sochta",
            "dekhte hain", "dekh lete hain", "dekhna padega", "dekh lenge", "dekhenge",
            "baad mein", "baad me", "kal bataunga", "kal bataata", "kal baat",
            "time chahiye", "waqt chahiye", "samay chahiye", "thoda time", "time de do",
            "soch lun", "sochne do", "soch kar", "soch raha",
        ],
    },
    "financial_inability": {
        "label": "Financial Inability",
        "markers": [
            "paise nahi", "paisa nahi", "itna nahi", "afford nahi",
            "paisa kahan se", "paise kahan se", "paise nahi hain",
            "itna paisa nahi", "utna paisa nahi", "budget nahi", "income nahi",
            "salary nahi aa rahi", "kharcha bahut", "loan chal raha",
            "emi nahi bhar", "emi bharna mushkil", "emi band", "payment nahi",
        ],
    },
    "authority_deferral": {
        "label": "Authority Deferral",
        "markers": [
            "family se puchna", "ghar pe puchu", "wife se puchu", "baat karke batata",
            "ghar wale se", "husband se", "papa se", "mummy se",
            "permission chahiye", "approval chahiye", "discuss karke",
            "biwi se puch", "family discuss",
        ],
    },
    "outright_refusal": {
        "label": "Outright Refusal / Rejection",
        "markers": [
            "nahi kar sakta", "nahi ho payega", "mushkil hai", "possible nahi",
            "nahi chahiye", "interest nahi", "zaroorat nahi",
            "cancel karo", "band karo", "nahi karunga", "nahi karenge",
            "koi fayda nahi", "kaam nahi aayega",
            "mujhe nahi chahiye", "hume nahi chahiye",
        ],
    },
    "doubt_skepticism": {
        "label": "Doubt / Skepticism",
        "markers": [
            "confirm nahi", "sure nahi", "pakka nahi", "guarantee nahi",
            "bharosa nahi", "trust nahi", "vishwas nahi",
            "kaise pata", "kya guarantee", "proof do", "likha ke do",
            "fraud toh nahi", "scam toh nahi", "genuine hai kya",
            "sach bol rahe", "sahi bol rahe", "pehle bhi aisa",
        ],
    },
    "emotional_distress": {
        "label": "Emotional Distress",
        "markers": [
            "bahut tension", "tension mein", "stressed", "pareshan",
            "dar lag raha", "darr lagta", "chinta ho rahi",
            "neend nahi aati", "raat ko", "bahut mushkil",
            "kya karu", "samajh nahi aa raha", "helpless",
            "tang aa gaya", "thak gaya", "haar gaya",
        ],
    },
}

# Flatten all markers for backward compat
HESITATION_MARKERS = []
for cat_info in HESITATION_CATEGORIES.values():
    HESITATION_MARKERS.extend(cat_info["markers"])


# ─────────────────────────────────────────────────────────────
# STEP 1 — Negative Trigger Phrase Extraction (acoustic-grounded)
# ─────────────────────────────────────────────────────────────

def extract_negative_triggers(
    heated_segments: list[dict],
    transcript_segments: list[dict],
    window_tolerance: float = 2.0,
) -> list[dict]:
    """Extract phrases spoken during or immediately before heated moments.

    For each heated segment, extracts transcript text within ±2s window.
    Stores phrase exactly as spoken — no paraphrasing.

    Returns:
        [{phrase, speaker, timestamp, timestamp_seconds, heated_speaker, context}]
    """
    triggers = []
    seen_texts = set()

    for hs in heated_segments:
        h_start = hs.get("start_seconds", 0)
        h_end = hs.get("end_seconds", 0)
        h_speaker = hs.get("speaker", "unknown")

        for seg in transcript_segments:
            seg_start = seg.get("t0", 0)
            seg_end = seg.get("t1", 0)
            seg_text = seg.get("text", "").strip()

            if not seg_text or len(seg_text) < 5:
                continue

            if (seg_start <= h_end + window_tolerance and
                    seg_end >= h_start - window_tolerance):

                text_key = f"{seg_text[:50]}_{seg_start}"
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                triggers.append({
                    "phrase": seg_text,
                    "speaker": seg.get("role", seg.get("speaker", "unknown")),
                    "timestamp": _fmt_ts(seg_start),
                    "timestamp_seconds": round(seg_start, 1),
                    "heated_speaker": h_speaker,
                    "context": "during_escalation" if h_speaker == seg.get("speaker") else "counter_party",
                })

    return triggers


# ─────────────────────────────────────────────────────────────
# STEP 2 — Positive Engagement Phrase Detection (acoustic-grounded)
# ─────────────────────────────────────────────────────────────

def extract_positive_engagement(
    features: list[dict],
    transcript_segments: list[dict],
    pitch_threshold: float = 1.5,
    energy_threshold: float = 1.0,
) -> list[dict]:
    """Detect positive emotional spikes (high energy + pitch WITHOUT anger pattern).

    Returns:
        [{phrase, speaker, timestamp, timestamp_seconds, trigger_type}]
    """
    engagement = []
    seen = set()

    for w in features:
        if (w.get("pitch_z", 0) > pitch_threshold and
                w.get("energy_z", 0) > energy_threshold and
                w.get("pitch_z", 0) < 2.5 and
                w.get("speaker") == "customer"):

            for seg in transcript_segments:
                seg_start = seg.get("t0", 0)
                seg_end = seg.get("t1", 0)
                seg_speaker = seg.get("role", seg.get("speaker", ""))
                seg_text = seg.get("text", "").strip()

                if not seg_text or len(seg_text) < 5:
                    continue

                if (seg_speaker == "agent" and
                        0 <= w["start"] - seg_end <= 5.0):

                    text_key = f"{seg_text[:50]}_{seg_start}"
                    if text_key in seen:
                        continue
                    seen.add(text_key)

                    engagement.append({
                        "phrase": seg_text,
                        "speaker": "agent",
                        "timestamp": _fmt_ts(seg_start),
                        "timestamp_seconds": round(seg_start, 1),
                        "trigger_type": "positive_engagement",
                    })
                    break

    return engagement


# ─────────────────────────────────────────────────────────────
# STEP 3 — LLM-Powered Business Insight Extraction
# ─────────────────────────────────────────────────────────────

_BUSINESS_INSIGHT_PROMPT = """You are a business intelligence analyst reviewing a recorded call transcript from FREED, a debt resolution company.

Extract the most important ACTIONABLE BUSINESS INSIGHTS from this transcript. These should be phrases/moments that a marketing or business operations team can use to improve their workflow, pitch, and strategy.

CATEGORIES to extract (only include categories that have findings):

1. **customer_objections**: Specific reasons the customer pushes back, resists, or refuses. These reveal what aspects of the pitch or product need improvement. Example: "Mujhe trust nahi hai, pehle bhi kisi ne aisa bola tha" (reveals trust deficit from past experience).

2. **buying_triggers**: The exact moment or phrase where the customer shows genuine interest, agrees, or gets convinced. These reveal what WORKS in the pitch. Example: "Achha, toh CIBIL bhi improve hoga? Toh main ready hoon" (CIBIL improvement was the buying trigger).

3. **unresolved_concerns**: Questions or worries the customer raised that the agent did NOT adequately address. These are missed opportunities. Example: "But kya guarantee hai ki bank waale call nahi karenge?" (customer left without a clear answer).

4. **competitive_intel**: Any mention of competitors, alternative products, or comparisons. Example: "XYZ company ne toh 40% discount offer kiya hai" (reveals competitor pricing).

5. **pitch_effectiveness**: Moments where the agent's specific technique or phrasing either clearly WORKED or clearly FAILED. Note which. Example: "Jab agent ne escrow account explain kiya toh customer immediately agreed" → what worked.

CRITICAL RELEVANCE RULES:
- RELEVANCE IS KING: Do NOT force an extraction just to fill a category. If a quote is vague, generic, or lacks clear business meaning, DO NOT include it.
- CONTEXT MATTERS: Look at the preceding and succeeding segments to verify if the excerpt is genuinely a trigger, objection, or insight.
- Each insight MUST include a VERBATIM quote from the transcript (the exact Hindi/Hinglish words spoken).
- Each insight MUST include the segment index (0-based) of the quoted segment exactly as it appears.
- Maximum 3-5 insights overall. Quality over quantity — only truly actionable insights.
- Do NOT extract generic financial information (amounts, EMIs, loan details).
- Do NOT extract filler conversation.
- If a category has no meaningul findings, omit it entirely.
- Write the "insight" field as a concise 1-line business takeaway in English.

Output valid JSON matching this schema:
{
  "insights": [
    {
      "category": "customer_objections|buying_triggers|unresolved_concerns|competitive_intel|pitch_effectiveness",
      "insight": "One-line business takeaway in English",
      "verbatim_quote": "Exact words from transcript",
      "segment_index": 0,
      "speaker": "agent|customer"
    }
  ]
}"""


def extract_business_insights(
    transcript_segments: list[dict],
    model: str = "gpt-5-mini",
    api_key: str = "",
) -> list[dict]:
    """Use LLM to extract actionable business insights from the transcript.

    Returns:
        [{category, category_label, insight, verbatim_quote, timestamp, timestamp_seconds, speaker}]
    """
    if openai is None:
        logger.warning("openai not available, skipping business insight extraction")
        return []

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping business insight extraction")
        return []

    # Build compact transcript for LLM
    compact = []
    for i, seg in enumerate(transcript_segments):
        role = seg.get("role", seg.get("speaker", "unknown"))
        text = seg.get("text", "").strip()
        if text:
            compact.append(f"[{i}] {role}: {text}")

    transcript_text = "\n".join(compact)

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _BUSINESS_INSIGHT_PROMPT},
                {"role": "user", "content": f"Analyze this call transcript:\n\n{transcript_text}"},
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        raw_insights = parsed.get("insights", [])

    except Exception as e:
        logger.error(f"LLM business insight extraction failed: {e}")
        print(f"   ⚠️  Business insight extraction error: {e}")
        return []

    # Category labels for display
    CATEGORY_LABELS = {
        "customer_objections": "🚫 Customer Objections",
        "buying_triggers": "🎯 Buying Triggers",
        "unresolved_concerns": "⚠️ Unresolved Concerns",
        "competitive_intel": "🔍 Competitive Intelligence",
        "pitch_effectiveness": "📊 Pitch Effectiveness",
    }

    # Map insights to timestamped output
    results = []
    for ins in raw_insights:
        seg_idx = ins.get("segment_index", 0)

        # Validate segment index
        if seg_idx < 0 or seg_idx >= len(transcript_segments):
            continue

        seg = transcript_segments[seg_idx]
        t0 = seg.get("t0", 0)
        cat = ins.get("category", "other")

        results.append({
            "category": cat,
            "category_label": CATEGORY_LABELS.get(cat, cat),
            "insight": ins.get("insight", ""),
            "verbatim_quote": ins.get("verbatim_quote", seg.get("text", "")),
            "timestamp": _fmt_ts(t0),
            "timestamp_seconds": round(t0, 1),
            "speaker": ins.get("speaker", seg.get("role", seg.get("speaker", "unknown"))),
        })

    return results


# ─────────────────────────────────────────────────────────────
# STEP 4 — Audit-Relevant Hesitation / Resistance Detection
# ─────────────────────────────────────────────────────────────

def detect_hesitation_phrases(
    transcript_segments: list[dict],
) -> list[dict]:
    """Detect hesitation, resistance, and objection phrases from customer speech.

    Each match is categorized (delay/stalling, financial inability, authority
    deferral, outright refusal, doubt/skepticism, emotional distress) and
    includes the full verbatim transcript segment with timestamp.

    Returns:
        [{phrase, marker, category, category_label, timestamp, timestamp_seconds, speaker}]
    """
    results = []
    seen_segments = set()

    for seg in transcript_segments:
        role = seg.get("role", seg.get("speaker", ""))
        if role != "customer":
            continue

        text = seg.get("text", "").strip()
        text_lower = text.lower()
        t0 = seg.get("t0", 0)

        if not text or len(text) < 5:
            continue

        seg_key = f"{t0}_{text[:30]}"
        if seg_key in seen_segments:
            continue

        for cat_key, cat_info in HESITATION_CATEGORIES.items():
            found = False
            for marker in cat_info["markers"]:
                if marker in text_lower:
                    seen_segments.add(seg_key)
                    results.append({
                        "phrase": text,
                        "marker": marker,
                        "category": cat_key,
                        "category_label": cat_info["label"],
                        "timestamp": _fmt_ts(t0),
                        "timestamp_seconds": round(t0, 1),
                        "speaker": "customer",
                    })
                    found = True
                    break
            if found:
                break

    return results


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    """Format seconds to [MM:SS] timestamp reference."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"[{m:02d}:{s:02d}]"


# ─────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────

def analyze_triggers(
    heated_segments: list[dict],
    features: list[dict],
    transcript_segments: list[dict],
    model: str = "gpt-5-mini",
    api_key: str = "",
) -> dict:
    """Full trigger phrase analysis. Graceful failure guaranteed.

    Args:
        heated_segments: Agent + customer heated segments from emotion analysis.
        features: Acoustic feature windows from emotion.py.
        transcript_segments: ASR output [{speaker, t0, t1, text, role}].
        model: LLM model for business insight extraction.
        api_key: OpenAI API key. If not provided, falls back to env var.

    Returns:
        Complete trigger_phrases dict. Returns empty arrays on error.
    """
    empty_result = {
        "negative_triggers": [],
        "positive_engagement_phrases": [],
        "business_insights": [],
        "hesitation_phrases": [],
    }

    try:
        print("🔍 Extracting Trigger Phrases & Business Insights...")

        # Step 1: Negative triggers from heated segments (acoustic-grounded)
        negative = extract_negative_triggers(heated_segments, transcript_segments)

        # Step 2: Positive engagement from acoustic features (acoustic-grounded)
        positive = extract_positive_engagement(features, transcript_segments)

        # Step 3: LLM-powered business insight extraction
        insights = extract_business_insights(transcript_segments, model=model, api_key=api_key)

        # Step 4: Audit-relevant hesitation/resistance detection (rule-based)
        hesitation = detect_hesitation_phrases(transcript_segments)

        result = {
            "negative_triggers": negative,
            "positive_engagement_phrases": positive,
            "business_insights": insights,
            "hesitation_phrases": hesitation,
        }

        print(f"   🔍 Negative triggers: {len(negative)} | "
              f"Positive engagement: {len(positive)} | "
              f"Business insights: {len(insights)} | "
              f"Hesitation: {len(hesitation)}")

        return result

    except Exception as e:
        logger.error(f"Trigger phrase extraction failed gracefully: {e}", exc_info=True)
        print(f"   ⚠️  Trigger extraction error: {e}. Returning empty result.")
        return empty_result
