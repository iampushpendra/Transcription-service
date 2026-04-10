"""
Winning Patterns Insights Engine
Aggregates all processed calls and extracts conversion patterns using LLM.
Result is cached to outputs/winning_patterns.json and refreshed whenever new calls arrive.
"""

import json
import logging
import os
import glob
from datetime import datetime
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
CACHE_FILE = os.path.join(OUTPUTS_DIR, "winning_patterns.json")


def _collect_call_data() -> list[dict]:
    """Read all transcript.json files and extract signals relevant for pattern analysis."""
    calls = []
    for path in sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*/transcript.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)

            summary = d.get("summary") or {}
            triggers = d.get("trigger_phrases") or {}
            emotion = d.get("emotion_analysis") or {}
            metadata = d.get("metadata") or {}

            # Talk ratio
            segments = d.get("segments", [])
            agent_dur = sum(s.get("t1", 0) - s.get("t0", 0) for s in segments if s.get("speaker") == "agent")
            customer_dur = sum(s.get("t1", 0) - s.get("t0", 0) for s in segments if s.get("speaker") == "customer")
            total_dur = agent_dur + customer_dur
            talk_ratio = round(agent_dur / total_dur * 100) if total_dur > 0 else None

            calls.append({
                "call": metadata.get("original_filename", os.path.basename(os.path.dirname(path))),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "overview": summary.get("overview", ""),
                "customer_pain_points": summary.get("pain_points_detailed", []),
                "most_appealing_aspect": (summary.get("customer_info") or {}).get("most_appealing_aspect", ""),
                "moment_of_interest": (summary.get("customer_info") or {}).get("moment_of_interest", ""),
                "objection_handling": summary.get("objection_handling", {}),
                "next_steps": summary.get("next_steps", ""),
                "call_quality_score": summary.get("call_quality_score"),
                "buying_triggers": [
                    b for b in triggers.get("business_insights", [])
                    if b.get("category") == "buying_triggers"
                ],
                "hesitation_phrases": triggers.get("hesitation_phrases", []),
                "negative_triggers": triggers.get("negative_triggers", []),
                "escalation_detected": emotion.get("escalation_detected", False),
                "dominant_customer_emotion": emotion.get("dominant_customer_emotion", ""),
                "agent_calming_effectiveness": emotion.get("agent_calming_effectiveness"),
                "agent_talk_pct": talk_ratio,
            })
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")

    return calls


def _run_llm_analysis(calls: list[dict], api_key: str) -> dict:
    """Send aggregated call data to GPT-4o and extract winning patterns."""
    client = OpenAI(api_key=api_key, timeout=120.0)

    prompt = f"""You are an expert sales coach analyzing converted sales calls for FREED — India's debt relief platform (DRP/DCP/DEP programs).

All {len(calls)} calls below resulted in successful customer enrollment.

Your job: Extract the KEY PATTERNS that drove conversions. Structure your response as valid JSON with this exact shape:

{{
  "headline_insight": "One punchy sentence summarizing the #1 thing that closes deals at FREED",
  "total_calls_analyzed": {len(calls)},
  "patterns": [
    {{
      "id": "short_snake_case_id",
      "title": "Pattern Name",
      "description": "What the agent did and why it worked (2-3 sentences, actionable)",
      "frequency": "X/{len(calls)} calls",
      "frequency_count": X,
      "evidence_quotes": [
        {{"quote": "verbatim customer quote in original language", "call": "filename"}},
        {{"quote": "another verbatim quote", "call": "filename"}}
      ],
      "agent_technique": "The specific thing the agent said or did",
      "customer_trigger": "What the customer said/felt that this technique addressed",
      "coaching_tip": "One-line tip for agents to replicate this"
    }}
  ],
  "top_objections": [
    {{
      "objection": "The objection customers raised",
      "frequency": "X/{len(calls)} calls",
      "winning_response": "What the agent said that overcame it"
    }}
  ],
  "talk_ratio_insight": "Observation about agent vs customer talk time and its impact on conversion",
  "emotional_pattern": "Pattern in customer emotional journey across successful calls",
  "red_flags_avoided": ["Things agents did NOT do that would have killed the deal"],
  "generated_at": "{datetime.now().isoformat()}"
}}

Be specific. Use verbatim quotes. If you observe a pattern in fewer than 3 calls, still include it if it's highly significant.
Only output valid JSON — no markdown, no explanation outside the JSON.

CALL DATA:
{json.dumps(calls, ensure_ascii=False, indent=2)}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


def refresh_insights(api_key: str, force: bool = False) -> Optional[dict]:
    """
    Generate or return cached winning patterns.
    Re-runs if there are new calls since last cache write.
    Returns None if fewer than 2 calls exist or no API key.
    """
    if not api_key:
        logger.info("Insights: no OpenAI key, skipping.")
        return None

    calls = _collect_call_data()
    if len(calls) < 2:
        logger.info(f"Insights: only {len(calls)} call(s), need at least 2.")
        return None

    # Check if cache is still fresh (no new calls since last run)
    if not force and os.path.exists(CACHE_FILE):
        try:
            cache_mtime = os.path.getmtime(CACHE_FILE)
            newest_call = max(
                os.path.getmtime(p)
                for p in glob.glob(os.path.join(OUTPUTS_DIR, "*/transcript.json"))
            )
            cached = json.load(open(CACHE_FILE, encoding="utf-8"))
            if newest_call <= cache_mtime and cached.get("total_calls_analyzed") == len(calls):
                logger.info("Insights: cache is fresh, returning cached result.")
                return cached
        except Exception:
            pass  # Regenerate if anything is wrong with cache

    logger.info(f"Insights: analyzing {len(calls)} calls...")
    try:
        result = _run_llm_analysis(calls, api_key)
        result["total_calls_analyzed"] = len(calls)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Insights: saved to winning_patterns.json")
        return result
    except Exception as e:
        logger.error(f"Insights LLM failed: {e}", exc_info=True)
        # Return stale cache if available
        if os.path.exists(CACHE_FILE):
            return json.load(open(CACHE_FILE, encoding="utf-8"))
        return None
