"""
Transcript reconstruction: merge segments, deduplicate, summarize.
"""

import re

from .utils import fmt
from .config import PipelineConfig

def clean_text(text: str, hallucinations: list[str]) -> str:
    """Filter meaningless segments generated as ASR hallucinations."""
    lower_t = text.lower()
    for h in hallucinations:
        if h in lower_t:
            return ""
    return text.strip()


def normalize_text(text: str, replacements: dict[str, str]) -> str:
    """Replace common mis-transcriptions with correct loan terminology."""
    if not text:
        return text
    
    # 1. Simple word replacement using regex to match whole words/phrases case-insensitively
    normalized = text
    for wrong, right in replacements.items():
        # Match boundaries to avoid replacing substrings inside other words
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        normalized = pattern.sub(right, normalized)

    # 2. Advanced Phonetic/Fuzzy Matching 
    import difflib
    
    # Extract unique target words from the replacements to fuzzy-match against 
    # (avoiding multi-word phrases for simple phonetic algo, focusing on single crucial words)
    target_keywords = set()
    for v in replacements.values():
        for w in v.split():
            if len(w) > 3: # Ignore small connecting words
                target_keywords.add(w)
                
    words = normalized.split()
    for i, w in enumerate(words):
        clean_w = re.sub(r'[^\w\s]', '', w).lower()
        if not clean_w or len(clean_w) <= 3:
            continue
            
        for target in target_keywords:
            clean_target = target.lower()
            
            # Simple heuristic: word must share at least the same starting letter usually, 
            # or be extremely similar overall to avoid catastrophic mis-replacements
            if clean_w[0] == clean_target[0] or clean_w[-1] == clean_target[-1]:
                
                # difflib gives a ratio 0.0 to 1.0 of sequence similarity
                ratio = difflib.SequenceMatcher(None, clean_w, clean_target).ratio()
                
                # If they are very similar (e.g. 85%+ match) AND essentially the same length
                if ratio > 0.849 and abs(len(clean_w) - len(clean_target)) <= 1:
                    
                    # Match case of original target
                    words[i] = w.replace(clean_w, target) if w.islower() else w.replace(clean_w.capitalize(), target)
                    break 

    return " ".join(words)


def reconstruct(
    segments: list[dict], cfg: PipelineConfig
) -> list[dict]:
    """
    Merge adjacent same-speaker segments and deduplicate overlapping text.

    Returns cleaned-up transcript segments.
    """
    if not segments:
        return []

    # First pass: clean out hallucinations, normalize terminology, and filter empty texts
    cleaned_segs = []
    for seg in segments:
        text = clean_text(seg["text"], cfg.hallucinations)
        text = normalize_text(text, cfg.term_replacements)
        seg["text"] = text
        if seg["text"]:
            cleaned_segs.append(seg)

    if not cleaned_segs:
        return []

    sorted_s = sorted(cleaned_segs, key=lambda x: x["t0"])
    merged = [sorted_s[0].copy()]

    for seg in sorted_s[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and (
            seg["t0"] - prev["t1"] <= cfg.merge_gap_seconds or seg["t0"] < prev["t1"]
        ):
            prev["t1"] = max(prev["t1"], seg["t1"])
            # Strict overlap text deduplication is flawed without acoustic alignment, 
            # so we just concatenate sentences with a space if they aren't identical.
            if seg["text"] not in prev["text"]:
                if prev["text"].endswith((".", "?", "!")):
                    prev["text"] += " " + seg["text"]
                else:
                    prev["text"] += " " + seg["text"]
        else:
            merged.append(seg.copy())

    # Clean whitespace
    for seg in merged:
        seg["text"] = " ".join(seg["text"].split())

    print(f"🔄 Reconstructed: {len(sorted_s)} → {len(merged)} segments")
    return merged


def correct_transcript_llm(transcript: list[dict], cfg: PipelineConfig) -> list[dict]:
    """Pass transcript through OpenAI structured output to correct domain Hinglish homophones contextually."""
    import json
    import openai
    from pydantic import BaseModel, Field
    from tqdm import tqdm
    
    print("🤖 Running LLM Contextual Transcript Correction (Phase 17)...")
    
    if not transcript:
        return []
        
    client = openai.OpenAI(
        api_key=cfg.openai_api_key,
        timeout=cfg.openai_timeout_s,
        max_retries=cfg.openai_max_retries,
    )
    
    class Segment(BaseModel):
        speaker: str
        t0: float
        t1: float
        text: str = Field(description="The contextually corrected Hinglish text.")
        
    class CorrectedTranscript(BaseModel):
        segments: list[Segment]
        
    system_prompt = (
        "You are an expert transcriber of Hindi-Hinglish financial advisory and debt collection telephonic calls.\n"
        "Your task is to fix phonetic ASR mis-transcriptions (homophones) in the provided transcript segments.\n"
        "The audio is from FREED, dealing with Debt Consolidation, Debt Resolution, CIBIL scores, NACH, NBFCs, settling loans, DRA certificates, and EMIs.\n\n"
        "CRITICAL RULES:\n"
        "1. DO NOT translate the Hinglish into English. Keep the exact mixed Hindi-English dialect.\n"
        "2. DO NOT summarize, hallucinate, or alter the meaning.\n"
        "3. EXTREMELY IMPORTANT: PRESERVE ALL FINANCIAL VALUES EXACTLY. Do not hallucinate or change the mathematical data.\n"
        "3a. COMBINE FRAGMENTED NUMBERS: When a large number is spoken in scattered Hinglish fragments (e.g. reading '1505' as 'pandhrah 100 paanch', or interpreting 'two point five lakhs' into '2,50,000'), mathematically combine them into a single, unified, comma-separated numeric string formatted according to the Indian numbering system (e.g. '1,505' or '2.5 Lakhs') as long as the total quantitative value remains 100% identical. Apply this strictly to ALL mathematical phrases spoken in the transcript.\n"
        "3b. DATES/DAYS: Spoken dates like 'bees taareekh' → '20 taareekh', 'pandrah June' → '15 June', 'teesri ko' → '3rd ko'. Always write the numeral, never the spelled-out word.\n"
        "3c. TIME REFERENCES: 'teen baje' → '3 baje', 'sava paanch' → '5:15', 'dhai baje' → '2:30', 'poune chaar' → '3:45'. Express as digits.\n"
        "3d. FINANCIAL AMOUNTS: 'do lakh pachaas hazaar' → '2,50,000', 'ek hazaar paanch sau' → '1,500', 'unees sau' → '1,900'. Always use Indian comma notation (1,00,000 not 100,000).\n"
        "3e. PERCENTAGES/RATES: 'saath percent' → '60%', 'unhattar percent' → '69%', 'byaaj' → 'interest'. Preserve decimal rates exactly as spoken.\n"
        "3f. ACCOUNT/PHONE NUMBERS: Sequences of individual spoken digits must be preserved exactly as a single number string. Do NOT split, re-order, or round them.\n"
        "3g. GOLDEN RULE: If you are uncertain about ANY number, percentage, date, or amount — preserve the original spoken form EXACTLY. Never round, approximate, or hallucinate figures. A wrong number in a fintech pipeline can cause catastrophic failure.\n"
        "4. ONLY fix contextual homophone errors targeting our domain vocabulary (e.g. if the agent says 'fees se baat kar rahi hoon', correct it to 'FREED se baat kar rahi hoon'. Or 'civil' -> 'CIBIL', 'knocks' -> 'NOC'). If 'fees' is used correctly (e.g. 'processing fees'), DO NOT change it.\n"
        "5. Keep the exact same number of segments and their exact original 'speaker', 't0', and 't1' values.\n\n"
        "Output valid JSON matching this schema: {\"segments\": [{\"speaker\": \"...\", \"t0\": 0.0, \"t1\": 0.0, \"text\": \"...\"}]}"
    )
    
    batch_size = 15  # smaller batches to reliably complete within OpenAI timeout
    corrected_results = []
    
    for i in tqdm(range(0, len(transcript), batch_size), desc="   LLM Correction"):
        batch = transcript[i:i+batch_size]
        
        # Prepare small JSON for the LLM to read easily
        clean_batch = [{"speaker": s["speaker"], "t0": s["t0"], "t1": s["t1"], "text": s["text"]} for s in batch]
        user_prompt = f"Correct these transcript segments:\n\n{json.dumps(clean_batch, indent=2)}"
        
        try:
            response = client.chat.completions.create(
                model=cfg.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            corrected_batch = parsed.get("segments", [])
            
            # Enforce length safety
            if len(corrected_batch) == len(batch):
                for original, corrected in zip(batch, corrected_batch):
                    original["text"] = corrected.get("text", original["text"])
                    corrected_results.append(original)
            else:
                print("   ⚠️  LLM Correction returned mismatched segment count, keeping original batch.")
                corrected_results.extend(batch)
                
        except Exception as e:
            print(f"   ⚠️  LLM Correction failed for batch, keeping original. Error: {e}")
            corrected_results.extend(batch)
            
    # Clean up whitespace again just in case
    for seg in corrected_results:
        seg["text"] = " ".join(seg["text"].split())
        
    return corrected_results


def rephrase_transcript_llm(transcript: list[dict], cfg: PipelineConfig) -> list[dict]:
    """Rephrase transcript segments for improved readability while preserving meaning.
    
    This is a light-touch rephrasing pass that:
    - Cleans up garbled/noisy ASR fragments
    - Fixes grammar in Hindi-English code-mixed speech
    - Makes sentences flow naturally
    - Preserves all financial data, names, dates, commitments EXACTLY
    - Stores original text in 'original_text' field (non-destructive)
    """
    import json
    import openai
    from pydantic import BaseModel, Field
    from tqdm import tqdm
    
    print("✏️  Running LLM Transcript Rephrasing (Phase 28)...")
    
    if not transcript:
        return []
    
    client = openai.OpenAI(
        api_key=cfg.openai_api_key,
        timeout=cfg.openai_timeout_s,
        max_retries=cfg.openai_max_retries,
    )
    
    class RephrasedSegment(BaseModel):
        speaker: str
        t0: float
        t1: float
        text: str = Field(description="The rephrased, more readable version of the segment text.")
    
    class RephrasedTranscript(BaseModel):
        segments: list[RephrasedSegment]
    
    system_prompt = (
        "You are an expert editor of Hindi-Hinglish financial advisory call transcripts from FREED, "
        "a leading Indian debt resolution and consolidation company.\n\n"
        
        "ABOUT FREED (use this context to understand domain vocabulary):\n"
        "- FREED offers Debt Resolution Program (DRP) for customers who have defaulted on unsecured loans.\n"
        "- FREED offers Debt Consolidation Program (DCP) for customers struggling with multiple EMIs.\n"
        "- FREED Shield: harassment protection service — customers upload proof of recovery harassment, "
        "complaints auto-escalate to lenders.\n"
        "- Credit Insight (CI): free credit evaluation and monthly CIBIL score refresh.\n"
        "- Escrow Account: FREED opens a savings account where customers deposit monthly savings; "
        "FREED uses these funds to negotiate lump-sum settlements with banks.\n"
        "- Key terms: CIBIL (credit score bureau), NOC (No Objection Certificate from bank after settlement), "
        "DRA (Debt Recovery Agent certificate), NACH (National Automated Clearing House mandate), "
        "FOIR (Fixed Obligation to Income Ratio), NPA (Non-Performing Asset), "
        "settlement letter, one-time settlement (OTS), credit report, bureau (Experian/CIBIL/Equifax).\n"
        "- Customer journey groups: NTC (New to Credit), DEP (Debt Evaluation Plan eligible), "
        "DCP eligible, DRP eligible, with eligibility based on CIBIL score, active loans, FOIR, and delinquency.\n"
        "- Agents typically explain: file creation, credit report download via bureau call/OTP, "
        "savings program duration (6/12 months), upfront processing fee, settlement estimate ranges, "
        "interest/penalty handling post-enrollment, agreement and DRA certificate provision.\n\n"
        
        "YOUR TASK:\n"
        "Rephrase each transcript segment to be MORE READABLE while preserving the EXACT MEANING.\n\n"
        
        "RULES (STRICT):\n"
        "1. LIGHT TOUCH ONLY — improve clarity, fix garbled/noisy ASR fragments, smooth grammar. "
        "Do NOT rewrite from scratch or change the conversational tone.\n"
        "2. PRESERVE CODE-SWITCHING — if the speaker mixes Hindi and English, keep that mix. "
        "Do NOT translate Hindi parts to English or vice versa. Only clean up unintelligible noise.\n"
        "3. FINANCIAL DATA IS SACRED — every number, amount, percentage, date, phone number, "
        "account number MUST remain EXACTLY the same. Do NOT round, approximate, or reformat numbers.\n"
        "4. PRESERVE SPEAKER INTENT — if the customer says 'haan' (yes), keep that affirmation. "
        "If they express doubt, keep the doubt. Do NOT make hesitant speech sound confident or vice versa.\n"
        "5. FIX OBVIOUS ASR ERRORS — common patterns:\n"
        "   - 'fees' → 'FREED' (when referring to the company, NOT financial fees)\n"
        "   - 'civil' → 'CIBIL', 'knocks' → 'NOC', 'nacch' → 'NACH'\n"
        "   - 'experience' → 'Experian' (when referring to the credit bureau)\n"
        "   - Garbled repetitions like 'haan haan school se yaar thik' → clean up to the intended meaning\n"
        "   - 'aadhaar card' when context is 'aadha' (half) → fix to contextually correct word\n"
        "6. MERGE FRAGMENTED SPEECH — if the ASR split one natural sentence across the segment, "
        "make the segment's text flow as a complete thought within its own boundary.\n"
        "7. VERY SHORT SEGMENTS (fillers like 'Haan', 'Ji', 'Okay', 'Hmm', 'Thik hai') — "
        "keep them exactly as-is. Do not expand filler segments.\n"
        "8. Keep the EXACT SAME number of segments with EXACT SAME speaker, t0, t1 values.\n\n"
        
        "Output valid JSON matching: {\"segments\": [{\"speaker\": \"...\", \"t0\": 0.0, \"t1\": 0.0, \"text\": \"...\"}]}"
    )
    
    batch_size = 30  # slightly smaller batches to give model room for reasoning
    rephrased_results = []
    
    for i in tqdm(range(0, len(transcript), batch_size), desc="   LLM Rephrase"):
        batch = transcript[i:i+batch_size]
        
        # Prepare clean JSON for the LLM
        clean_batch = [{"speaker": s["speaker"], "t0": s["t0"], "t1": s["t1"], "text": s["text"]} for s in batch]
        user_prompt = f"Rephrase these transcript segments for better readability:\n\n{json.dumps(clean_batch, indent=2, ensure_ascii=False)}"
        
        try:
            response = client.chat.completions.create(
                model=cfg.rephrase_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            rephrased_batch = parsed.get("segments", [])
            
            # Enforce length safety — must return same number of segments
            if len(rephrased_batch) == len(batch):
                for original, rephrased in zip(batch, rephrased_batch):
                    # Store original text before overwriting
                    original["original_text"] = original["text"]
                    original["text"] = rephrased.get("text", original["text"])
                    rephrased_results.append(original)
            else:
                print(f"   ⚠️  LLM Rephrase returned {len(rephrased_batch)} segments (expected {len(batch)}), keeping originals.")
                for seg in batch:
                    seg["original_text"] = seg["text"]  # still store original_text
                    rephrased_results.append(seg)
                    
        except Exception as e:
            print(f"   ⚠️  LLM Rephrase failed for batch, keeping original. Error: {e}")
            for seg in batch:
                seg["original_text"] = seg["text"]
                rephrased_results.append(seg)
    
    # Clean up whitespace
    for seg in rephrased_results:
        seg["text"] = " ".join(seg["text"].split())
        
    # Stats
    changed = sum(1 for s in rephrased_results if s.get("original_text", "") != s["text"])
    print(f"   ✏️  Rephrased {changed}/{len(rephrased_results)} segments ({changed/max(len(rephrased_results),1)*100:.0f}%)")
    
    return rephrased_results

def summarize_customer(
    transcript: list[dict], duration: float, top_n: int = 15
) -> dict:
    """Extractive summary from customer speech segments."""
    cust_texts = [
        s["text"]
        for s in transcript
        if s["speaker"] == "customer" and s["text"].strip()
    ]

    if not cust_texts:
        print("   ⚠️  No customer speech")
        return {"summary": "No customer speech"}

    full = " ".join(cust_texts)
    sents = [
        s.strip() for s in re.split(r"[.!?\u0964\n]+", full) if len(s.strip()) > 15
    ]

    if not sents:
        return {"text": full[:500], "segs": len(cust_texts)}

    keywords = [
        "loan", "debt", "payment", "amount", "settle", "emi", "paisa",
        "rupees", "bank", "credit", "pay", "problem", "issue", "help",
        "nahi", "haan",
    ]

    scored = [
        (
            0.6 * min(len(s.split()) / 20, 1.0)
            + 0.4 * sum(1 for k in keywords if k in s.lower()) / len(keywords),
            s,
        )
        for s in sents
    ]
    scored.sort(reverse=True)

    summary = " | ".join(s for _, s in scored[:top_n])
    cd = sum(
        s["t1"] - s["t0"] for s in transcript if s["speaker"] == "customer"
    )

    print(
        f"📊 {len(cust_texts)} segs | {len(full.split())} words | "
        f"{cd:.1f}s ({cd / duration * 100:.1f}%)"
    )
    print(f"📝 {summary[:300]}")

    return {
        "text": summary,
        "customer_segments": len(cust_texts),
        "customer_words": len(full.split()),
        "customer_s": round(cd, 2),
        "customer_pct": round(cd / duration * 100, 1),
    }


def format_transcript_preview(transcript: list[dict], max_segs: int = 10) -> str:
    """Pretty print first N segments of transcript."""
    lines = []
    for seg in transcript[:max_segs]:
        label = "🔵 AGENT" if seg.get("speaker", "unknown") == "agent" else "🟢 CUSTOMER"
        preview = seg.get("text", "")[:120] + ("..." if len(seg.get("text", "")) > 120 else "")
        t0 = seg.get("t0", 0.0)
        t1 = seg.get("t1", 0.0)
        lines.append(f"\n[{fmt(t0)} → {fmt(t1)}] {label}")
        lines.append(f"  {preview}")
    return "\n".join(lines)


def summarize_call_structured(
    transcript: list[dict],
    cfg: PipelineConfig,
    emotion_analysis: dict | None = None,
    trigger_phrases: dict | None = None,
) -> dict:
    """Generate a structured JSON summary of the call using an LLM.

    Accepts optional emotion_analysis and trigger_phrases dicts so the LLM
    can use acoustic + behavioral signals to ground its evidence fields.
    """
    import json
    import openai
    from pydantic import BaseModel, Field

    print("🤖 Generating Structured LLM Summary...")

    if not transcript:
        return {"error": "Empty transcript"}

    # Build transcript text for the prompt
    lines = []
    for seg in transcript:
        role_label = seg.get("role", seg.get("speaker", "Unknown")).upper()
        t0 = seg.get('t0', 0.0)
        timestamp = fmt(t0)[:8]
        lines.append(f"[{timestamp}] {role_label}: {seg.get('text', '')}")
    transcript_text = "\n".join(lines)

    # ── Build optional Acoustic & Behavioral Context block ─────────────────
    def _build_context_block(ea: dict | None, tp: dict | None) -> str:
        if not ea and not tp:
            return ""
        parts: list[str] = ["", "=== ACOUSTIC & BEHAVIORAL CONTEXT ==="]

        if ea:
            dom_agent = ea.get("dominant_agent_emotion", "unknown")
            dom_cust  = ea.get("dominant_customer_emotion", "unknown")
            vol       = ea.get("emotional_volatility_score")
            switches  = ea.get("emotion_switch_count", 0)
            peak      = ea.get("highest_emotional_intensity_moment")

            parts.append(
                f"Dominant Emotions — Agent: {dom_agent} | Customer: {dom_cust}"
            )
            if vol is not None:
                parts.append(
                    f"Emotional Volatility: {round(vol * 100)}% ({switches} state switches)"
                )
            if peak:
                ts = peak.get("timestamp", "")
                spk = peak.get("speaker", "")
                emo = peak.get("emotion_label", peak.get("emotion", ""))
                parts.append(f"Peak Intensity: {emo} @ {ts} ({spk})")

            # Heated escalation moments (top 3)
            heated_all = (
                ea.get("agent_heated_segments", [])
                + ea.get("customer_heated_segments", [])
            )
            heated_all.sort(key=lambda h: h.get("energy_z", 0) + h.get("pitch_z", 0), reverse=True)
            if heated_all:
                parts.append("Heated Escalation Moments:")
                for h in heated_all[:3]:
                    ts_h = h.get("timestamp", "")
                    spk_h = h.get("speaker", "")
                    ez = round(h.get("energy_z", 0), 2)
                    pz = round(h.get("pitch_z", 0), 2)
                    parts.append(f"  - {ts_h} {spk_h.upper()} (energy_z={ez}, pitch_z={pz})")

        if tp:
            # Hesitation phrases (top 5, grouped by category)
            hes = tp.get("hesitation_phrases", [])
            if hes:
                from collections import defaultdict
                by_cat: dict = defaultdict(list)
                for h in hes:
                    by_cat[h.get("category_label", "Hesitation")].append(h)
                parts.append("Customer Hesitation Signals:")
                for cat, items in list(by_cat.items())[:3]:
                    for item in items[:2]:
                        ts_h = item.get("timestamp", "")
                        phrase = (item.get("phrase", "") or "")[:70]
                        parts.append(f"  - {ts_h} [{cat}] \"{phrase}\"")

            # Business insight signals (objections + buying triggers)
            insights = tp.get("business_insights", [])
            signal_cats = {"customer_objections", "buying_triggers", "unresolved_concerns"}
            signals = [i for i in insights if i.get("category") in signal_cats]
            if signals:
                parts.append("LLM-Verified Business Signals:")
                for sig in signals[:5]:
                    cat_label = sig.get("category_label", sig.get("category", ""))
                    ts_s = sig.get("timestamp", "")
                    quote = (sig.get("verbatim_quote", "") or "")[:80]
                    insight = sig.get("insight", "")
                    parts.append(f"  - {cat_label} {ts_s}: \"{quote}\" — {insight}")

        return "\n".join(parts)

    context_block = _build_context_block(emotion_analysis, trigger_phrases)
    has_context = bool(context_block.strip())

    # ── System prompt ───────────────────────────────────────────────────────
    context_instruction = (
        "\n\nACOUSTIC CONTEXT USAGE (CRITICAL):\n"
        "You are provided with an '=== ACOUSTIC & BEHAVIORAL CONTEXT ===' block above the transcript.\n"
        "- You MUST cross-reference this block when filling ALL evidence fields "
        "(pain_point_evidence, state_of_mind_evidence, conversion_evidence, customer_intent_signals).\n"
        "- When citing a hesitation signal or heated moment, include its timestamp and what it reveals.\n"
        "- Dominant emotions and volatility score MUST inform the customer_state_of_mind_category choice.\n"
        "- Business signals (objections/triggers) MUST be reflected in conversion_analysis.\n"
        "- Keep all answers crisp — enrich with acoustic evidence, do NOT expand word count unnecessarily."
    ) if has_context else ""

    system_prompt = (
        "Below are notes from different sections of one long Hindi financial advisory call.\n\n"
        "(prerequisite context: \n"
        "the financial advisor is from FREED offering \n"
        "Debt Consolidation and Debt Resolution)\n\n"
        "Your task:\n"
        "Create a detailed call brief that can be handed to another sales representative.\n\n"
        "REQUIREMENTS:\n"
        "- Do NOT hallucinate\n"
        "- Do NOT assume missing details\n"
        "- Do NOT speculate or add fluff\n"
        "- Preserve all financial numbers exactly\n"
        "- If unclear, write \"Not clearly specified\"\n"
        "- Professional English only\n"
        "- Every section must be CRISP, CONCISE, and FACTUAL\n"
        "- Use bullet points (- ) for every multi-item field — NEVER use paragraphs\n\n"
        "FORMATTING RULES (CRITICAL):\n"
        "- For EVERY field that has multiple points, items, or pieces of information, format each as a SEPARATE bullet point starting with '- '.\n"
        "- Use a newline (\\n) before each bullet point so they appear as separate lines.\n"
        "- NEVER pack multiple distinct points into a single long paragraph. Each separate fact, detail, or observation gets its own line.\n"
        "- For 'main_points_pitched', 'resolved_and_unresolved_questions', 'details_shared', 'benefits_mentioned', 'conditions_mentioned', 'customer_intent_signals', 'major_keywords' — you MUST use bullet points.\n\n"
        "OVERVIEW RULES (CRITICAL):\n"
        "- The 'overview' field MUST explicitly list all customer pain points as a bullet sub-list.\n"
        "- Format: Start with a 1-2 sentence call summary, then:\\n- Pain Point 1\\n- Pain Point 2\\n...\n"
        "- If no pain points are identified, explicitly state: 'No specific pain points mentioned in the call.'\n\n"
        "NUMERIC DEDUPLICATION (CRITICAL):\n"
        "- When the same debt/loan amount is mentioned multiple times in the call (e.g., by both agent and customer), count it ONLY ONCE at the highest confirmed value.\n"
        "- For 'total_identified_debt_inr': Sum UNIQUE loan accounts by their outstanding amounts. If a customer has 3 loans with 3 different outstandings, add them once each. Do NOT add the same loan's amount multiple times because it was discussed at different points.\n"
        "- If the agent mentions '₹5 lakh' and the customer also confirms '₹5 lakh' for the same loan — that is ONE occurrence of ₹5,00,000, NOT two.\n"
        "- For 'other_numbers_with_context': Only list numbers that are NOT already captured in loan_amounts, interest_rates, emis, tenure, or fees_or_charges. Avoid repeating the same figure across fields.\n\n"
        "The brief MUST include:\n"
        "1. Overview of Discussion\n"
        "2. Product/Service Explained\n"
        "3. Financial Details Discussed\n"
        "5. Detailed answers to:\n"
        "   a. Biggest pain point of the customer\n"
        "   b. Which aspect of the program appealed most and when\n"
        "   c. Main points pitched\n"
        "   d. Customer questions resolved/unresolved\n"
        "   e. Customer state of mind\n\n"
        "If information is missing, explicitly write:\n"
        "\"No information mentioned in the call.\"\n"
        "- CITATIONS: When referencing a specific detail (number, keyword, observation), include the exact transcript timestamp as a standalone reference like `[MM:SS]` or `[HH:MM:SS]` immediately after the statement or clause it supports. Use the timestamps from the provided notes, do NOT invent timestamps. Example: 'The total debt is 9,58,730 rupees [00:00:14]'.\n"
        "- SPEAKER ATTRIBUTION: You MUST carefully read whether a line was spoken by the AGENT or the CUSTOMER. Do NOT falsely attribute Agent statements (e.g. pitch points, settlement rules) as being said or requested by the Customer. When making a citation, explicitly mention who said it if relevant (AGENT) or (CUSTOMER).\n"
        "6. Conversion & Dropoff Analysis: Did the customer enroll or agree to proceed? If not, what specific objections, hesitations, or moments caused the dropoff? What could the agent have done differently to convert? Include precise timestamps.\n"
        "7. Call Categories with Evidence: For each classification (pain point, sentiment, lead probability), you MUST also provide an evidence field citing the specific conversation moments, quotes, and timestamps that justify the classification. Do not just classify — explain WHY with references.\n"
        "8. Major Keywords: Each keyword must include a timestamp and a brief one-line context explaining its significance. Format: 'Keyword [MM:SS] — Brief explanation of why this keyword matters in context.'"
        + context_instruction
    )

    user_prompt = f"Notes:\n{context_block}\n\n{transcript_text}" if has_context else f"Notes:\n{transcript_text}"

    class ProductServiceExplained(BaseModel):
        details_shared: str = Field(description="Exact details shared")
        benefits_mentioned: str
        conditions_mentioned: str

    class FinancialDetails(BaseModel):
        loan_amounts: str = Field(description="Loan amounts discussed")
        interest_rates: str
        emis: str
        tenure: str
        fees_or_charges: str = Field(description="Fees or charges mentioned")
        other_numbers_with_context: str = Field(description="Any other numbers (with context)")

    class CustomerAnalysis(BaseModel):
        biggest_pain_point: str = Field(description="The biggest pain point of the customer. Include verbatim quotes + timestamps. Cross-reference hesitation signals from the ACOUSTIC CONTEXT if provided. Use bullet points (- ) for multiple aspects.")
        most_appealing_aspect_and_moment_of_interest: str = Field(description="Which aspect of our program appeals to the customer the most, at which moment did the customer show interest? Cite buying trigger signals from ACOUSTIC CONTEXT if provided. Separate distinct moments with bullet points (- ).")
        main_points_pitched: str = Field(description="What are the main points pitched. MUST use bullet points — each point on its own line starting with '- '. Never pack into a single paragraph.")
        resolved_and_unresolved_questions: str = Field(description="What questions did the customer have that were resolved/unresolved by the sales rep. Separate resolved and unresolved into distinct sections with bullet points (- ) for each question.")
        customer_state_of_mind: str = Field(description="Customer state of mind based on conversation AND acoustic emotion signals from the ACOUSTIC CONTEXT block if provided. Include dominant emotion, volatility, and any hesitation patterns observed.")

    class CallCategories(BaseModel):
        primary_pain_point_category: str = Field(description="Strictly one of: 'Harassment Calls', 'High EMI Burden', 'Job Loss/Income Drop', 'Medical Emergency', 'Other'")
        pain_point_evidence: str = Field(description="Cite the exact moments, quotes, and timestamps from the conversation that prove this pain point category. Also reference any hesitation signals or heated moments from the ACOUSTIC CONTEXT block. Use bullet points (- ) for each piece of evidence. Be specific — reference 2-3 key moments.")
        customer_state_of_mind_category: str = Field(description="Strictly one of: 'Distressed/Panicked', 'Relieved/Hopeful', 'Skeptical/Hesitant', 'Angry/Frustrated', 'Neutral'. MUST be consistent with dominant customer emotion from ACOUSTIC CONTEXT if provided.")
        state_of_mind_evidence: str = Field(description="Cite the exact moments, quotes, timestamps AND acoustic signals (dominant emotion, volatility, hesitation phrases) that justify this classification. Use bullet points (- ) for each piece of evidence. Reference 2-3 key moments including at least one acoustic signal if available.")
        lead_conversion_probability: str = Field(description="Strictly one of: 'High', 'Medium', 'Low', 'Not Applicable'")
        conversion_evidence: str = Field(description="Cite conversation moments, timestamps AND LLM-Verified Business Signals from ACOUSTIC CONTEXT (objections, buying triggers) that indicate lead probability. Use bullet points (- ) for each signal. Reference 2-3 specific buying signals or objections.")
        total_identified_debt_inr: int = Field(description="Total sum of all UNIQUE, DEDUPLICATED debt/loan outstanding amounts in INR. Count each loan account ONCE at its highest confirmed outstanding value. If the same amount is mentioned by both agent and customer, count it only once. Use 0 if none is mentioned.")
        major_keywords: list[str] = Field(description="A list of 5-8 keywords. Each keyword MUST follow this format: 'Keyword [MM:SS] — One-line context explaining why this keyword is significant in the call.' Example: 'Escrow Account [05:33] — Agent explained FREED\'s escrow-based savings model for settlement funds'. Do NOT omit the timestamp or context.")

    class ConversionAnalysis(BaseModel):
        did_customer_enroll: str = Field(description="Strictly one of: 'Yes', 'No', 'Unclear'")
        enrollment_outcome_summary: str = Field(description="Brief description of the enrollment outcome and what happened at the end of the call")
        dropoff_reasons: str = Field(description="If the customer did NOT enroll, identify the specific reasons/objections including any hesitation signals or objections from the ACOUSTIC CONTEXT block. Use bullet points (- ) for each reason. Reference timestamps. Write 'N/A - Customer enrolled' if they enrolled.")
        missed_opportunities: str = Field(description="What could the agent have done differently to convert the customer? Use bullet points (- ) for each opportunity. Write 'N/A' if customer enrolled successfully.")
        customer_intent_signals: str = Field(description="List specific moments showing customer interest or hesitation, with timestamps and positive/negative signal label. Cross-reference the ACOUSTIC CONTEXT buying triggers, hesitation phrases, and heated moments. Each signal on its own line with '- '.")

    class CallSummary(BaseModel):
        overview: str = Field(description="What was discussed overall.")
        product_service_explained: ProductServiceExplained
        financial_details: FinancialDetails
        customer_analysis: CustomerAnalysis
        conversion_analysis: ConversionAnalysis
        call_categories: CallCategories

    try:
        client = openai.OpenAI(
            api_key=cfg.openai_api_key,
            timeout=cfg.openai_summary_timeout_s,
            max_retries=cfg.openai_max_retries,
        )
        response = client.beta.chat.completions.parse(
            model=cfg.summary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=CallSummary,
        )
        summary_dict = json.loads(response.choices[0].message.content)
        return summary_dict
    except Exception as e:
        print(f"   ⚠️  LLM Structured Summarization failed: {e}")
        return {"error": str(e)}


def verify_and_inject_inline_citations(data, transcript_segments: list[dict]):
    """
    Recursively search for citation patterns and verify/correct timestamps
    against actual transcript segments.
    
    Handles both **Keyword**[MM:SS] and standalone [MM:SS] patterns.
    Snaps LLM-generated timestamps to the nearest actual segment boundary.
    """
    import re
    import difflib

    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = verify_and_inject_inline_citations(v, transcript_segments)
        return data
    elif isinstance(data, list):
        return [verify_and_inject_inline_citations(item, transcript_segments) for item in data]
    elif isinstance(data, str):
        # Pattern 1: **Keyword**[MM:SS]
        bold_pattern = re.compile(r'\*\*([^*]+)\*\*\[(\d{2}:\d{2}(?::\d{2})?)\]')
        # Pattern 2: standalone [MM:SS] (not already inside an HTML tag)
        standalone_pattern = re.compile(r'(?<!\w)\[(\d{2}:\d{2}(?::\d{2})?)\]')
        
        def _time_to_seconds(time_str: str) -> float:
            parts = time_str.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return int(parts[0]) * 60 + int(parts[1])
        
        def _snap_to_nearest_segment(target_s: float, keyword: str | None = None) -> float | None:
            """Find the nearest segment t0 within ±60s, preferring segments that contain the keyword."""
            if not transcript_segments:
                return None
            
            # First pass: look for keyword match within ±60s window
            if keyword:
                kw_lower = keyword.lower()
                best_t0 = None
                best_dist = float('inf')
                for seg in transcript_segments:
                    dist = min(abs(seg['t0'] - target_s), abs(seg['t1'] - target_s))
                    if dist <= 60:  # 60s window
                        seg_text = seg.get('text', '').lower()
                        if kw_lower in seg_text:
                            if dist < best_dist:
                                best_dist = dist
                                best_t0 = seg['t0']
                        elif len(kw_lower.split()) <= 4:
                            # Fuzzy check for multi-word keywords
                            words = seg_text.split()
                            kw_words = kw_lower.split()
                            for i in range(max(0, len(words) - len(kw_words) + 1)):
                                phrase = ' '.join(words[i:i+len(kw_words)])
                                if difflib.SequenceMatcher(None, phrase, kw_lower).ratio() > 0.75:
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_t0 = seg['t0']
                                    break
                if best_t0 is not None:
                    return best_t0
            
            # Second pass: snap to nearest segment by t0 regardless of keyword
            closest_t0 = None
            min_dist = float('inf')
            for seg in transcript_segments:
                dist = abs(seg['t0'] - target_s)
                if dist < min_dist:
                    min_dist = dist
                    closest_t0 = seg['t0']
                dist2 = abs(seg['t1'] - target_s)
                if dist2 < min_dist:
                    min_dist = dist2
                    closest_t0 = seg['t0']  # Still use t0 as the anchor
            
            return closest_t0

        def bold_replacer(match):
            keyword = match.group(1).strip()
            time_str = match.group(2)
            target_s = _time_to_seconds(time_str)
            snapped_t0 = _snap_to_nearest_segment(target_s, keyword)
            
            if snapped_t0 is not None:
                styled_link = (
                    f'<a href="#" onclick="event.preventDefault(); scrollToTime({snapped_t0});" '
                    f'style="color: #38bdf8; text-decoration: none; font-weight: 600; border-bottom: 1px dotted rgba(56, 189, 248, 0.5); transition: all 0.2s;" '
                    f'onmouseover="this.style.color=\'#a78bfa\'; this.style.borderBottom=\'1px solid #a78bfa\';" '
                    f'onmouseout="this.style.color=\'#38bdf8\'; this.style.borderBottom=\'1px dotted rgba(56, 189, 248, 0.5)\';">'
                    f'{keyword}</a>'
                )
                return styled_link
            else:
                return f'**{keyword}**'
                
        result = bold_pattern.sub(bold_replacer, data)
        return result
    else:
        return data


def format_structured_summary(summary: dict) -> str:
    """Format the structured JSON summary into a readable markdown/text string."""
    if "error" in summary:
        return f"Summary failed: {summary['error']}"

    def _split_to_bullets(text: str) -> str:
        """Split a long paragraph into bullet points on numbered items or existing dashes."""
        if not text or text.strip() == 'No information mentioned in the call.':
            return text
        
        # If already has bullet points / newlines, leave as is
        if '\n- ' in text or '\n* ' in text:
            return text
        
        # Split on numbered patterns: "1) ", "2) ", "1. ", etc.
        import re
        numbered = re.split(r'(?<=\.)\s*(?=\d+[\)\.]\s)', text)
        if len(numbered) > 1:
            return '\n'.join(f'- {item.strip()}' for item in numbered if item.strip())
        
        # Split on " - " used as inline separators within paragraphs
        dashes = text.split(' - ')
        if len(dashes) > 2:  # At least 3 items separated by " - "
            return '\n'.join(f'- {item.strip()}' for item in dashes if item.strip())
        
        return text

    lines = []
    lines.append("OVERVIEW OF DISCUSSION")
    lines.append("-" * 20)
    lines.append(summary.get("overview", "No information mentioned in the call."))
    
    lines.append("\nPRODUCT/SERVICE EXPLAINED")
    lines.append("-" * 20)
    prod = summary.get("product_service_explained", {})
    lines.append(f"Details Shared:")
    lines.append(_split_to_bullets(prod.get('details_shared', 'No information mentioned in the call.')))
    lines.append(f"\nBenefits Mentioned:")
    lines.append(_split_to_bullets(prod.get('benefits_mentioned', 'No information mentioned in the call.')))
    lines.append(f"\nConditions Mentioned:")
    lines.append(_split_to_bullets(prod.get('conditions_mentioned', 'No information mentioned in the call.')))
    
    lines.append("\nFINANCIAL DETAILS DISCUSSED")
    lines.append("-" * 20)
    fin = summary.get("financial_details", {})
    lines.append(f"Loan Amounts:\n{_split_to_bullets(fin.get('loan_amounts', 'No information mentioned in the call.'))}")
    lines.append(f"\nInterest Rates: {fin.get('interest_rates', 'No information mentioned in the call.')}")
    lines.append(f"\nEMIs:\n{_split_to_bullets(fin.get('emis', 'No information mentioned in the call.'))}")
    lines.append(f"\nTenure: {fin.get('tenure', 'No information mentioned in the call.')}")
    lines.append(f"\nFees/Charges:\n{_split_to_bullets(fin.get('fees_or_charges', 'No information mentioned in the call.'))}")
    lines.append(f"\nOther Numbers (with context):\n{_split_to_bullets(fin.get('other_numbers_with_context', 'No information mentioned in the call.'))}")
    
    lines.append("\nCUSTOMER ANALYSIS")
    lines.append("-" * 20)
    cust = summary.get("customer_analysis", {})
    lines.append(f"Biggest Pain Point:")
    lines.append(_split_to_bullets(cust.get('biggest_pain_point', 'No information mentioned in the call.')))
    lines.append(f"\nMost Appealing Aspect & Moment:")
    lines.append(_split_to_bullets(cust.get('most_appealing_aspect_and_moment_of_interest', 'No information mentioned in the call.')))
    lines.append(f"\nMain Points Pitched:")
    lines.append(_split_to_bullets(cust.get('main_points_pitched', 'No information mentioned in the call.')))
    lines.append(f"\nResolved/Unresolved Questions:")
    lines.append(_split_to_bullets(cust.get('resolved_and_unresolved_questions', 'No information mentioned in the call.')))
    lines.append(f"\nCustomer State of Mind: {cust.get('customer_state_of_mind', 'No information mentioned in the call.')}")

    lines.append("\nCONVERSION & DROPOFF ANALYSIS")
    lines.append("-" * 20)
    conv = summary.get("conversion_analysis", {})
    lines.append(f"Did Customer Enroll: {conv.get('did_customer_enroll', 'Not specified')}")
    lines.append(f"\nEnrollment Outcome: {conv.get('enrollment_outcome_summary', 'No information mentioned in the call.')}")
    lines.append(f"\nDropoff Reasons:")
    lines.append(_split_to_bullets(conv.get('dropoff_reasons', 'No information mentioned in the call.')))
    lines.append(f"\nMissed Opportunities:")
    lines.append(_split_to_bullets(conv.get('missed_opportunities', 'No information mentioned in the call.')))
    lines.append(f"\nCustomer Intent Signals:")
    lines.append(_split_to_bullets(conv.get('customer_intent_signals', 'No information mentioned in the call.')))
    
    lines.append("\nCALL CATEGORIES")
    lines.append("-" * 20)
    cats = summary.get("call_categories", {})
    lines.append(f"Primary Pain Point: {cats.get('primary_pain_point_category', 'Not specified')}")
    lines.append(f"  Evidence:\n{_split_to_bullets(cats.get('pain_point_evidence', 'No evidence provided.'))}")
    lines.append(f"\nCustomer State of Mind: {cats.get('customer_state_of_mind_category', 'Not specified')}")
    lines.append(f"  Evidence:\n{_split_to_bullets(cats.get('state_of_mind_evidence', 'No evidence provided.'))}")
    lines.append(f"\nLead Conversion Probability: {cats.get('lead_conversion_probability', 'Not specified')}")
    lines.append(f"  Evidence:\n{_split_to_bullets(cats.get('conversion_evidence', 'No evidence provided.'))}")
    lines.append(f"\nTotal Identified Debt (deduplicated): ₹{cats.get('total_identified_debt_inr', 0):,}")
    kws = cats.get('major_keywords', [])
    if kws:
        lines.append("\nMajor Keywords:")
        for kw in kws:
            lines.append(f"- {kw}")
    else:
        lines.append("\nMajor Keywords: None identified")

    return "\n".join(lines)
