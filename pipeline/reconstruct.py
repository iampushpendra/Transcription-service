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

    # Assign stable zero-indexed IDs used for LLM citations (e.g., [S42] → seg[42].t0).
    for i, seg in enumerate(merged):
        seg["id"] = i

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

    print("\U0001f916 Generating Structured LLM Summary...")

    if not transcript:
        return {"error": "Empty transcript"}

    # Build transcript text for the prompt.
    # Each line is prefixed with [S<id>] so the LLM can cite segments exactly.
    # Timestamps stay visible as context but the LLM is instructed to cite via S-IDs only.
    lines = []
    for i, seg in enumerate(transcript):
        role_label = seg.get("role", seg.get("speaker", "Unknown")).upper()
        seg_id = seg.get("id", i)
        t0 = seg.get('t0', 0.0)
        timestamp = fmt(t0)[:8]
        lines.append(f"[S{seg_id}] [{timestamp}] {role_label}: {seg.get('text', '')}")
    transcript_text = "\n".join(lines)

    # Build optional Acoustic & Behavioral Context block
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

            parts.append(f"Dominant Emotions \u2014 Agent: {dom_agent} | Customer: {dom_cust}")
            if vol is not None:
                parts.append(f"Emotional Volatility: {round(vol * 100)}% ({switches} state switches)")
            if peak:
                ts = peak.get("timestamp", "")
                spk = peak.get("speaker", "")
                emo = peak.get("emotion_label", peak.get("emotion", ""))
                parts.append(f"Peak Intensity: {emo} @ {ts} ({spk})")

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
                        parts.append(f'  - {ts_h} [{cat}] "{phrase}"')

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
                    parts.append(f'  - {cat_label} {ts_s}: "{quote}" \u2014 {insight}')

        return "\n".join(parts)

    context_block = _build_context_block(emotion_analysis, trigger_phrases)
    has_context = bool(context_block.strip())

    context_instruction = (
        "\n\nACOUSTIC CONTEXT USAGE (CRITICAL):\n"
        "You are provided with an '=== ACOUSTIC & BEHAVIORAL CONTEXT ==' block above the transcript.\n"
        "- Cross-reference this block when filling ALL evidence fields (pain_point_evidence, state_of_mind_evidence, conversion_evidence, customer_intent_signals).\n"
        "- Cite hesitation signals or heated moments with an S-ID citation (e.g., `[S42]`) and what it reveals.\n"
        "- Dominant emotions and volatility MUST inform customer_state_of_mind_category.\n"
        "- Business signals (objections/triggers) MUST be reflected in conversion_analysis.\n"
        "- Keep all answers crisp \u2014 enrich with acoustic evidence, do NOT expand unnecessarily."
    ) if has_context else ""

    checklist_reference = (
        "\nAGENT SALES CALL CHECKLIST (21 items to evaluate):\n"
        "  BASICS: (1) Call opening \u2014 greeting + intro + reason for call. (2) Confirm customer full name.\n"
        "  DEBT UNDERSTANDING: (3) Checked total outstanding unsecured loans (PL + CC). (4) Checked last payment date. "
        "(5) Identified hardship/reason for non-payment. (6) Empathized with customer.\n"
        "  AFFORDABILITY: (7) Confirmed income source + monthly affordability + future funds possibility.\n"
        "  PROGRAM PITCH: (8) Explained how FREED works + suitable program. (9) If DRP: settlement concept clearly explained. "
        "(10) Explained program flow \u2014 SPA + settlement conditions. (11) Explained estimated savings + service fee + tenure.\n"
        "  ACCOUNT CHECKS: (12) Checked auto-debit/NACH status + salary account linked.\n"
        "  VALUE & FEES: (13) Explained CHPP, Shield, i-FREED app, support, negotiation benefits. (14) Explained platform fee.\n"
        "  COMPLIANCE: (15) Discussed collection calls + borrower rights (timings, WhatsApp, visit docs, recording). "
        "(16) Discussed credit score impact (short-term dip, long-term recovery).\n"
        "  VERIFICATION: (17) Re-verified via credit report \u2014 debt load + last payment date + 3 good-faith payments. "
        "(18) Discussed monthly budget + total savings required.\n"
        "  CLOSURE: (19) Sent the agreement. (20) Collected evaluation fee. (21) Set up E-mandate (monthly SPA contribution).\n"
    )

    freed_domain_context = (
        "FREED DOMAIN CONTEXT:\n"
        "- DRP: settlement at ~45% of enrolled debt. Service fee: 15%+GST. Total SPA savings required: ~62.7%.\n"
        "- DCP: multiple EMIs merged into one via third-party lender.\n"
        "- DEP: structured accelerated repayment plan.\n"
        "- SPA = Special Purpose Account (escrow account for monthly savings toward settlement).\n"
        "- CHPP = Creditor Harassment Protection Programme. FREED Shield = legal protection. i-FREED = customer app.\n"
        "- Platform fee: 10%+GST of monthly SPA contribution. Evaluation fee = one-time onboarding fee.\n"
        "- Settlement timeline: 2-3 years typically. Settlement triggered when SPA ~45% of debt AND creditor agrees.\n"
        "- Borrower Rights: calls only 8AM-7PM, no Sunday/holiday obligation, record all calls, no payment date commitment.\n"
        "- DRA visit: must carry Authorization Letter, Company ID, DRA Certificate, Recovery Notice.\n"
        "- CALL TYPE: First Call (new prospect), Follow-Up Call (existing prospect re-engagement), "
        "or Existing Customer Call (enrolled customer seeking support).\n"
    )

    system_prompt = (
        "You are an expert call analyst for FREED \u2014 India's leading debt relief platform.\n\n"
        + freed_domain_context
        + "\nYour output serves: (1) Sales managers for coaching & handoff, "
        "(2) Audit team for compliance review, (3) Next agent picking up this prospect.\n\n"
        "HARD REQUIREMENTS:\n"
        "- Do NOT hallucinate. Do NOT assume missing details. Do NOT add fluff.\n"
        "- Preserve all financial numbers exactly as spoken.\n"
        "- If unclear, write \"Not clearly specified in the call.\"\n"
        "- Professional English only. Every field must be CRISP, FACTUAL, and EVIDENCED.\n"
        "- Use bullet points (- ) for every multi-value field \u2014 NEVER use paragraphs.\n\n"
        "FORMATTING:\n"
        "- Each distinct fact/observation gets its own bullet line starting with '- '.\n"
        "- NEVER pack multiple points into one paragraph.\n"
        "- CITATIONS: After any specific detail, add a segment reference like `[S42]` immediately. "
        "The ID must be copied EXACTLY from one of the `[S<number>]` prefixes in the numbered transcript below. "
        "Never invent IDs. Never cite with timestamps like `[MM:SS]`.\n"
        "- SPEAKER ATTRIBUTION: Clearly distinguish AGENT vs CUSTOMER speech. "
        "Do NOT attribute agent pitch points to the customer.\n\n"
        "OVERVIEW FIELD:\n"
        "- Start with a 1-2 sentence call summary (call type + outcome).\n"
        "- Then list all customer pain points as separate bullets.\n"
        "- Format: Summary sentence.\\n- Pain Point 1\\n- Pain Point 2\\n...\n\n"
        "NUMERIC DEDUPLICATION:\n"
        "- Count each unique loan account ONCE at its highest confirmed value.\n"
        "- Do NOT add the same amount twice if mentioned by both agent and customer.\n"
        "- Avoid repeating the same figure across multiple fields.\n\n"
        "AGENT CHECKLIST EVALUATION:\n"
        + checklist_reference
        + "\nFor each of the 21 items, evaluate and return: item description, category, "
        "status (Completed/Partially Completed/Not Done/N/A - Follow-up call), "
        "and 1-line evidence (quote + S-ID citation if done, gap explanation if not).\n"
        "For FOLLOW-UP or EXISTING CUSTOMER calls, closure items (19-21) if truly not applicable: mark 'N/A - Follow-up call'.\n\n"
        "AUDIT COMPLIANCE FLAGS:\n"
        "Flag: false promises, threatening language, unauthorized commitments on behalf of FREED, "
        "incorrect fee/settlement figures, failure to disclose credit score impact, failure to mention borrower rights, "
        "misrepresentation of FREED's service.\n\n"
        "OBJECTION HANDLING: For each major customer objection, classify agent response as: "
        "'Well Handled' (resolved with correct info + empathy), 'Partially Handled' (addressed but incomplete), "
        "or 'Not Handled' (ignored/deflected).\n"
        + context_instruction
    )

    user_prompt = f"Notes:\n{context_block}\n\n{transcript_text}" if has_context else f"Notes:\n{transcript_text}"

    class ProductServiceExplained(BaseModel):
        details_shared: str = Field(
            description="Exact FREED program details explained \u2014 which program (DRP/DCP/DEP), how it works, what's included. One bullet per distinct detail with S-ID citation."
        )
        benefits_mentioned: str = Field(
            description="Specific benefits pitched: CHPP, FREED Shield, i-FREED app, SPA savings model, negotiation team, pre-litigation support. One bullet per benefit with S-ID citation."
        )
        conditions_mentioned: str = Field(
            description="Conditions explained: delinquency requirement, credit score impact, tenure, saving commitment, service fee structure. One bullet per condition."
        )

    class FinancialDetails(BaseModel):
        loan_amounts: str = Field(
            description="Each unique loan/credit card outstanding \u2014 lender name, amount, type. One bullet per account. Deduplicate if same loan mentioned multiple times."
        )
        interest_rates: str = Field(
            description="Interest rates mentioned for any loan or card. Write 'Not mentioned' if absent."
        )
        emis: str = Field(
            description="Current EMI amounts per loan or card, with lender if mentioned. Write 'Not mentioned' if absent."
        )
        tenure: str = Field(
            description="Expected FREED program tenure or remaining loan tenure if discussed. Write 'Not mentioned' if absent."
        )
        fees_or_charges: str = Field(
            description="All fees discussed: evaluation fee, platform fee (10%+GST of SPA contribution), service fee (15%+GST of enrolled debt on settlement), with amounts/percentages if quoted."
        )
        settlement_target: str = Field(
            description="Settlement target discussed \u2014 e.g. '45% of enrolled debt', with exact INR amount if computed by agent. Write 'Not discussed' if absent."
        )
        other_numbers_with_context: str = Field(
            description="Any other numbers not captured above \u2014 CIBIL scores, account numbers, dates, etc. Each with one-line context."
        )

    class CustomerAnalysis(BaseModel):
        biggest_pain_point: str = Field(
            description="Primary hardship \u2014 why customer can't pay. Include exact trigger (job loss, medical, income drop, harassment). Cite verbatim quotes + S-ID citations."
        )
        delinquency_status: str = Field(
            description="Current payment status \u2014 how many months overdue per account, which accounts are delinquent. Include specifics if mentioned."
        )
        most_appealing_aspect_and_moment_of_interest: str = Field(
            description="Which FREED feature resonated most with the customer. At what moment did customer show interest or positively respond? Cite quotes with S-ID citations (e.g., `[S42]`)."
        )
        main_points_pitched: str = Field(
            description="Agent's key pitch points \u2014 each as a separate bullet with S-ID citation. Focus on 5-6 strongest arguments made."
        )
        resolved_and_unresolved_questions: str = Field(
            description="RESOLVED: question + agent answer + S-ID. UNRESOLVED: question + why not addressed. Use sub-bullets for each category."
        )
        customer_state_of_mind: str = Field(
            description="Overall customer emotional state with specific quote evidence. Cross-reference acoustic emotion data if available. Note hesitation or buying signals."
        )
        customer_program_fit: str = Field(
            description="Which FREED program (DRP/DCP/DEP) best fits this customer based on debt profile, income, delinquency. State if agent's recommendation was appropriate."
        )

    class CallCategories(BaseModel):
        primary_pain_point_category: str = Field(
            description="Strictly one of: 'Harassment/Recovery Agent Calls', 'High EMI Burden', 'Job Loss/Income Drop', 'Medical Emergency', 'Debt Already Delinquent', 'Credit Score Concern', 'Unable to Pay (General)', 'Other'"
        )
        pain_point_evidence: str = Field(
            description="2-3 specific quotes + S-ID citations proving the pain point category. Also cite acoustic hesitation signals or heated moments if available."
        )
        customer_state_of_mind_category: str = Field(
            description="Strictly one of: 'Distressed/Panicked', 'Relieved/Hopeful', 'Skeptical/Hesitant', 'Angry/Frustrated', 'Neutral/Curious'. Must be consistent with acoustic emotion context if available."
        )
        state_of_mind_evidence: str = Field(
            description="2-3 specific quotes + S-ID citations justifying the state. Include acoustic emotion signals (dominant emotion, volatility, hesitation phrases) if available."
        )
        lead_conversion_probability: str = Field(
            description="Strictly one of: 'High', 'Medium', 'Low', 'Not Applicable'"
        )
        conversion_evidence: str = Field(
            description="2-3 buying signals or objections with S-ID citations determining lead probability. Cross-reference acoustic business signals if available."
        )
        total_identified_debt_inr: int = Field(
            description="Sum of UNIQUE deduplicated loan outstanding amounts in INR. Each account counted ONCE at highest confirmed value. Use 0 if none mentioned."
        )
        major_keywords: list[str] = Field(
            description="5-8 domain-specific keywords. Format: '**Keyword**[S42] \u2014 One-line context explaining significance.' Must include a segment S-ID citation for each."
        )

    class ConversionAnalysis(BaseModel):
        did_customer_enroll: str = Field(description="Strictly one of: 'Yes', 'No', 'Unclear'")
        enrollment_outcome_summary: str = Field(
            description="What happened at end of call \u2014 did customer agree, ask for time, raise final objection? Be specific with S-ID citation."
        )
        dropoff_reasons: str = Field(
            description="If no enrollment: each specific objection or hesitation with S-ID citation and whether agent addressed it. Write 'N/A - Customer enrolled' if enrolled."
        )
        missed_opportunities: str = Field(
            description="Concrete moments where agent could have pivoted or closed better. Each as bullet with suggested alternative. Write 'N/A - Successfully enrolled' if enrolled."
        )
        customer_intent_signals: str = Field(
            description="All signals of customer interest/hesitation with S-ID citations and label [+Positive] or [-Negative]. Cross-reference acoustic hesitation and buying trigger signals."
        )

    class ChecklistItem(BaseModel):
        item: str = Field(description="The checklist item description from the 21-item Sales Call Checklist")
        category: str = Field(description="One of: Basics, Debt Understanding, Affordability, Program Pitch, Account Checks, Value & Fees, Compliance, Verification, Closure")
        status: str = Field(description="One of: 'Completed', 'Partially Completed', 'Not Done', 'N/A - Follow-up call'")
        evidence: str = Field(description="1-line: quote with S-ID citation (e.g., `[S42]`) if completed, or specific gap explanation if not done.")

    class AgentChecklist(BaseModel):
        checklist_items: list[ChecklistItem] = Field(
            description="All 21 checklist items evaluated against the actual call. Must return exactly 21 items in order."
        )
        overall_score: str = Field(
            description="Score as 'X/21 completed' (count only 'Completed' items). Classify: 'Strong (18-21)', 'Adequate (12-17)', or 'Needs Improvement (<12)'."
        )
        critical_gaps: str = Field(
            description="Items entirely skipped that are critical for compliance or conversion \u2014 especially items 15 (borrower rights), 16 (credit score), 8 (FREED explanation). List each with why it matters."
        )
        agent_strengths: str = Field(
            description="2-3 things the agent did particularly well. Be specific with S-ID citations."
        )

    class AuditTeamInsights(BaseModel):
        call_type: str = Field(
            description="Strictly one of: 'First Call', 'Follow-Up Call', 'Existing Customer Call'. Infer from context."
        )
        agent_communication_quality: str = Field(
            description="3-4 bullet assessment of agent's language clarity, tone, empathy, and script adherence. Cite specific moments. Distinguish strengths from weaknesses."
        )
        compliance_flags: str = Field(
            description="List ANY compliance issues: false promises, incorrect figures, failure to disclose credit impact or borrower rights, unauthorized FREED commitments, pressure language. Write 'No compliance flags identified' if clean."
        )
        objections_raised_and_handling: str = Field(
            description="For each major customer objection: OBJECTION: [quote with S-ID citation] | AGENT RESPONSE: [summary] | QUALITY: Well Handled / Partially Handled / Not Handled. One entry per objection."
        )
        recommended_next_action: str = Field(
            description="What should happen next: follow-up call (when + what to address), enrollment push, escalate to senior agent, send agreement, etc."
        )

    class CallSummary(BaseModel):
        overview: str = Field(
            description="Call type + 1-2 sentence summary. Then list all customer pain points as bullets. Format: 'This was a [First/Follow-Up] call. [Summary].\\n- Pain Point 1\\n- Pain Point 2'"
        )
        call_type: str = Field(
            description="Strictly one of: 'First Call', 'Follow-Up Call', 'Existing Customer Call'"
        )
        product_service_explained: ProductServiceExplained
        financial_details: FinancialDetails
        customer_analysis: CustomerAnalysis
        conversion_analysis: ConversionAnalysis
        call_categories: CallCategories
        audit_team_insights: AuditTeamInsights
        agent_checklist: AgentChecklist

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
        print(f"   \u26a0\ufe0f  LLM Structured Summarization failed: {e}")
        return {"error": str(e)}




# --- Citation resolver constants ---------------------------------------------

# Primary citation forms (preferred): **Keyword**[S42] and bare [S42]
_BOLD_SID_PATTERN = re.compile(r'\*\*([^*]+)\*\*\[S(\d+)\]')
_BARE_SID_PATTERN = re.compile(r'(?<![\w/])\[S(\d+)\](?!/)')  # single-call scope; exclude the /C<n> compound

# Compound citations used inside chain-level artifacts: [S42/C2]
_BOLD_COMPOUND_PATTERN = re.compile(r'\*\*([^*]+)\*\*\[S(\d+)/C(\d+)\]')
_BARE_COMPOUND_PATTERN = re.compile(r'(?<![\w/])\[S(\d+)/C(\d+)\]')

# Fallback: legacy [MM:SS] / [HH:MM:SS] timestamps
_BOLD_MMSS_PATTERN = re.compile(r'\*\*([^*]+)\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\]')
_BARE_MMSS_PATTERN = re.compile(r'(?<![\w/])\[(\d{1,2}:\d{2}(?::\d{2})?)\]')

# Keyword-match window for MM:SS fallback. Tight on purpose; outside this window
# the citation is dropped rather than mis-linked.
_MMSS_KEYWORD_WINDOW_S = 10.0


def _linkify_keyword(keyword: str, t0: float) -> str:
    """Wrap a keyword in a clickable anchor that jumps to t0 in the transcript."""
    return (
        f'<a href="#" onclick="event.preventDefault(); scrollToTime({t0});" '
        f'style="color: #38bdf8; text-decoration: none; font-weight: 600; '
        f'border-bottom: 1px dotted rgba(56, 189, 248, 0.5); transition: all 0.2s;" '
        f'onmouseover="this.style.color=\'#a78bfa\'; this.style.borderBottom=\'1px solid #a78bfa\';" '
        f'onmouseout="this.style.color=\'#38bdf8\'; this.style.borderBottom=\'1px dotted rgba(56, 189, 248, 0.5)\';">'
        f'{keyword}</a>'
    )


def _linkify_arrow(t0: float) -> str:
    """Small superscript arrow link used for bare citations (no keyword to wrap)."""
    return (
        f'<a href="#" onclick="event.preventDefault(); scrollToTime({t0});" '
        f'style="color: rgba(56, 189, 248, 0.7); text-decoration: none; cursor: pointer; '
        f'font-size: 0.85em; margin-left: 3px; vertical-align: super; transition: all 0.2s;" '
        f'onmouseover="this.style.color=\'#38bdf8\'" '
        f'onmouseout="this.style.color=\'rgba(56, 189, 248, 0.7)\'" '
        f'title="Jump to transcript">[↗]</a>'
    )


def _time_to_seconds(time_str: str) -> float:
    parts = time_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0]) * 60 + int(parts[1])


def _snap_mmss_with_keyword(
    target_s: float,
    keyword: str,
    segments: list[dict],
) -> float | None:
    """
    Hardened MM:SS fallback snap. Returns the anchor (t0 OR t1 — whichever is
    actually closer) of the best segment where `keyword` appears within
    ±_MMSS_KEYWORD_WINDOW_S of target_s. Returns None if no such segment — the
    caller drops the citation rather than guessing.
    """
    if not keyword or not segments:
        return None

    kw_lower = keyword.lower().strip()
    best_anchor: float | None = None
    best_dist = float('inf')

    for seg in segments:
        t0 = float(seg.get('t0', 0.0))
        t1 = float(seg.get('t1', t0))
        dist_t0 = abs(t0 - target_s)
        dist_t1 = abs(t1 - target_s)
        # Distance from target to the segment (0 if inside it).
        if t0 <= target_s <= t1:
            dist = 0.0
        else:
            dist = min(dist_t0, dist_t1)
        if dist > _MMSS_KEYWORD_WINDOW_S:
            continue
        if kw_lower not in seg.get('text', '').lower():
            continue
        if dist < best_dist:
            best_dist = dist
            # Use the actually-closer anchor so the jump lands as close as possible.
            best_anchor = t0 if dist_t0 <= dist_t1 else t1

    return best_anchor


def verify_and_inject_inline_citations(
    data,
    transcript_segments: list[dict],
    chain_context: dict[int, list[dict]] | None = None,
):
    """
    Recursively resolve citations in LLM output to clickable transcript links.

    Primary (single-call): `**Keyword**[S42]` / `[S42]` → exact ID lookup in
    `transcript_segments`.

    Primary (chain-level): `**Keyword**[S42/C2]` / `[S42/C2]` → look up call 2
    in `chain_context` and resolve seg 42 inside it. Used inside chain-level
    artifacts where a citation must identify both the call and the segment.

    Fallback: `**Keyword**[MM:SS]` / `[MM:SS]` → snap to the closest segment
    containing the keyword within ±10s. A bare `[MM:SS]` with no keyword falls
    back to exact time-within-segment match or passes through unchanged.

    Invalid references are dropped rather than misdirected. Any resolve
    failure degrades to plain text — never raises.
    """
    if isinstance(data, dict):
        return {k: verify_and_inject_inline_citations(v, transcript_segments, chain_context) for k, v in data.items()}
    if isinstance(data, list):
        return [verify_and_inject_inline_citations(item, transcript_segments, chain_context) for item in data]
    if not isinstance(data, str):
        return data

    # Build {id: t0} lookup. Fall back to enumerate order for legacy segments
    # that predate the 'id' field.
    id_to_t0: dict[int, float] = {}
    for i, seg in enumerate(transcript_segments or []):
        sid = seg.get('id', i)
        try:
            id_to_t0[int(sid)] = float(seg.get('t0', 0.0))
        except (TypeError, ValueError):
            continue

    result = data

    # --- Compound (chain-scoped) citations: [S42/C2] ----------------------
    # Run first so the single-call S-ID pattern doesn't consume their inner [S42] token.
    def _compound_lookup(sid: int, call_idx: int) -> float | None:
        if not chain_context:
            return None
        segs = chain_context.get(call_idx)
        if not segs:
            return None
        for i, seg in enumerate(segs):
            seg_id = seg.get('id', i)
            try:
                if int(seg_id) == sid:
                    return float(seg.get('t0', 0.0))
            except (TypeError, ValueError):
                continue
        return None

    def _bold_compound_sub(m: re.Match) -> str:
        keyword = m.group(1).strip()
        sid, call_idx = int(m.group(2)), int(m.group(3))
        t0 = _compound_lookup(sid, call_idx)
        if t0 is None:
            return f'**{keyword}**'
        return _linkify_keyword(keyword, t0)

    def _bare_compound_sub(m: re.Match) -> str:
        sid, call_idx = int(m.group(1)), int(m.group(2))
        t0 = _compound_lookup(sid, call_idx)
        if t0 is None:
            return ''
        return _linkify_arrow(t0)

    result = _BOLD_COMPOUND_PATTERN.sub(_bold_compound_sub, result)
    result = _BARE_COMPOUND_PATTERN.sub(_bare_compound_sub, result)

    # --- Primary: single-call segment-ID citations ------------------------
    def _bold_sid_sub(m: re.Match) -> str:
        keyword = m.group(1).strip()
        sid = int(m.group(2))
        t0 = id_to_t0.get(sid)
        if t0 is None:
            return f'**{keyword}**'  # drop cite, keep bold keyword
        return _linkify_keyword(keyword, t0)

    def _bare_sid_sub(m: re.Match) -> str:
        sid = int(m.group(1))
        t0 = id_to_t0.get(sid)
        if t0 is None:
            return ''  # drop unresolved bare cite
        return _linkify_arrow(t0)

    result = _BOLD_SID_PATTERN.sub(_bold_sid_sub, result)
    result = _BARE_SID_PATTERN.sub(_bare_sid_sub, result)

    # --- Fallback: legacy MM:SS citations ----------------------------------
    def _bold_mmss_sub(m: re.Match) -> str:
        keyword = m.group(1).strip()
        target_s = _time_to_seconds(m.group(2))
        t0 = _snap_mmss_with_keyword(target_s, keyword, transcript_segments)
        if t0 is None:
            return f'**{keyword}**'  # drop cite, keep bold keyword
        return _linkify_keyword(keyword, t0)

    def _bare_mmss_sub(m: re.Match) -> str:
        # No keyword to anchor on. If the target time falls inside a real
        # segment span, we can still resolve it exactly. Otherwise leave the
        # token in place so the frontend can make its own best-effort link.
        target_s = _time_to_seconds(m.group(1))
        for seg in transcript_segments or []:
            t0 = float(seg.get('t0', 0.0))
            t1 = float(seg.get('t1', t0))
            if t0 <= target_s <= t1:
                return _linkify_arrow(t0)
        return m.group(0)  # unchanged → frontend fallback

    result = _BOLD_MMSS_PATTERN.sub(_bold_mmss_sub, result)
    result = _BARE_MMSS_PATTERN.sub(_bare_mmss_sub, result)

    # Clean any double spaces left behind by dropped cites.
    result = re.sub(r'  +', ' ', result)
    return result


def format_structured_summary(summary: dict) -> str:
    """Format the structured JSON summary into a readable markdown/text string."""
    if "error" in summary:
        return f"Summary failed: {summary['error']}"

    def _split_to_bullets(text: str) -> str:
        """Split a long paragraph into bullet points on numbered items or existing dashes."""
        if not text or text.strip() in ('No information mentioned in the call.', 'Not clearly specified in the call.', 'Not mentioned', 'Not discussed'):
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

    # OVERVIEW
    call_type = summary.get("call_type", "")
    call_type_badge = f"[{call_type}]" if call_type else ""
    lines.append(f"OVERVIEW OF DISCUSSION {call_type_badge}")
    lines.append("-" * 30)
    lines.append(summary.get("overview", "No information mentioned in the call."))

    # PRODUCT / SERVICE
    lines.append("\nPRODUCT/SERVICE EXPLAINED")
    lines.append("-" * 30)
    prod = summary.get("product_service_explained", {})
    lines.append("Program Details Shared:")
    lines.append(_split_to_bullets(prod.get('details_shared', 'No information mentioned in the call.')))
    lines.append("\nBenefits Mentioned:")
    lines.append(_split_to_bullets(prod.get('benefits_mentioned', 'No information mentioned in the call.')))
    lines.append("\nConditions & Expectations:")
    lines.append(_split_to_bullets(prod.get('conditions_mentioned', 'No information mentioned in the call.')))

    # FINANCIAL DETAILS
    lines.append("\nFINANCIAL DETAILS DISCUSSED")
    lines.append("-" * 30)
    fin = summary.get("financial_details", {})
    lines.append(f"Loan Accounts & Outstandings:\n{_split_to_bullets(fin.get('loan_amounts', 'Not mentioned'))}")
    lines.append(f"\nInterest Rates: {fin.get('interest_rates', 'Not mentioned')}")
    lines.append(f"\nCurrent EMIs:\n{_split_to_bullets(fin.get('emis', 'Not mentioned'))}")
    lines.append(f"\nExpected Tenure: {fin.get('tenure', 'Not mentioned')}")
    lines.append(f"\nFees & Charges:\n{_split_to_bullets(fin.get('fees_or_charges', 'Not mentioned'))}")
    lines.append(f"\nSettlement Target: {fin.get('settlement_target', 'Not discussed')}")
    lines.append(f"\nOther Numbers:\n{_split_to_bullets(fin.get('other_numbers_with_context', 'Not mentioned'))}")

    # CUSTOMER ANALYSIS
    lines.append("\nCUSTOMER ANALYSIS")
    lines.append("-" * 30)
    cust = summary.get("customer_analysis", {})
    lines.append("Primary Pain Point:")
    lines.append(_split_to_bullets(cust.get('biggest_pain_point', 'Not clearly specified in the call.')))
    lines.append(f"\nDelinquency Status: {cust.get('delinquency_status', 'Not clearly specified in the call.')}")
    lines.append("\nMost Appealing Aspect & Moment of Interest:")
    lines.append(_split_to_bullets(cust.get('most_appealing_aspect_and_moment_of_interest', 'Not clearly specified in the call.')))
    lines.append("\nMain Points Pitched:")
    lines.append(_split_to_bullets(cust.get('main_points_pitched', 'Not clearly specified in the call.')))
    lines.append("\nResolved / Unresolved Questions:")
    lines.append(_split_to_bullets(cust.get('resolved_and_unresolved_questions', 'Not clearly specified in the call.')))
    lines.append(f"\nCustomer State of Mind: {cust.get('customer_state_of_mind', 'Not clearly specified in the call.')}")
    lines.append(f"\nRecommended Program Fit: {cust.get('customer_program_fit', 'Not clearly specified in the call.')}")

    # CONVERSION & DROPOFF
    lines.append("\nCONVERSION & DROPOFF ANALYSIS")
    lines.append("-" * 30)
    conv = summary.get("conversion_analysis", {})
    lines.append(f"Did Customer Enroll: {conv.get('did_customer_enroll', 'Not specified')}")
    lines.append(f"\nEnrollment Outcome: {conv.get('enrollment_outcome_summary', 'Not clearly specified in the call.')}")
    lines.append("\nDropoff Reasons:")
    lines.append(_split_to_bullets(conv.get('dropoff_reasons', 'Not clearly specified in the call.')))
    lines.append("\nMissed Opportunities:")
    lines.append(_split_to_bullets(conv.get('missed_opportunities', 'Not clearly specified in the call.')))
    lines.append("\nCustomer Intent Signals:")
    lines.append(_split_to_bullets(conv.get('customer_intent_signals', 'Not clearly specified in the call.')))

    # CALL CATEGORIES
    lines.append("\nCALL CATEGORIES")
    lines.append("-" * 30)
    cats = summary.get("call_categories", {})
    lines.append(f"Primary Pain Point Category: {cats.get('primary_pain_point_category', 'Not specified')}")
    lines.append(f"  Evidence:\n{_split_to_bullets(cats.get('pain_point_evidence', 'No evidence provided.'))}")
    lines.append(f"\nCustomer State of Mind: {cats.get('customer_state_of_mind_category', 'Not specified')}")
    lines.append(f"  Evidence:\n{_split_to_bullets(cats.get('state_of_mind_evidence', 'No evidence provided.'))}")
    lines.append(f"\nLead Conversion Probability: {cats.get('lead_conversion_probability', 'Not specified')}")
    lines.append(f"  Evidence:\n{_split_to_bullets(cats.get('conversion_evidence', 'No evidence provided.'))}")
    lines.append(f"\nTotal Identified Debt (deduplicated): \u20b9{cats.get('total_identified_debt_inr', 0):,}")
    kws = cats.get('major_keywords', [])
    if kws:
        lines.append("\nMajor Keywords:")
        for kw in kws:
            lines.append(f"- {kw}")
    else:
        lines.append("\nMajor Keywords: None identified")

    # AUDIT TEAM INSIGHTS
    audit = summary.get("audit_team_insights", {})
    if audit:
        lines.append("\nAUDIT TEAM INSIGHTS")
        lines.append("-" * 30)
        audit_call_type = audit.get("call_type", "")
        if audit_call_type:
            lines.append(f"Call Type: {audit_call_type}")
        lines.append("\nAgent Communication Quality:")
        lines.append(_split_to_bullets(audit.get('agent_communication_quality', 'Not assessed.')))
        lines.append("\nCompliance Flags:")
        lines.append(_split_to_bullets(audit.get('compliance_flags', 'No compliance flags identified.')))
        lines.append("\nObjections & Handling Quality:")
        lines.append(_split_to_bullets(audit.get('objections_raised_and_handling', 'Not assessed.')))
        lines.append(f"\nRecommended Next Action: {audit.get('recommended_next_action', 'Not specified.')}")

    # AGENT CALL CHECKLIST
    checklist = summary.get("agent_checklist", {})
    if checklist:
        lines.append("\nAGENT CALL CHECKLIST")
        lines.append("-" * 30)
        lines.append(f"Overall Score: {checklist.get('overall_score', 'Not evaluated')}")
        lines.append("\nAgent Strengths:")
        lines.append(_split_to_bullets(checklist.get('agent_strengths', 'Not assessed.')))
        lines.append("\nCritical Gaps:")
        lines.append(_split_to_bullets(checklist.get('critical_gaps', 'None identified.')))

        items = checklist.get("checklist_items", [])
        if items:
            lines.append("\nChecklist Items:")
            for item in items:
                status = item.get("status", "?")
                icon = {"Completed": "\u2705", "Partially Completed": "\u26a0\ufe0f", "Not Done": "\u274c"}.get(status, "\u2014")
                item_text = item.get("item", "")
                evidence = item.get("evidence", "")
                cat = item.get("category", "")
                lines.append(f"  {icon} [{cat}] {item_text}")
                if evidence and evidence.strip():
                    lines.append(f"     \u2192 {evidence}")

    return "\n".join(lines)
