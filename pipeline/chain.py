"""
Call chaining — compose multiple per-call outputs into an ordered series
with a combined transcript. Each call still processes independently; this
module is a composition layer on top of existing artifacts.

Storage model:
    outputs/
    ├── <existing per-call dirs>     # each gains an optional "chain" block
    └── __chains__/<chain_id>/
        ├── manifest.json
        └── chain_transcript.json

Chains are "open" — they can grow via append, shrink via remove, or be closed
to signal that no more calls will be added.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
import uuid
from datetime import datetime, timezone


# Module-level paths. Tests monkeypatch these; production code reads the live
# values via the module, not a cached import.
OUTPUTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
CHAINS_DIR: str = os.path.join(OUTPUTS_DIR, "__chains__")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str | None) -> str:
    """Lowercase, ASCII-only, hyphen-separated. Empty string for empty input."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return cleaned


def _manifest_path(chain_id: str) -> str:
    return os.path.join(CHAINS_DIR, chain_id, "manifest.json")


def _chain_transcript_path(chain_id: str) -> str:
    return os.path.join(CHAINS_DIR, chain_id, "chain_transcript.json")


def _chain_summary_path(chain_id: str) -> str:
    return os.path.join(CHAINS_DIR, chain_id, "chain_summary.json")


def _load_manifest(chain_id: str) -> dict | None:
    p = _manifest_path(chain_id)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: dict) -> None:
    manifest["updated_at"] = _now_iso()
    os.makedirs(os.path.join(CHAINS_DIR, manifest["id"]), exist_ok=True)
    with open(_manifest_path(manifest["id"]), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _call_transcript_path(call_dir: str) -> str:
    return os.path.join(OUTPUTS_DIR, call_dir, "transcript.json")


def _load_call_transcript(call_dir: str) -> dict | None:
    p = _call_transcript_path(call_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_call_transcript(call_dir: str, data: dict) -> None:
    p = _call_transcript_path(call_dir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_chain_block(call_dir: str, manifest: dict, index: int) -> None:
    """Stamp the call's transcript with chain membership."""
    data = _load_call_transcript(call_dir)
    if data is None:
        return
    data["chain"] = {
        "id": manifest["id"],
        "slug": manifest["slug"],
        "index": index,
        "label": manifest.get("label"),
    }
    _save_call_transcript(call_dir, data)


def _clear_chain_block(call_dir: str) -> None:
    data = _load_call_transcript(call_dir)
    if data is None:
        return
    data.pop("chain", None)
    _save_call_transcript(call_dir, data)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def create_chain(label: str | None = None, customer_identifier: str | None = None) -> dict:
    """Create a new empty chain manifest. Returns the manifest dict."""
    chain_id = str(uuid.uuid4())
    slug = _slugify(label) or _slugify(customer_identifier) or f"chain-{chain_id[:8]}"
    manifest = {
        "id": chain_id,
        "slug": slug,
        "label": label,
        "customer_identifier": customer_identifier,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "closed": False,
        "calls": [],
    }
    _save_manifest(manifest)
    return manifest


def get_chain(chain_id: str) -> dict | None:
    """Read a chain manifest. None if it doesn't exist."""
    return _load_manifest(chain_id)


def list_chains() -> list[dict]:
    """All chain manifests, newest first."""
    if not os.path.exists(CHAINS_DIR):
        return []
    out = []
    for name in os.listdir(CHAINS_DIR):
        m = _load_manifest(name)
        if m:
            out.append(m)
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


def append_call_to_chain(chain_id: str, call_dir: str, index: int | None = None) -> dict:
    """Add a completed call to a chain.

    `call_dir` is the directory *name* under outputs/ (not a full path).
    `index` defaults to the next slot (N+1).
    """
    manifest = _load_manifest(chain_id)
    if manifest is None:
        raise ValueError(f"Chain not found: {chain_id}")

    existing = {c["dir"] for c in manifest["calls"]}
    if call_dir in existing:
        return manifest  # already in chain; idempotent

    next_index = index if index is not None else len(manifest["calls"]) + 1
    manifest["calls"].append({
        "index": next_index,
        "dir": call_dir,
        "added_at": _now_iso(),
    })
    # Keep calls ordered by index
    manifest["calls"].sort(key=lambda c: c["index"])
    _save_manifest(manifest)

    _write_chain_block(call_dir, manifest, next_index)
    rebuild_chain_transcript(chain_id)
    return manifest


def remove_call_from_chain(chain_id: str, call_dir: str) -> dict:
    """Remove a call from a chain and renumber the remaining members."""
    manifest = _load_manifest(chain_id)
    if manifest is None:
        raise ValueError(f"Chain not found: {chain_id}")

    manifest["calls"] = [c for c in manifest["calls"] if c["dir"] != call_dir]
    # Renumber 1..N
    for new_idx, call in enumerate(manifest["calls"], start=1):
        call["index"] = new_idx
    _save_manifest(manifest)

    _clear_chain_block(call_dir)
    for call in manifest["calls"]:
        _write_chain_block(call["dir"], manifest, call["index"])

    rebuild_chain_transcript(chain_id)
    return manifest


def close_chain(chain_id: str) -> dict:
    manifest = _load_manifest(chain_id)
    if manifest is None:
        raise ValueError(f"Chain not found: {chain_id}")
    manifest["closed"] = True
    _save_manifest(manifest)
    return manifest


def reopen_chain(chain_id: str) -> dict:
    manifest = _load_manifest(chain_id)
    if manifest is None:
        raise ValueError(f"Chain not found: {chain_id}")
    manifest["closed"] = False
    _save_manifest(manifest)
    return manifest


def delete_chain(chain_id: str) -> None:
    """Delete the chain manifest dir. Member calls are untouched except
    their `chain` block is cleared so they surface as singles again."""
    manifest = _load_manifest(chain_id)
    if manifest is None:
        return
    for call in manifest["calls"]:
        _clear_chain_block(call["dir"])
    shutil.rmtree(os.path.join(CHAINS_DIR, chain_id), ignore_errors=True)


def rebuild_chain_transcript(chain_id: str) -> dict:
    """Concatenate member transcripts into a single chain-level view.

    Skips members that are missing or corrupt (e.g., a queued call whose
    processing hasn't finished yet). The chain is a live composition — a
    partial chain is a valid chain.
    """
    manifest = _load_manifest(chain_id)
    if manifest is None:
        raise ValueError(f"Chain not found: {chain_id}")

    entries: list[dict] = []
    chain_cursor = 0.0

    for call in manifest["calls"]:
        data = _load_call_transcript(call["dir"])
        if data is None:
            continue  # member not ready yet — skip silently

        duration = float(data.get("metadata", {}).get("duration_seconds", 0.0) or 0.0)
        date_hint = data.get("metadata", {}).get("original_filename", call["dir"])

        entries.append({
            "type": "call_boundary",
            "call_index": call["index"],
            "dir": call["dir"],
            "duration": duration,
            "chain_t0": chain_cursor,
            "label": date_hint,
        })

        for seg in data.get("segments", []):
            t0 = float(seg.get("t0", 0.0))
            entries.append({
                "type": "segment",
                "call_index": call["index"],
                "id": seg.get("id"),
                "t0": t0,
                "t1": float(seg.get("t1", t0)),
                "chain_t0": chain_cursor + t0,
                "speaker": seg.get("speaker"),
                "role": seg.get("role"),
                "text": seg.get("text", ""),
                "original_text": seg.get("original_text"),
            })

        chain_cursor += duration

    out = {
        "chain": {
            "id": manifest["id"],
            "slug": manifest["slug"],
            "label": manifest.get("label"),
            "total_calls": len(manifest["calls"]),
            "closed": manifest.get("closed", False),
        },
        "entries": entries,
    }

    os.makedirs(os.path.join(CHAINS_DIR, chain_id), exist_ok=True)
    with open(_chain_transcript_path(chain_id), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


# -----------------------------------------------------------------------------
# Chain-level summary (Slice 2)
# -----------------------------------------------------------------------------

# Per-call transcript excerpt sizes when we need to trim (very long chains).
_TRANSCRIPT_HEAD_TAIL_SEGS = 30
_TRANSCRIPT_FULL_THRESHOLD_SEGS = 600  # chains ≤ this many total segments get full fidelity


def _manifest_signature(chain_id: str) -> str:
    """
    Deterministic signature of a chain's "shape" — id list, ordering, and the
    sha256 of each member's transcript.json. Changes when any member is added,
    removed, reordered, or its transcript content changes.
    """
    manifest = _load_manifest(chain_id)
    if manifest is None:
        return ""
    h = hashlib.sha256()
    h.update(manifest["id"].encode())
    for call in manifest["calls"]:
        h.update(f"|{call['index']}|{call['dir']}".encode())
        p = _call_transcript_path(call["dir"])
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _prepare_chain_prompt_inputs(chain_id: str) -> dict | None:
    """
    Assemble the three things the chain-summary LLM call needs:
    - `per_call_summaries`: list of {call_index, duration, summary}
    - `transcript_text`: combined transcript string, each segment line prefixed
      with `[S<id>/C<index>]` so the LLM can cite exactly
    - `chain_context`: {call_index: segments[]} for the citation resolver
    Returns None if the chain has no members with transcripts yet.
    """
    manifest = _load_manifest(chain_id)
    if manifest is None or not manifest["calls"]:
        return None

    per_call: list[dict] = []
    chain_context: dict[int, list[dict]] = {}
    lines: list[str] = []
    total_segments = 0
    loaded_any = False

    # First pass: count total segments so we can decide full-vs-trim.
    loaded_transcripts: list[tuple[int, str, dict]] = []  # (call_index, dir, data)
    for call in manifest["calls"]:
        data = _load_call_transcript(call["dir"])
        if data is None:
            continue
        loaded_transcripts.append((call["index"], call["dir"], data))
        total_segments += len(data.get("segments", []))
    if not loaded_transcripts:
        return None

    use_full = total_segments <= _TRANSCRIPT_FULL_THRESHOLD_SEGS

    for call_index, call_dir, data in loaded_transcripts:
        loaded_any = True
        segments = data.get("segments", [])
        chain_context[call_index] = segments
        duration = float(data.get("metadata", {}).get("duration_seconds", 0.0) or 0.0)

        per_call.append({
            "call_index": call_index,
            "dir": call_dir,
            "duration_seconds": duration,
            "summary": data.get("summary"),
        })

        # Boundary label for the LLM
        lines.append(f"\n=== Call {call_index} (dir={call_dir}, duration={duration:.0f}s) ===")

        if use_full:
            slice_segs = segments
        else:
            # Head + tail excerpt when transcript is too big to fit.
            if len(segments) <= 2 * _TRANSCRIPT_HEAD_TAIL_SEGS:
                slice_segs = segments
            else:
                head = segments[:_TRANSCRIPT_HEAD_TAIL_SEGS]
                tail = segments[-_TRANSCRIPT_HEAD_TAIL_SEGS:]
                slice_segs = head + [{"_excerpt_gap": True}] + tail

        for seg in slice_segs:
            if seg.get("_excerpt_gap"):
                lines.append("    … [transcript excerpt trimmed for length] …")
                continue
            role = (seg.get("role") or seg.get("speaker") or "UNK").upper()
            sid = seg.get("id")
            t0 = float(seg.get("t0", 0.0))
            mins, secs = int(t0 // 60), int(t0 % 60)
            text = (seg.get("text") or "").strip()
            lines.append(f"[S{sid}/C{call_index}] [{mins:02d}:{secs:02d}] {role}: {text}")

    if not loaded_any:
        return None

    return {
        "per_call_summaries": per_call,
        "transcript_text": "\n".join(lines),
        "chain_context": chain_context,
        "truncated": not use_full,
        "total_segments": total_segments,
        "chain_label": manifest.get("label") or manifest.get("slug"),
    }


def _build_chain_summary_prompt(inputs: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the chain-level summarizer."""
    freed_context = (
        "FREED DOMAIN CONTEXT:\n"
        "- DRP: settlement at ~45% of enrolled debt. Service fee: 15%+GST. "
        "Total SPA savings required: ~62.7%.\n"
        "- DCP: multiple EMIs merged into one via third-party lender.\n"
        "- DEP: structured accelerated repayment plan.\n"
        "- SPA = Special Purpose Account (escrow for settlements).\n"
        "- CHPP / FREED Shield / i-FREED are ancillary products.\n"
        "- Common compliance points: borrower rights, credit score impact, "
        "  accurate fee/settlement figures, no false promises.\n"
    )

    system_prompt = (
        "You are a senior sales/audit analyst reviewing a SERIES of related calls "
        "between the same FREED customer and one or more agents. Your output "
        "summarizes the ARC ACROSS CALLS — not each call separately.\n\n"
        + freed_context +
        "\n"
        "HARD REQUIREMENTS:\n"
        "- Cite every specific detail with a compound reference `[S<id>/C<index>]` "
        "  copied EXACTLY from the `[S<id>/C<index>]` prefixes in the transcript below.\n"
        "- NEVER invent IDs. NEVER cite with timestamps `[MM:SS]` or single-call `[S<id>]`.\n"
        "- If something is unclear from the transcript, write 'Not clearly specified in the chain.'\n"
        "- Preserve financial numbers exactly as spoken.\n"
        "- Describe the ARC: what changed between calls, what objections were "
        "  raised/resolved, how sentiment moved, what finally happened.\n"
        "- Critical moments should be genuinely pivotal — 3 to 6 total across the series.\n"
    )

    per_call_block = json.dumps(
        [{"call_index": p["call_index"], "summary": p.get("summary")} for p in inputs["per_call_summaries"]],
        ensure_ascii=False,
        indent=2,
    )

    truncation_note = (
        "\n\nNOTE: Transcript is excerpted (head + tail per long call) to fit the context window. "
        "Per-call summaries above are authoritative; transcript excerpts are for citation anchoring.\n"
        if inputs.get("truncated") else ""
    )

    user_prompt = (
        f"Chain label: {inputs.get('chain_label') or 'Unlabeled chain'}\n"
        f"Total calls: {len(inputs['per_call_summaries'])}\n"
        f"Total segments: {inputs['total_segments']}\n\n"
        f"Per-call summaries (pre-computed):\n{per_call_block}\n"
        f"{truncation_note}\n"
        f"Combined transcript (with compound [S/C] citations to copy into your output):\n"
        f"{inputs['transcript_text']}"
    )
    return system_prompt, user_prompt


def summarize_chain(chain_id: str, cfg) -> dict:
    """
    Generate a chain-level structured summary. Blocking LLM call.
    On success writes `chain_summary.json` and returns:
        { summary: {...}, generated_at, manifest_sig, stale: False }
    On empty chain returns { error: ... }.
    """
    from pydantic import BaseModel, Field
    import openai

    inputs = _prepare_chain_prompt_inputs(chain_id)
    if inputs is None:
        return {"error": "Chain has no members with transcripts yet."}

    # --- Pydantic schema --------------------------------------------------
    class ChainCallBreakdown(BaseModel):
        call_index: int
        role_in_arc: str
        one_line_outcome: str

    class ObjectionLifecycle(BaseModel):
        objection: str
        raised_on_call: int
        raised_quote: str = Field(description="Verbatim quote with [S<id>/C<index>] citation.")
        resolved_on_call: int | None
        resolution_quote: str | None
        status: str = Field(description="Resolved / Partially Resolved / Unresolved / Deflected")

    class RecurringTopic(BaseModel):
        topic: str
        mentioned_on_calls: list[int]
        significance: str

    class SentimentPoint(BaseModel):
        call_index: int
        sentiment: str
        direction_from_prev: str
        evidence: str = Field(description="Quote with [S<id>/C<index>] citation.")

    class ConversionPoint(BaseModel):
        call_index: int
        lead_probability: str
        pivotal_moment: str = Field(description="With [S<id>/C<index>] citation.")

    class ComplianceSnapshot(BaseModel):
        items_covered_in_chain: list[str]
        items_missed_entirely: list[str]
        compliance_flags_by_call: str

    class CriticalMoment(BaseModel):
        call_index: int
        timestamp_in_call: str
        description: str
        quote: str = Field(description="Verbatim with [S<id>/C<index>] citation.")

    class ChainSummary(BaseModel):
        series_arc: str
        final_outcome: str
        outcome_evidence: str
        total_calls: int
        spanning_duration_summary: str
        call_breakdown: list[ChainCallBreakdown]
        customer_initial_state: str
        customer_final_state: str
        sentiment_trajectory: list[SentimentPoint]
        objections_lifecycle: list[ObjectionLifecycle]
        recurring_topics: list[RecurringTopic]
        agent_consistency: str
        compliance_snapshot: ComplianceSnapshot
        chain_quality_score: str
        conversion_trajectory: list[ConversionPoint]
        critical_moments: list[CriticalMoment]

    # --- LLM call ---------------------------------------------------------
    system_prompt, user_prompt = _build_chain_summary_prompt(inputs)

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
                {"role": "user", "content": user_prompt},
            ],
            response_format=ChainSummary,
        )
        raw = json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Chain summarization failed: {e}"}

    # --- Resolve compound [S/C] citations to clickable links --------------
    from pipeline.reconstruct import verify_and_inject_inline_citations
    resolved = verify_and_inject_inline_citations(
        raw, transcript_segments=[], chain_context=inputs["chain_context"],
    )

    payload = {
        "generated_at": _now_iso(),
        "manifest_sig": _manifest_signature(chain_id),
        "summary": resolved,
    }

    os.makedirs(os.path.join(CHAINS_DIR, chain_id), exist_ok=True)
    with open(_chain_summary_path(chain_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {**payload, "stale": False}


def get_chain_summary(chain_id: str) -> dict:
    """
    Return the cached chain summary with a staleness indicator. Does NOT
    trigger generation. Returned shape:
        { summary: dict|None, generated_at: str|None, stale: bool, reason: str|None }
    """
    p = _chain_summary_path(chain_id)
    if not os.path.exists(p):
        return {"summary": None, "generated_at": None, "stale": True, "reason": "never_generated"}

    try:
        with open(p, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"summary": None, "generated_at": None, "stale": True, "reason": "cache_corrupt"}

    current_sig = _manifest_signature(chain_id)
    stale = cached.get("manifest_sig") != current_sig
    return {
        "summary": cached.get("summary"),
        "generated_at": cached.get("generated_at"),
        "stale": stale,
        "reason": "chain_mutated" if stale else None,
    }
