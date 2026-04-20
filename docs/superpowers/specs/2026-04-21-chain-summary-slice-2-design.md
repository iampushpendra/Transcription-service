# Call Chaining — Slice 2: Chain Summary

**Date:** 2026-04-21
**Status:** Approved, implementing
**Scope:** One LLM-generated chain-level summary that understands the arc across calls. Per-call summaries remain untouched.

---

## Goals

1. A single structured summary that describes **what happened across the chain**, not a concatenation of per-call summaries.
2. Citations link back to the exact segment of the exact call via the compound `[S<id>/C<index>]` format.
3. Generation is explicit (costly LLM call) — triggered from the dashboard or when the chain is closed.
4. Result is cached in `chain_summary.json` and served fast from disk; staleness is surfaced to the UI when the chain has mutated since generation.

## Schema

```python
class ChainCallBreakdown(BaseModel):
    call_index: int
    role_in_arc: str          # "Initial pitch", "Objection handling", "Close attempt", etc.
    one_line_outcome: str     # "Customer asked for time to think"

class ObjectionLifecycle(BaseModel):
    objection: str            # Short label: "CIBIL score concern"
    raised_on_call: int
    raised_quote: str         # Verbatim, with [S/C] citation
    resolved_on_call: int | None
    resolution_quote: str | None
    status: str               # 'Resolved' | 'Partially Resolved' | 'Unresolved' | 'Deflected'

class RecurringTopic(BaseModel):
    topic: str
    mentioned_on_calls: list[int]
    significance: str         # Why recurrence matters (customer forgot / agent re-pitched / unresolved)

class SentimentPoint(BaseModel):
    call_index: int
    sentiment: str            # 'Distressed', 'Hopeful', 'Skeptical', etc.
    direction_from_prev: str  # 'improved' | 'worsened' | 'stable' | 'n/a'
    evidence: str             # Quote + compound citation

class ConversionPoint(BaseModel):
    call_index: int
    lead_probability: str     # 'High' | 'Medium' | 'Low' | 'Not Applicable'
    pivotal_moment: str       # What shifted the trajectory, with [S/C] citation

class ComplianceSnapshot(BaseModel):
    items_covered_in_chain: list[str]       # e.g. "Borrower rights — Call 2 [S45/C2]"
    items_missed_entirely: list[str]        # Items never covered across any call
    compliance_flags_by_call: str           # Per-call audit issues consolidated

class CriticalMoment(BaseModel):
    call_index: int
    timestamp_in_call: str
    description: str
    quote: str                # With [S<id>/C<index>] compound citation

class ChainSummary(BaseModel):
    # Top line
    series_arc: str           # 2-4 sentence narrative across the series
    final_outcome: str        # 'Enrolled', 'Dropped off', 'Awaiting decision', etc.
    outcome_evidence: str     # Quote from last call with citation

    # Composition
    total_calls: int
    spanning_duration_summary: str  # "4 calls over ~2 weeks"
    call_breakdown: list[ChainCallBreakdown]

    # Customer journey
    customer_initial_state: str
    customer_final_state: str
    sentiment_trajectory: list[SentimentPoint]

    # Issues & resolution
    objections_lifecycle: list[ObjectionLifecycle]
    recurring_topics: list[RecurringTopic]

    # Agent / compliance
    agent_consistency: str                  # "Same agent throughout" or "Handed off at Call 3"
    compliance_snapshot: ComplianceSnapshot
    chain_quality_score: str                # "Strong / Adequate / Needs Improvement" + one line

    # Conversion
    conversion_trajectory: list[ConversionPoint]

    # Evidence anchors
    critical_moments: list[CriticalMoment]  # 3-6 pivotal moments across calls
```

## Prompt strategy

**Input to LLM:**
- FREED domain context (same as `summarize_call_structured`)
- Per-call summaries from member transcripts (already structured — LLM gets a pre-digested take)
- Combined transcript excerpts **tagged with compound IDs** — each line is `[S<id>/C<index>] [MM:SS] ROLE: text`
- A directive to cite every specific detail with `[S<id>/C<index>]` — never bare `[S<id>]` or `[MM:SS]`

**Size management:**
- Budget: ~60k tokens for input (gpt-5-mini context)
- If combined transcript > budget: truncate — keep full per-call summaries + heated moments + boundary windows (first/last 10 segments per call)
- Full-fidelity path remains for small chains (≤ 3 short calls)

**Citation contract:**
- LLM outputs `[S<id>/C<index>]` only
- Post-processing via `verify_and_inject_inline_citations(data, [], chain_context=…)` resolves to exact `scrollToTime(<t0>)` links
- Invalid citations drop silently

## Caching & staleness

`__chains__/<id>/chain_summary.json` holds:
```json
{
  "generated_at": "…",
  "manifest_sig": "<sha256 of manifest.json contents>",
  "summary": { …ChainSummary… }
}
```

On read:
- If file missing → `stale: true`, `reason: "never_generated"`
- If stored `manifest_sig` differs from current manifest's sig → `stale: true`, `reason: "chain_mutated"`
- Else → `stale: false`

Staleness is reported but the stored summary is **still returned** — the UI can show "Last generated X, chain changed since — Regenerate?".

## API

| Method | Path | Behavior |
|---|---|---|
| POST | `/api/chain/{id}/summarize` | Blocking LLM generation (~30–120s). Returns full summary on success. 409 if already generating. |
| GET | `/api/chain/{id}/summary` | Returns `{ summary, generated_at, stale, reason }`. Never triggers generation. |

No auto-generation on `close` in this slice — keeps cost explicit. Future: auto-trigger on close via a background task.

## Frontend

Chain dashboard gets a **Chain Summary** tab/panel beside the combined transcript. States:
- **Empty:** "This chain has no summary yet. [Generate Summary]" button.
- **Generating:** spinner + elapsed time.
- **Ready:** renders sections of ChainSummary. Stale banner + Regenerate button if `stale=true`.
- **Failed:** retry button, error text.

All compound citations (`[S/C]`) come pre-resolved as `<a onclick="scrollToTime(<t0>)">` links that jump into the combined transcript panel.

## Non-goals (this slice)

- Streaming generation / partial updates
- Background/auto-generation on close
- Per-chain LLM cost accounting
- Summary editing / user-curation
- Cross-chain summaries

## Tests

- **Unit:**
  - Manifest sig round-trips deterministically; mutating manifest invalidates sig
  - `_prepare_chain_prompt_inputs` selects per-call summaries + transcript slices correctly
  - Compound citation resolution on synthetic LLM output (full `[S/C]` + mixed `[S]`/`[S/C]` inputs)
  - Schema round-trip: dict → `ChainSummary` → dict
- **Integration (mocked LLM):** stub OpenAI client, verify end-to-end that `summarize_chain` produces a valid resolved summary, writes `chain_summary.json`, and staleness works.
