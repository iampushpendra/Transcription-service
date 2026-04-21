# Transcript Citation Redirection — Fix Design

**Date:** 2026-04-20
**Owner:** Rajat Kumawat
**Status:** Approved, implementing
**Scope:** Fix "summary insight → wrong transcript segment" redirection.

---

## Problem

Clicking an insight link in the dashboard summary sometimes jumps to the wrong `.message` in the transcript timeline.

## Root causes (current state)

1. **`verify_and_inject_inline_citations` is not wired into the live paths** (`server.py`, `run.py`). Only `scripts/regenerate_summaries.py` uses it. Live summaries reach the frontend as raw LLM text with `[MM:SS]` citations, handled entirely by `formatMarkdownAndTimestamps`/`scrollToTime`.
2. **`standalone_pattern` at `reconstruct.py:791` is compiled but never applied.** Dead code.
3. **±60-second keyword search window** (`reconstruct.py:812`) is too wide — common terms (CIBIL, DRP, settlement) repeat many times per call and the snap picks the nearest *instance*, not the cited one.
4. **`t1`-closest returns `t0`** (`reconstruct.py:840-843`). Landing position drifts several seconds on long merged segments.
5. **MM:SS precision loss.** LLM emits minute:second with no sub-second; boundary quotes fall into adjacent segments after rounding.
6. **Frontend fallback reinforces the bug** (`index.html:2002`) — same `min(|t0-s|, |t1-s|)` pattern.

## Solution — Segment-ID citations with MM:SS fallback

**Primary change:** LLM cites via opaque segment IDs (`[S42]`) which resolve exactly. Removes the fuzzy-match failure class.

**Belt & braces:** Harden the MM:SS path too, so any stray `[MM:SS]` still lands correctly.

### Data contract

- `reconstruct()` assigns `segments[i].id = i` after merge/dedup. Stable integer, zero-indexed.
- `segments[].id` is persisted in `transcript.json`.

### Prompt contract

- System prompt: *"CITATIONS: Every specific detail must end with a segment reference like `[S42]`, taken exactly from the numbered transcript below. Never invent IDs. Never use timestamps."*
- Transcript prepended with `[S<id>]` prefix per line.
- All `Field(description=...)` strings updated from `[MM:SS]` → `[S<id>]`.

### Resolver (`verify_and_inject_inline_citations`)

Primary:
- `**Keyword**[S<id>]` → `<a onclick="scrollToTime(<t0>)">Keyword</a>` (wrap keyword).
- `[S<id>]` → `<a onclick="scrollToTime(<t0>)">[↗]</a>` (superscript arrow).
- Invalid or out-of-range ID → drop citation, preserve surrounding text.

Fallback (hardened MM:SS, same tag output):
- Keyword window: ±10s (was ±60s).
- Keyword match required — no keyword, no link.
- Return the *actually closer* anchor (`t0` vs `t1`).
- Apply to standalone `[MM:SS]` too.

### Integration

- Call `verify_and_inject_inline_citations(summary_dict, transcript)` right after `summarize_call_structured` in `server.py` and `run.py`.

### Frontend (`scrollToTime`)

- Mirror the server-side snap anchor fix (return the closer anchor).
- Add `data-seg-id` on message divs for future use.
- No other changes needed — summary arrives pre-resolved with exact `t0` values baked into `onclick`.

### Error handling

- Any resolve failure degrades to plain text; never raises.
- `stderr` warnings for dropped citations (debugging aid).

## Testing

New `tests/test_citation_resolver.py`:
- Valid `[S0]`, `[S42]` → correct t0.
- Invalid `[S999]`, `[Sxyz]`, `[S]`, `[S-1]` → dropped, no crash.
- `**CIBIL**[S3]` → keyword wrapped in link.
- Bare `[S3]` → superscript arrow rendered.
- Fallback `[02:45]` with nearby keyword in ±10s window → snapped.
- Fallback `[02:45]` with no keyword nearby → dropped.
- Nested dict/list recursion preserved.

## Out of scope (this spec)

- Re-running the resolver over legacy `outputs/` summaries.
- Any change to emotion/triggers/sarcasm pipelines.
- Other three features (compliance flagging, call chaining, keyword updates) — separate specs.
