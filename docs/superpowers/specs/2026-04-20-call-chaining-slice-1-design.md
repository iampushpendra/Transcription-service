# Call Chaining — Slice 1 Design

**Date:** 2026-04-20
**Status:** Approved, implementing
**Scope:** Primitive chain infrastructure + combined transcript + UI for creating/viewing chains. Chain-level LLM intelligence (evolution, chain summary, arc analysis) is Slice 2.

---

## Goals

1. Users upload N calls in order and see one coherent analysis across the series.
2. Singular-call flow is entirely unchanged.
3. Chains are **open** — a 4-call chain can grow to 5 calls weeks later without reprocessing.
4. Per-call ML (ASR, diarization, emotion) is unchanged — chains are a composition layer on top.

## Data model

**No new storage engine.** Chains are manifests that reference existing per-call output directories.

```
outputs/
├── <existing per-call dirs, untouched schema except optional `chain` block>
└── __chains__/
    └── <chain_id>/
        ├── manifest.json          # ordered member list + metadata
        ├── chain_transcript.json  # cached combined transcript (rebuilt on mutate)
        └── chain_summary.json     # Slice 2 — placeholder for now
```

**`manifest.json`:**
```json
{
  "id": "<uuid4>",
  "slug": "gaurav-bhatotia-2026-04-20",
  "label": "Gaurav Bhatotia — enrollment",
  "customer_identifier": "9695942655",
  "created_at": "2026-04-20T17:03:31Z",
  "updated_at": "2026-04-20T17:45:12Z",
  "closed": false,
  "calls": [
    { "index": 1, "dir": "Call-1-..._3f5608", "added_at": "..." }
  ]
}
```

Optional `chain` block on each member's `transcript.json`:
```json
"chain": { "id": "...", "slug": "...", "index": 2, "label": "..." }
```

Segment IDs remain scoped to each call (`S0..S<n>`).

## Combined transcript (`chain_transcript.json`)

Rebuilt on any chain mutation. Each entry is either a segment or a call-boundary marker:

```json
{
  "chain": { "id": "...", "label": "...", "total_calls": 5 },
  "entries": [
    { "type": "call_boundary", "call_index": 1, "dir": "...", "duration": 1721.7, "date": "..." },
    { "type": "segment", "call_index": 1, "id": 0, "t0": 0.0, "t1": 4.5, "chain_t0": 0.0, "role": "agent", "text": "..." },
    ...
    { "type": "call_boundary", "call_index": 2, ... },
    { "type": "segment", "call_index": 2, "id": 0, "t0": 0.0, "chain_t0": 1721.7, ... }
  ]
}
```

`chain_t0` = cumulative offset using each prior call's `duration_seconds` from its metadata.

## Compound citations

Citations within chain-level artifacts (Slice 2 mostly, but plumbed now):
- `[S42/C2]` — segment 42 of call 2.
- `**Keyword**[S42/C2]` — keyword wrapped in link.

Resolver extension: `verify_and_inject_inline_citations` accepts an optional `chain_context` argument holding `{call_index → segments[]}` and resolves compound cites to the correct call's segment t0. Dashboard clicks route to the right call's view.

## API

```
POST   /api/chain                 body: { label?, customer_identifier?, files[] (multipart), order[] }
                                  → creates chain, enqueues each file as a job tagged with chain+index
                                  → returns { chain_id, slug, call_job_ids }

GET    /api/chain/{id}            → manifest + per-call job statuses

POST   /api/chain/{id}/append     body: { file (multipart), index? (default: N+1) }
                                  → enqueues one call into existing chain

DELETE /api/chain/{id}            → removes __chains__/{id}/; member calls untouched

POST   /api/chain/{id}/close      → closed=true, triggers chain summary rebuild (Slice 2)
```

`GET /api/history` augmented to group chain members under a single row:
```json
{
  "chains": [ { ...manifest..., "member_calls": [ ...full call data... ] } ],
  "singles": [ ...unchained call data... ]
}
```
Old clients can still iterate a flat list by merging `chains[].member_calls + singles`.

## Upload flow

The existing queue worker processes jobs sequentially. A chain just creates N jobs in order, each carrying `chain_id` + `call_index`. When each job finishes:
1. Its `transcript.json` gets the `chain` block written.
2. The chain manifest's `updated_at` is touched.
3. `chain_transcript.json` is rebuilt from all completed members.

Partial chains are first-class: if call 3 is still processing, the chain has 2 entries so far.

## Frontend

**Upload panel** — two segmented tabs: *Single Call* / *Chain of Calls*.
- Chain tab: drop zone accepts multiple files; renders them as numbered, drag-reorderable cards.
- Optional fields: Label, Customer identifier (auto-populated from first filename if detectable).
- "Process Chain" submits to `/api/chain`.

**History panel** — two row types, visually distinct:
- Single call row (today's design).
- Chain row (new): label, `N calls`, total duration, status `X/N complete`, expand chevron reveals members, "Add Call" action button.

**Chain dashboard** — opened from the "View Combined" action on a chain row.
Four tabs:
1. **Combined Transcript** — scrollable concatenation with `=== Call K of N · <date> · <dur> ===` separators.
2. **Per-Call** — select-box to jump to any member's individual existing dashboard.
3. **Chain Summary** — placeholder card saying *"Chain-level summary: coming in Slice 2"* so Slice 1 ships with a clean empty state.
4. **Evolution View** — same placeholder.

## Edge cases (Slice 1 handling)

| Case | Handling |
|---|---|
| Same file twice in one chain | SHA-256 on upload; reject with clear error. |
| Upload mid-chain fails | Call row shows error + retry; chain holds partial artifacts; rebuild retries on retry. |
| User deletes a member call | Manifest re-indexed; `chain_transcript.json` rebuilt. |
| User deletes chain | Manifest + chain dir removed; member calls remain as singles. |
| Out-of-order upload | Respect user order; soft warning if recording dates disagree. |
| Incremental append | Natively supported via `/append`. |
| Singular upload via single-call tab | No chain dir; no `chain` block; fully unchanged. |

## Out of scope (deferred to Slice 2)

- `summarize_chain()` LLM call + chain summary schema
- Evolution view real content (cross-call objection tracking, sentiment arc)
- Chain-level 21-point checklist rollup
- Objection-lifecycle tracking
- Recurring-topic detection

## Test plan

- **Unit (backend):**
  - manifest CRUD: create, append, remove-call, delete, reopen, slugify
  - `rebuild_chain_transcript` with 0/1/2/N member calls, varying durations
  - compound-cite resolver `[S42/C2]` with valid/invalid indices
  - `chain` block written onto transcript.json on job completion
- **Integration:**
  - End-to-end through 2 fake calls (stubbed audio or short real files)
  - `/api/chain` + `/api/chain/{id}` + `/api/chain/{id}/append` round-trip
  - Chain deletion preserves member calls
- **Frontend:**
  - Manual smoke test of upload flow
  - Reorder cards → order persists in submit payload
  - Chain row expand/collapse + "Add Call" modal
  - Chain dashboard loads combined transcript with correct boundary markers
