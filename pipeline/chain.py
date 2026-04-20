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
