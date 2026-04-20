"""Tests for chain-level summary generation (Slice 2)."""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from pipeline import chain as chain_mod


@pytest.fixture
def tmp_outputs(monkeypatch):
    d = tempfile.mkdtemp(prefix="chainsum_")
    monkeypatch.setattr(chain_mod, "OUTPUTS_DIR", d)
    monkeypatch.setattr(chain_mod, "CHAINS_DIR", os.path.join(d, "__chains__"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _seed_chain(outputs_dir, n_calls=2):
    """Create a chain with N minimal member calls; returns (chain_id, call_dirs)."""
    from datetime import datetime, timezone
    call_dirs = []
    for i in range(n_calls):
        name = f"call_{i + 1}"
        d = os.path.join(outputs_dir, name)
        os.makedirs(d, exist_ok=True)
        segs = [
            {"id": 0, "t0": 0.0, "t1": 5.0, "role": "agent",    "speaker": "a", "text": f"Call {i + 1} greeting"},
            {"id": 1, "t0": 5.0, "t1": 12.0, "role": "customer", "speaker": "c", "text": f"Call {i + 1} customer statement"},
            {"id": 2, "t0": 12.0, "t1": 20.0, "role": "agent",   "speaker": "a", "text": f"Call {i + 1} agent pitch"},
        ]
        data = {
            "metadata": {"duration_seconds": 100.0 + i * 50, "original_filename": f"{name}.mp3"},
            "segments": segs,
            "summary": {
                "call_type": "Follow-Up Call" if i > 0 else "First Call",
                "overview": f"Overview for call {i + 1}. Customer raised concerns about CIBIL score.",
                "customer_analysis": {
                    "biggest_pain_point": "EMI burden",
                    "customer_state_of_mind": "Skeptical then Hopeful",
                },
                "call_categories": {
                    "lead_conversion_probability": "Medium",
                },
            },
        }
        with open(os.path.join(d, "transcript.json"), "w") as f:
            json.dump(data, f)
        call_dirs.append(name)

    m = chain_mod.create_chain(label=f"Test Chain {n_calls} calls", customer_identifier="test-user")
    for cd in call_dirs:
        chain_mod.append_call_to_chain(m["id"], cd)
    return m["id"], call_dirs


# ----------------------------------------------------------------------------
# Manifest signature for stale detection
# ----------------------------------------------------------------------------

def test_manifest_signature_is_deterministic(tmp_outputs):
    chain_id, _ = _seed_chain(tmp_outputs, n_calls=2)
    sig1 = chain_mod._manifest_signature(chain_id)
    sig2 = chain_mod._manifest_signature(chain_id)
    assert sig1 == sig2
    assert isinstance(sig1, str) and len(sig1) == 64  # sha256 hex


def test_manifest_signature_changes_when_chain_mutates(tmp_outputs):
    chain_id, _ = _seed_chain(tmp_outputs, n_calls=2)
    sig_before = chain_mod._manifest_signature(chain_id)

    # Add a third call
    d = os.path.join(tmp_outputs, "call_3")
    os.makedirs(d)
    with open(os.path.join(d, "transcript.json"), "w") as f:
        json.dump({"metadata": {"duration_seconds": 50.0}, "segments": []}, f)
    chain_mod.append_call_to_chain(chain_id, "call_3")

    sig_after = chain_mod._manifest_signature(chain_id)
    assert sig_before != sig_after


# ----------------------------------------------------------------------------
# Prompt input construction
# ----------------------------------------------------------------------------

def test_prepare_chain_inputs_gathers_per_call_summaries(tmp_outputs):
    chain_id, call_dirs = _seed_chain(tmp_outputs, n_calls=2)
    inputs = chain_mod._prepare_chain_prompt_inputs(chain_id)
    assert inputs is not None
    assert len(inputs["per_call_summaries"]) == 2
    assert inputs["per_call_summaries"][0]["call_index"] == 1
    assert "Overview for call 1" in inputs["per_call_summaries"][0]["summary"]["overview"]


def test_prepare_chain_inputs_labels_lines_with_compound_ids(tmp_outputs):
    chain_id, _ = _seed_chain(tmp_outputs, n_calls=2)
    inputs = chain_mod._prepare_chain_prompt_inputs(chain_id)
    text = inputs["transcript_text"]
    # Every segment line should start with [S<id>/C<index>]
    assert "[S0/C1]" in text
    assert "[S1/C1]" in text
    assert "[S0/C2]" in text
    # Boundaries should be labelled too
    assert "Call 1" in text
    assert "Call 2" in text


def test_prepare_chain_inputs_exposes_chain_context_for_resolver(tmp_outputs):
    chain_id, _ = _seed_chain(tmp_outputs, n_calls=2)
    inputs = chain_mod._prepare_chain_prompt_inputs(chain_id)
    ctx = inputs["chain_context"]
    assert set(ctx.keys()) == {1, 2}
    assert len(ctx[1]) == 3  # 3 segments per call
    assert ctx[1][0]["t0"] == 0.0


def test_prepare_chain_inputs_empty_chain_returns_none(tmp_outputs):
    m = chain_mod.create_chain(label="Empty")
    inputs = chain_mod._prepare_chain_prompt_inputs(m["id"])
    assert inputs is None


# ----------------------------------------------------------------------------
# End-to-end summarize with a mocked LLM
# ----------------------------------------------------------------------------

_STUB_SUMMARY = {
    "series_arc": "Customer progressed from skeptical to hopeful. Final **enrollment**[S2/C2] followed two calls.",
    "final_outcome": "Enrolled",
    "outcome_evidence": "Customer agreed at [S2/C2].",
    "total_calls": 2,
    "spanning_duration_summary": "2 calls",
    "call_breakdown": [
        {"call_index": 1, "role_in_arc": "Initial pitch", "one_line_outcome": "Customer raised CIBIL concern"},
        {"call_index": 2, "role_in_arc": "Close", "one_line_outcome": "Customer enrolled"},
    ],
    "customer_initial_state": "Skeptical",
    "customer_final_state": "Hopeful",
    "sentiment_trajectory": [
        {"call_index": 1, "sentiment": "Skeptical", "direction_from_prev": "n/a", "evidence": "See [S1/C1]."},
        {"call_index": 2, "sentiment": "Hopeful",   "direction_from_prev": "improved", "evidence": "See [S1/C2]."},
    ],
    "objections_lifecycle": [
        {
            "objection": "CIBIL concern",
            "raised_on_call": 1,
            "raised_quote": "Raised at [S1/C1].",
            "resolved_on_call": 2,
            "resolution_quote": "Addressed at [S2/C2].",
            "status": "Resolved",
        }
    ],
    "recurring_topics": [
        {"topic": "CIBIL score", "mentioned_on_calls": [1, 2], "significance": "Core worry"}
    ],
    "agent_consistency": "Same agent throughout",
    "compliance_snapshot": {
        "items_covered_in_chain": ["Borrower rights — Call 2 [S2/C2]"],
        "items_missed_entirely": ["None"],
        "compliance_flags_by_call": "No compliance flags identified",
    },
    "chain_quality_score": "Adequate — arc complete, minor gaps",
    "conversion_trajectory": [
        {"call_index": 1, "lead_probability": "Medium", "pivotal_moment": "Initial pitch at [S2/C1]"},
        {"call_index": 2, "lead_probability": "High",   "pivotal_moment": "Enrollment at [S2/C2]"},
    ],
    "critical_moments": [
        {"call_index": 1, "timestamp_in_call": "00:05", "description": "CIBIL raised", "quote": "Raised at [S1/C1]."},
        {"call_index": 2, "timestamp_in_call": "00:12", "description": "Enrollment",  "quote": "Confirmed at [S2/C2]."},
    ],
}


class _StubCompletion:
    """Shape-compatible stub of `client.beta.chat.completions.parse` response."""
    def __init__(self, payload):
        self.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]


@pytest.fixture
def mock_openai(monkeypatch):
    """Patch `openai.OpenAI(...).beta.chat.completions.parse` to return our stub."""
    stub = MagicMock()
    stub.beta.chat.completions.parse.return_value = _StubCompletion(_STUB_SUMMARY)

    def _fake_openai(*args, **kwargs):
        return stub

    monkeypatch.setattr("openai.OpenAI", _fake_openai)
    return stub


def _make_cfg():
    # Minimal config surface — chain.summarize reads only a handful of fields.
    cfg = MagicMock()
    cfg.openai_api_key = "sk-test"
    cfg.openai_summary_timeout_s = 30
    cfg.openai_max_retries = 0
    cfg.summary_model = "gpt-test"
    return cfg


def test_summarize_chain_writes_cache_file_and_returns_resolved(tmp_outputs, mock_openai):
    chain_id, _ = _seed_chain(tmp_outputs, n_calls=2)
    out = chain_mod.summarize_chain(chain_id, _make_cfg())

    # Cache file exists
    cache = os.path.join(tmp_outputs, "__chains__", chain_id, "chain_summary.json")
    assert os.path.exists(cache)
    with open(cache) as f:
        persisted = json.load(f)
    assert persisted["manifest_sig"] == chain_mod._manifest_signature(chain_id)
    assert "generated_at" in persisted
    assert persisted["summary"]["series_arc"]

    # Return payload
    assert out["stale"] is False
    # Compound citations resolved into <a onclick="scrollToTime(<t0>)">
    arc_html = out["summary"]["series_arc"]
    assert "scrollToTime(" in arc_html


def test_summarize_chain_resolves_compound_citations(tmp_outputs, mock_openai):
    chain_id, _ = _seed_chain(tmp_outputs, n_calls=2)
    out = chain_mod.summarize_chain(chain_id, _make_cfg())

    import re
    # [S1/C1] → call_1 seg id 1 t0=5.0; [S2/C2] → call_2 seg id 2 t0=12.0
    full_text = json.dumps(out["summary"])
    t0s = set(float(x) for x in re.findall(r"scrollToTime\(([\d.]+)\)", full_text))
    assert 5.0 in t0s   # [S1/C1]
    assert 12.0 in t0s  # [S2/C2]
    # No unresolved compound tokens should remain
    assert not re.search(r"\[S\d+/C\d+\]", full_text)


def test_get_chain_summary_reports_stale_after_mutation(tmp_outputs, mock_openai):
    chain_id, _ = _seed_chain(tmp_outputs, n_calls=2)
    chain_mod.summarize_chain(chain_id, _make_cfg())
    first = chain_mod.get_chain_summary(chain_id)
    assert first["stale"] is False

    # Mutate chain: add a third call
    d = os.path.join(tmp_outputs, "call_3")
    os.makedirs(d)
    with open(os.path.join(d, "transcript.json"), "w") as f:
        json.dump({"metadata": {"duration_seconds": 50.0}, "segments": []}, f)
    chain_mod.append_call_to_chain(chain_id, "call_3")

    second = chain_mod.get_chain_summary(chain_id)
    assert second["stale"] is True
    assert second["reason"] == "chain_mutated"
    assert second["summary"] is not None  # stale data still returned


def test_get_chain_summary_missing_reports_never_generated(tmp_outputs):
    chain_id, _ = _seed_chain(tmp_outputs, n_calls=2)
    result = chain_mod.get_chain_summary(chain_id)
    assert result["stale"] is True
    assert result["reason"] == "never_generated"
    assert result["summary"] is None


def test_summarize_chain_empty_chain_returns_error(tmp_outputs, mock_openai):
    m = chain_mod.create_chain(label="Empty")
    out = chain_mod.summarize_chain(m["id"], _make_cfg())
    assert "error" in out
