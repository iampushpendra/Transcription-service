"""Tests for pipeline/chain.py — chain manifest + combined transcript."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from pipeline import chain as chain_mod


@pytest.fixture
def tmp_outputs(monkeypatch):
    """Point chain module at a clean throwaway outputs/ dir."""
    d = tempfile.mkdtemp(prefix="chaintest_")
    monkeypatch.setattr(chain_mod, "OUTPUTS_DIR", d)
    monkeypatch.setattr(chain_mod, "CHAINS_DIR", os.path.join(d, "__chains__"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_call(outputs_dir, name, segments, duration):
    """Write a minimal transcript.json + matching dir under outputs_dir."""
    call_dir = os.path.join(outputs_dir, name)
    os.makedirs(call_dir, exist_ok=True)
    data = {
        "metadata": {"duration_seconds": duration, "original_filename": f"{name}.mp3"},
        "segments": segments,
        "summary": None,
    }
    with open(os.path.join(call_dir, "transcript.json"), "w") as f:
        json.dump(data, f)
    return call_dir


# ---- Manifest CRUD ----------------------------------------------------------

def test_create_chain_produces_uuid_slug_and_manifest(tmp_outputs):
    m = chain_mod.create_chain(label="Gaurav Bhatotia", customer_identifier="9695942655")
    assert m["id"]
    assert m["slug"] == "gaurav-bhatotia"
    assert m["label"] == "Gaurav Bhatotia"
    assert m["customer_identifier"] == "9695942655"
    assert m["closed"] is False
    assert m["calls"] == []
    assert os.path.exists(os.path.join(tmp_outputs, "__chains__", m["id"], "manifest.json"))


def test_slug_falls_back_to_customer_identifier(tmp_outputs):
    m = chain_mod.create_chain(label=None, customer_identifier="9695942655")
    assert m["slug"] == "9695942655"
    assert m["label"] is None


def test_slug_falls_back_to_chain_id_prefix_when_both_missing(tmp_outputs):
    m = chain_mod.create_chain(label=None, customer_identifier=None)
    assert m["slug"].startswith("chain-")
    assert len(m["slug"]) > len("chain-")


def test_get_chain_returns_persisted_state(tmp_outputs):
    m = chain_mod.create_chain(label="Test")
    fetched = chain_mod.get_chain(m["id"])
    assert fetched["id"] == m["id"]
    assert fetched["slug"] == m["slug"]


def test_get_chain_missing_returns_none(tmp_outputs):
    assert chain_mod.get_chain("not-a-real-id") is None


def test_append_call_to_chain_assigns_correct_index(tmp_outputs):
    m = chain_mod.create_chain(label="Series")
    _make_call(tmp_outputs, "call_A", [], 100.0)
    _make_call(tmp_outputs, "call_B", [], 200.0)

    chain_mod.append_call_to_chain(m["id"], "call_A")
    chain_mod.append_call_to_chain(m["id"], "call_B")

    fresh = chain_mod.get_chain(m["id"])
    assert [c["index"] for c in fresh["calls"]] == [1, 2]
    assert [c["dir"] for c in fresh["calls"]] == ["call_A", "call_B"]


def test_append_writes_chain_block_to_call_transcript(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    _make_call(tmp_outputs, "call_A", [], 100.0)
    chain_mod.append_call_to_chain(m["id"], "call_A")

    with open(os.path.join(tmp_outputs, "call_A", "transcript.json")) as f:
        data = json.load(f)
    assert data["chain"]["id"] == m["id"]
    assert data["chain"]["index"] == 1
    assert data["chain"]["slug"] == m["slug"]


def test_remove_call_from_chain_renumbers(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    for name in ("A", "B", "C"):
        _make_call(tmp_outputs, name, [], 100.0)
        chain_mod.append_call_to_chain(m["id"], name)

    chain_mod.remove_call_from_chain(m["id"], "B")

    fresh = chain_mod.get_chain(m["id"])
    assert [c["dir"] for c in fresh["calls"]] == ["A", "C"]
    assert [c["index"] for c in fresh["calls"]] == [1, 2]

    # A's transcript chain.index must stay 1, C's must flip from 3 to 2.
    with open(os.path.join(tmp_outputs, "A", "transcript.json")) as f:
        assert json.load(f)["chain"]["index"] == 1
    with open(os.path.join(tmp_outputs, "C", "transcript.json")) as f:
        assert json.load(f)["chain"]["index"] == 2


def test_delete_chain_removes_manifest_dir_but_preserves_member_calls(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    _make_call(tmp_outputs, "A", [], 100.0)
    chain_mod.append_call_to_chain(m["id"], "A")

    chain_mod.delete_chain(m["id"])
    assert not os.path.exists(os.path.join(tmp_outputs, "__chains__", m["id"]))

    # Member call dir still present
    assert os.path.exists(os.path.join(tmp_outputs, "A", "transcript.json"))
    # And its chain block must be cleared
    with open(os.path.join(tmp_outputs, "A", "transcript.json")) as f:
        assert "chain" not in json.load(f)


def test_close_chain_flips_flag(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    chain_mod.close_chain(m["id"])
    assert chain_mod.get_chain(m["id"])["closed"] is True


# ---- Combined transcript ----------------------------------------------------

def test_rebuild_chain_transcript_with_zero_calls(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    ct = chain_mod.rebuild_chain_transcript(m["id"])
    assert ct["chain"]["id"] == m["id"]
    assert ct["entries"] == []


def test_rebuild_chain_transcript_with_two_calls(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    segs_A = [{"id": 0, "t0": 0.0, "t1": 5.0, "speaker": "agent", "role": "agent", "text": "Hello"}]
    segs_B = [
        {"id": 0, "t0": 0.0, "t1": 3.0, "speaker": "agent", "role": "agent", "text": "Welcome back"},
        {"id": 1, "t0": 3.0, "t1": 8.0, "speaker": "customer", "role": "customer", "text": "Thanks"},
    ]
    _make_call(tmp_outputs, "A", segs_A, 100.0)
    _make_call(tmp_outputs, "B", segs_B, 200.0)
    chain_mod.append_call_to_chain(m["id"], "A")
    chain_mod.append_call_to_chain(m["id"], "B")

    ct = chain_mod.rebuild_chain_transcript(m["id"])
    types = [e["type"] for e in ct["entries"]]
    # Expected: boundary, seg, boundary, seg, seg
    assert types == ["call_boundary", "segment", "call_boundary", "segment", "segment"]

    # chain_t0 must cumulate: call A offset=0, call B offset=100.
    seg_entries = [e for e in ct["entries"] if e["type"] == "segment"]
    assert seg_entries[0]["chain_t0"] == pytest.approx(0.0)
    assert seg_entries[1]["chain_t0"] == pytest.approx(100.0)
    assert seg_entries[2]["chain_t0"] == pytest.approx(103.0)

    # Each segment carries call_index
    assert seg_entries[0]["call_index"] == 1
    assert seg_entries[2]["call_index"] == 2


def test_rebuild_chain_transcript_persists_to_disk(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    _make_call(tmp_outputs, "A", [{"id": 0, "t0": 0.0, "t1": 1.0, "speaker": "a", "role": "agent", "text": "x"}], 10.0)
    chain_mod.append_call_to_chain(m["id"], "A")
    chain_mod.rebuild_chain_transcript(m["id"])

    path = os.path.join(tmp_outputs, "__chains__", m["id"], "chain_transcript.json")
    assert os.path.exists(path)
    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk["chain"]["id"] == m["id"]


def test_append_triggers_transcript_rebuild(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    _make_call(tmp_outputs, "A", [{"id": 0, "t0": 0.0, "t1": 1.0, "speaker": "a", "role": "agent", "text": "x"}], 10.0)
    chain_mod.append_call_to_chain(m["id"], "A")

    path = os.path.join(tmp_outputs, "__chains__", m["id"], "chain_transcript.json")
    assert os.path.exists(path)  # rebuilt automatically on append


def test_rebuild_skips_missing_or_corrupt_member(tmp_outputs):
    m = chain_mod.create_chain(label="S")
    _make_call(tmp_outputs, "A", [{"id": 0, "t0": 0.0, "t1": 1.0, "speaker": "a", "role": "agent", "text": "x"}], 10.0)
    chain_mod.append_call_to_chain(m["id"], "A")

    # Manifest pointing at non-existent "B"
    manifest = chain_mod.get_chain(m["id"])
    manifest["calls"].append({"index": 2, "dir": "B_missing", "added_at": "now"})
    chain_mod._save_manifest(manifest)

    ct = chain_mod.rebuild_chain_transcript(m["id"])
    # Only A's boundary + segment, missing member silently skipped
    seg_entries = [e for e in ct["entries"] if e["type"] == "segment"]
    assert len(seg_entries) == 1
    assert seg_entries[0]["call_index"] == 1


# ---- Slugify ----------------------------------------------------------------

def test_slugify_cleans_special_chars():
    assert chain_mod._slugify("Gaurav Bhatotia — Enrollment!") == "gaurav-bhatotia-enrollment"
    assert chain_mod._slugify("  Multiple   Spaces  ") == "multiple-spaces"
    assert chain_mod._slugify("Unicode Café ☕") == "unicode-cafe"
    assert chain_mod._slugify("") == ""


# ---- Compound citation resolver --------------------------------------------

def test_compound_citation_resolves_to_correct_call(tmp_outputs):
    from pipeline.reconstruct import verify_and_inject_inline_citations
    segs_A = [{"id": 0, "t0": 0.0, "t1": 5.0, "speaker": "a", "role": "agent", "text": "Hello"}]
    segs_B = [{"id": 42, "t0": 100.0, "t1": 110.0, "speaker": "a", "role": "agent", "text": "Later moment"}]

    chain_context = {1: segs_A, 2: segs_B}
    out = verify_and_inject_inline_citations(
        "**Later**[S42/C2] was key.", [], chain_context=chain_context
    )
    import re
    t0s = [float(x) for x in re.findall(r"scrollToTime\(([\d.]+)", out)]
    assert t0s == [100.0]


def test_compound_citation_invalid_call_index_drops(tmp_outputs):
    from pipeline.reconstruct import verify_and_inject_inline_citations
    chain_context = {1: [{"id": 0, "t0": 0.0, "t1": 1.0}]}
    out = verify_and_inject_inline_citations(
        "**Later**[S0/C9] missing.", [], chain_context=chain_context
    )
    assert "scrollToTime" not in out
    assert "**Later**" in out
