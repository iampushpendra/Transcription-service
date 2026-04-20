"""Tests for transcript citation resolver (segment-ID + MM:SS fallback)."""

import re

import pytest

from pipeline.reconstruct import verify_and_inject_inline_citations


@pytest.fixture
def segments():
    return [
        {"id": 0, "t0": 0.0,  "t1": 4.5,  "speaker": "agent",    "text": "Namaste, FREED mein aapka swagat hai."},
        {"id": 1, "t0": 4.8,  "t1": 12.3, "speaker": "customer", "text": "Mujhe DRP ke baare mein jaanna tha."},
        {"id": 2, "t0": 45.0, "t1": 52.1, "speaker": "agent",    "text": "CIBIL score affect hoga short-term mein."},
        {"id": 3, "t0": 52.1, "t1": 60.0, "speaker": "customer", "text": "Mera CIBIL already kharab hai, aur girega toh?"},
        {"id": 4, "t0": 165.0, "t1": 170.0, "speaker": "agent",  "text": "CIBIL recover ho jaayega 12-18 mahine mein."},
    ]


def _onclick_t0(html: str) -> list[float]:
    """Extract t0 values from scrollToTime(<t0>) calls in rendered HTML."""
    return [float(m) for m in re.findall(r"scrollToTime\(([\d.]+)\)", html)]


# ---- Primary: segment-ID resolver -------------------------------------------

def test_bold_keyword_with_valid_id_wraps_keyword(segments):
    out = verify_and_inject_inline_citations("**CIBIL**[S2] is mentioned.", segments)
    assert "<a " in out
    assert _onclick_t0(out) == [45.0]
    assert ">CIBIL</a>" in out
    assert "[S2]" not in out  # ID token consumed


def test_bare_id_renders_arrow(segments):
    out = verify_and_inject_inline_citations("Customer agreed to proceed [S3].", segments)
    assert _onclick_t0(out) == [52.1]
    assert "[↗]" in out
    assert "[S3]" not in out


def test_invalid_id_dropped_keyword_preserved(segments):
    out = verify_and_inject_inline_citations("**CIBIL**[S999] was cited.", segments)
    assert "scrollToTime" not in out
    assert "CIBIL" in out  # keyword text preserved
    assert "[S999]" not in out  # invalid cite stripped


def test_bare_invalid_id_dropped(segments):
    out = verify_and_inject_inline_citations("Something happened [S42].", segments)
    assert "scrollToTime" not in out
    assert "[S42]" not in out


def test_malformed_ids_do_not_crash(segments):
    # None of these should produce links
    for bad in ["[S]", "[Sfoo]", "[S-1]", "[S 1]", "[ S1 ]"]:
        out = verify_and_inject_inline_citations(f"Bad citation {bad} here.", segments)
        assert "scrollToTime" not in out


def test_multiple_valid_ids_all_resolve(segments):
    text = "**FREED**[S0] explained; customer asked **DRP**[S1]; later **CIBIL**[S2]."
    out = verify_and_inject_inline_citations(text, segments)
    assert _onclick_t0(out) == [0.0, 4.8, 45.0]


def test_recursion_into_dict_and_list(segments):
    data = {
        "overview": "Customer asked about **DRP**[S1].",
        "key_topics": ["**FREED**[S0]", "**CIBIL**[S2]"],
        "nested": {"evidence": "See [S3]."},
    }
    out = verify_and_inject_inline_citations(data, segments)
    assert _onclick_t0(out["overview"]) == [4.8]
    assert _onclick_t0(out["key_topics"][0]) == [0.0]
    assert _onclick_t0(out["key_topics"][1]) == [45.0]
    assert _onclick_t0(out["nested"]["evidence"]) == [52.1]


def test_non_string_values_passthrough(segments):
    data = {"count": 42, "enabled": True, "score": 0.8, "tags": None}
    out = verify_and_inject_inline_citations(data, segments)
    assert out == data


# ---- Fallback: hardened MM:SS resolver --------------------------------------

def test_fallback_mmss_with_keyword_in_window_snaps(segments):
    # target "00:46" (46s) → segment 2 @ 45.0 contains "CIBIL", within ±10s
    out = verify_and_inject_inline_citations("**CIBIL**[00:46] affected.", segments)
    assert _onclick_t0(out) == [45.0]


def test_fallback_mmss_without_keyword_nearby_is_dropped(segments):
    # target "01:30" (90s). No segment within ±10s contains "DRP"
    out = verify_and_inject_inline_citations("**DRP**[01:30] mentioned.", segments)
    assert "scrollToTime" not in out
    assert "DRP" in out


def test_fallback_mmss_standalone_inside_segment_resolves(segments):
    # Standalone [MM:SS] with target inside seg 4 (165.0 - 170.0) → exact match.
    out = verify_and_inject_inline_citations("See [02:47] for context.", segments)
    assert _onclick_t0(out) == [165.0]


def test_fallback_mmss_standalone_outside_any_segment_passes_through(segments):
    # Standalone [MM:SS] with target not inside any segment → left intact for frontend.
    out = verify_and_inject_inline_citations("Check [99:59] later.", segments)
    assert "scrollToTime" not in out
    assert "[99:59]" in out  # token preserved for frontend handling


def test_fallback_picks_nearest_of_repeated_keyword_within_window(segments):
    # "CIBIL" appears at 45.0, 52.1, 165.0. Target 02:45 (165s) → should snap to 165.0, not 45.0.
    out = verify_and_inject_inline_citations("**CIBIL**[02:45] recovery.", segments)
    assert _onclick_t0(out) == [165.0]


def test_primary_preferred_over_fallback(segments):
    # Input has both [S<id>] and [MM:SS]; only the S-id should produce a link
    out = verify_and_inject_inline_citations("**CIBIL**[S2] at [03:00].", segments)
    t0s = _onclick_t0(out)
    assert 45.0 in t0s
    # The [03:00] standalone has no keyword window → dropped
    assert len(t0s) == 1


# ---- Anti-regression: old bug where t1-closest returned t0 ------------------

def test_fallback_returns_correct_anchor_on_t1_closer(segments):
    # Build segments where t1 is the closer anchor. seg 2: t0=45.0 t1=52.1, keyword "CIBIL"
    # Target 00:52 (52s). |52-45|=7, |52-52.1|=0.1 → t1 is closer; anchor should be 52.1.
    out = verify_and_inject_inline_citations("**CIBIL**[00:52] reached.", segments)
    assert _onclick_t0(out) == [52.1]
