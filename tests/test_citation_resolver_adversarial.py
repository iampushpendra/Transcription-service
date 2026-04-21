"""Adversarial / edge-case tests for the citation resolver.

These go beyond the base happy-path tests and probe weird LLM outputs,
malformed structures, and real-world regression patterns.
"""

import re

import pytest

from pipeline.reconstruct import verify_and_inject_inline_citations


@pytest.fixture
def segments():
    return [
        {"id": 0, "t0": 0.0,   "t1": 4.5,   "speaker": "agent",    "text": "Namaste FREED."},
        {"id": 1, "t0": 4.8,   "t1": 12.3,  "speaker": "customer", "text": "Mujhe DRP jaanna tha."},
        {"id": 2, "t0": 45.0,  "t1": 52.1,  "speaker": "agent",    "text": "CIBIL score affect hoga."},
        {"id": 3, "t0": 52.1,  "t1": 60.0,  "speaker": "customer", "text": "Mera CIBIL kharab hai."},
        {"id": 4, "t0": 165.0, "t1": 170.0, "speaker": "agent",    "text": "CIBIL recover 12 mahine."},
    ]


def _t0s(html: str) -> list[float]:
    return [float(m) for m in re.findall(r"scrollToTime\(([\d.]+)\)", html)]


# --- LLM output with weirdly formatted citations -----------------------------

def test_extra_whitespace_in_bold_citation(segments):
    # Real LLM outputs sometimes add a space: **CIBIL** [S2]
    # Our regex is strict about no-space; the S-ID is dropped as a bare token.
    # Keyword stays bold; the bare [S2] still renders an arrow.
    out = verify_and_inject_inline_citations("**CIBIL** [S2] matters.", segments)
    assert "[↗]" in out
    assert _t0s(out) == [45.0]


def test_citation_inside_sentence_with_punctuation(segments):
    out = verify_and_inject_inline_citations(
        "Agent explained **CIBIL**[S2], then **DRP**[S1].", segments
    )
    assert _t0s(out) == [45.0, 4.8]


def test_citation_at_end_of_line_with_period(segments):
    out = verify_and_inject_inline_citations("Customer agreed [S3].", segments)
    assert _t0s(out) == [52.1]
    assert out.endswith(".")  # punctuation preserved


def test_multiple_cites_same_id(segments):
    out = verify_and_inject_inline_citations(
        "See **CIBIL**[S2] and again **CIBIL**[S2].", segments
    )
    assert _t0s(out) == [45.0, 45.0]


def test_citation_with_hhmmss_format(segments):
    # Segment 4 is at 165s = 00:02:45 in HH:MM:SS format; keyword "CIBIL" matches.
    out = verify_and_inject_inline_citations("**CIBIL**[00:02:45] recovery.", segments)
    assert _t0s(out) == [165.0]


def test_nested_markdown_ignored(segments):
    # LLM sometimes produces ***triple stars*** which is bold+italic; our regex
    # shouldn't match that. Keyword surrounded by '*' not '**' stays plain.
    out = verify_and_inject_inline_citations("***CIBIL***[S2]", segments)
    # The outer ** matches as bold for "*CIBIL*" keyword, inner stars survive.
    # Valid resolve should still happen on inner pattern.
    assert _t0s(out) == [45.0]


def test_citation_with_numbers_in_keyword(segments):
    out = verify_and_inject_inline_citations("**SPA 45%**[S2] savings.", segments)
    assert _t0s(out) == [45.0]


def test_empty_string_returns_empty(segments):
    assert verify_and_inject_inline_citations("", segments) == ""


def test_plain_text_unchanged(segments):
    text = "This has no citations at all, just a plain summary."
    out = verify_and_inject_inline_citations(text, segments)
    assert out == text


# --- Malformed / adversarial input -------------------------------------------

def test_unclosed_bracket(segments):
    # `[S2` without closing bracket should not match any pattern
    out = verify_and_inject_inline_citations("Badly **formed**[S2 citation.", segments)
    assert "scrollToTime" not in out


def test_sid_followed_by_text_not_bracket(segments):
    # "[S2 minutes]" — not our pattern
    out = verify_and_inject_inline_citations("Wait [S2 minutes].", segments)
    assert "scrollToTime" not in out


def test_huge_sid_number(segments):
    out = verify_and_inject_inline_citations("**Topic**[S999999999] cited.", segments)
    assert "scrollToTime" not in out


def test_zero_id_resolves(segments):
    # S0 must resolve — off-by-one risk
    out = verify_and_inject_inline_citations("**Hello**[S0].", segments)
    assert _t0s(out) == [0.0]


def test_mixed_valid_and_invalid_ids(segments):
    text = "**A**[S0] **B**[S99] **C**[S2] **D**[S100]"
    out = verify_and_inject_inline_citations(text, segments)
    assert _t0s(out) == [0.0, 45.0]
    # Invalid ones should preserve bold
    assert "<strong>B</strong>" not in out  # strong conversion happens in frontend, not here
    assert "**B**" in out
    assert "**D**" in out


def test_resolver_does_not_crash_on_unicode(segments):
    unicode_segs = [
        {"id": 0, "t0": 0.0, "t1": 5.0, "speaker": "agent", "text": "नमस्ते FREED में।"},
        {"id": 1, "t0": 5.0, "t1": 10.0, "speaker": "customer", "text": "ग्राहक कहता है।"},
    ]
    out = verify_and_inject_inline_citations("**नमस्ते**[S0] welcome.", unicode_segs)
    assert _t0s(out) == [0.0]


def test_html_already_in_input_preserved(segments):
    # Legacy summaries may already contain HTML links from a previous run.
    # Re-running the resolver should be idempotent — don't mangle existing tags.
    prior = '<a href="#" onclick="scrollToTime(45.0)">CIBIL</a>'
    out = verify_and_inject_inline_citations(prior, segments)
    assert out == prior


def test_bold_without_sid_untouched(segments):
    out = verify_and_inject_inline_citations("Just **bold text** with no cite.", segments)
    assert out == "Just **bold text** with no cite."


# --- Pattern boundary edge cases ---------------------------------------------

def test_sid_in_middle_of_word_not_matched(segments):
    # "abc[S2]xyz" — look-ahead `(?<![\w/])` prevents matches where preceded by word char
    out = verify_and_inject_inline_citations("abc[S2]xyz", segments)
    # abc has no word boundary problem — this should still match the bare-id pattern.
    # But prefix "abc" has word chars, so negative lookbehind blocks it.
    # Result: no link, text unchanged.
    assert "scrollToTime" not in out


def test_sid_after_punctuation_matches(segments):
    # "(text)[S2]" — punctuation is not \w, so should match
    out = verify_and_inject_inline_citations("(fee)[S2] note.", segments)
    assert _t0s(out) == [45.0]


def test_sid_with_leading_space_matches(segments):
    out = verify_and_inject_inline_citations("Context [S2] here.", segments)
    assert _t0s(out) == [45.0]


def test_markdown_link_not_confused(segments):
    # Do not match [text](url) markdown links
    out = verify_and_inject_inline_citations("See [link text](https://example.com) and **CIBIL**[S2].", segments)
    assert _t0s(out) == [45.0]
    assert "link text" in out  # preserved


# --- Recursion depth and mixed structures ------------------------------------

def test_deeply_nested_structure(segments):
    data = {
        "a": {
            "b": {
                "c": [
                    {"d": "Deep **CIBIL**[S2] cite"},
                    {"e": ["List **DRP**[S1] item"]},
                ]
            }
        }
    }
    out = verify_and_inject_inline_citations(data, segments)
    assert _t0s(out["a"]["b"]["c"][0]["d"]) == [45.0]
    assert _t0s(out["a"]["b"]["c"][1]["e"][0]) == [4.8]


def test_list_of_strings(segments):
    data = ["**A**[S0]", "**B**[S1]", "**C**[S2]"]
    out = verify_and_inject_inline_citations(data, segments)
    assert _t0s(out[0]) == [0.0]
    assert _t0s(out[1]) == [4.8]
    assert _t0s(out[2]) == [45.0]


def test_dict_keys_not_transformed(segments):
    # Keys shouldn't get rewritten, only values
    data = {"[S2]": "**CIBIL**[S2]"}
    out = verify_and_inject_inline_citations(data, segments)
    assert "[S2]" in list(out.keys())
    assert _t0s(out["[S2]"]) == [45.0]


def test_boolean_and_none_values(segments):
    data = {"enabled": True, "disabled": False, "empty": None, "count": 0}
    out = verify_and_inject_inline_citations(data, segments)
    assert out == data


# --- Fallback window boundary tests -----------------------------------------

def test_fallback_mmss_exactly_at_window_boundary(segments):
    # Segment 2: t0=45.0, t1=52.1, contains "CIBIL".
    # Target 00:55 (55s). Distance from segment = max(0, 55-52.1) = 2.9s. Within ±10s → snap.
    out = verify_and_inject_inline_citations("**CIBIL**[00:55] later.", segments)
    assert _t0s(out) == [52.1]  # t1 anchor (55 is closer to 52.1 than 45.0)


def test_fallback_mmss_just_outside_window_dropped(segments):
    # Target 01:30 (90s). Nearest CIBIL seg: seg 3 ends at 60, dist = 30s > 10s. Drop.
    out = verify_and_inject_inline_citations("**CIBIL**[01:30] later.", segments)
    assert "scrollToTime" not in out


def test_fallback_keyword_case_insensitive(segments):
    # LLM might emit lowercase keyword; matching is case-insensitive.
    out = verify_and_inject_inline_citations("**cibil**[00:46] noted.", segments)
    assert _t0s(out) == [45.0]


def test_fallback_substring_keyword_matches(segments):
    # "score" appears inside "CIBIL score affect hoga" — substring match works.
    out = verify_and_inject_inline_citations("**score**[00:46] mentioned.", segments)
    assert _t0s(out) == [45.0]


# --- Consistency / round-trip -----------------------------------------------

def test_resolver_is_idempotent_on_plain_text(segments):
    text = "Plain summary with no special markers, **bold**, or citations."
    first = verify_and_inject_inline_citations(text, segments)
    second = verify_and_inject_inline_citations(first, segments)
    assert first == second


def test_count_of_links_matches_count_of_valid_sids(segments):
    text = "**A**[S0] **B**[S1] **C**[S2] **D**[S3] **E**[S4]"
    out = verify_and_inject_inline_citations(text, segments)
    assert len(_t0s(out)) == 5
