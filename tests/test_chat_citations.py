"""Citation markers in a grounded answer must refer to a passage we supplied.

The system prompt says "never invent a citation number that wasn't given to
you". A prompt is a request, not a constraint. A model that writes "[4]" when
three passages were supplied produces a claim that *looks* sourced and cannot
be checked -- strictly worse than an uncited claim, because the bracket buys
unearned credibility.

`_validate_citations` strips those markers and reports them.
"""
from __future__ import annotations

import pytest

from uir_pipeline.chat import (
    MIN_COSINE_SCORE,
    _cited_indices,
    _validate_citations,
    answer,
)


# ---------------------------------------------------------------------------
# _validate_citations
# ---------------------------------------------------------------------------

def test_valid_markers_are_left_alone():
    text = "The model uses 8 heads [1] and 6 layers [2]."
    cleaned, invalid = _validate_citations(text, 3)
    assert cleaned == text
    assert invalid == []


def test_out_of_range_marker_is_stripped_and_reported():
    cleaned, invalid = _validate_citations("Revenue grew 12% [4].", 3)
    assert "[4]" not in cleaned
    assert invalid == [4]


def test_stripping_does_not_leave_a_space_before_punctuation():
    cleaned, _ = _validate_citations("Revenue grew 12% [4].", 3)
    assert cleaned == "Revenue grew 12%."


def test_zero_is_out_of_range():
    """Passages are 1-based; [0] refers to nothing."""
    cleaned, invalid = _validate_citations("Claim [0].", 3)
    assert invalid == [0]
    assert "[0]" not in cleaned


def test_mixed_valid_and_invalid_keeps_only_the_valid():
    cleaned, invalid = _validate_citations("A [1] and B [9] and C [2].", 2)
    assert invalid == [9]
    assert "[1]" in cleaned and "[2]" in cleaned and "[9]" not in cleaned


def test_multiple_invalid_markers_are_all_reported_sorted():
    _, invalid = _validate_citations("A [7] B [4] C [7].", 3)
    assert invalid == [4, 7]


def test_no_contexts_means_every_marker_is_invalid():
    cleaned, invalid = _validate_citations("Grounded claim [1].", 0)
    assert invalid == [1]
    assert "[1]" not in cleaned


def test_non_numeric_brackets_are_untouched():
    """Markdown links and [a] are not citation markers; don't mangle prose."""
    text = "See [a] and [see the docs](http://x)."
    cleaned, invalid = _validate_citations(text, 2)
    assert cleaned == text
    assert invalid == []


def test_a_code_subscript_is_not_a_citation():
    """`array[0]` is an index. Deleting it would corrupt quoted code."""
    text = "Use array[0] and weights[7] to index."
    cleaned, invalid = _validate_citations(text, 2)
    assert cleaned == text, "subscripts must survive citation stripping"
    assert invalid == []
    assert _cited_indices(text, 2) == []


def test_two_digit_markers_are_handled():
    cleaned, invalid = _validate_citations("Claim [12].", 12)
    assert cleaned == "Claim [12]." and invalid == []
    _, invalid = _validate_citations("Claim [12].", 11)
    assert invalid == [12]


# ---------------------------------------------------------------------------
# _cited_indices
# ---------------------------------------------------------------------------

def test_cited_indices_are_deduped_and_in_order_of_appearance():
    assert _cited_indices("B [2], then A [1], then B again [2].", 3) == [2, 1]


def test_cited_indices_ignores_out_of_range():
    assert _cited_indices("A [1] and [9].", 2) == [1]


def test_cited_indices_empty_when_uncited():
    assert _cited_indices("An answer with no markers.", 3) == []


# ---------------------------------------------------------------------------
# answer() wiring
# ---------------------------------------------------------------------------

def _contexts(n):
    return [
        {"doc_id": "d", "doc_title": "T", "chunk_id": f"c{i}",
         "page": 1, "text": f"passage {i}", "score": 0.9}
        for i in range(1, n + 1)
    ]


class _FakeResponse:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}], "usage": {}}


@pytest.fixture
def _fireworks(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    def install(content):
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(content))

    return install


def test_answer_strips_a_hallucinated_citation(_fireworks):
    _fireworks("Heads: 8 [1]. Revenue: $4M [4].")
    out = answer("q", _contexts(3))
    assert out["success"] is True
    assert "[4]" not in out["answer"]
    assert out["invalid_citations"] == [4]
    assert out["cited"] == [1]


def test_answer_reports_no_invalid_citations_on_a_clean_reply(_fireworks):
    _fireworks("Heads: 8 [1] and layers: 6 [2].")
    out = answer("q", _contexts(3))
    assert out["invalid_citations"] == []
    assert out["cited"] == [1, 2]
    assert out["answer"] == "Heads: 8 [1] and layers: 6 [2]."


def test_empty_contexts_short_circuit_carries_the_new_keys():
    """The no-retrieval path must not omit keys the route reads."""
    out = answer("q", [])
    assert out["grounded"] is False
    assert out["cited"] == [] and out["invalid_citations"] == []
    assert out["model"] is None  # the model was never called


# ---------------------------------------------------------------------------
# the retrieval floor
# ---------------------------------------------------------------------------

def test_cosine_floor_sits_inside_the_measured_gap():
    """Swept on a 267-chunk UIR with bge-small-en-v1.5.

    Out-of-domain top-1 cosine peaked at 0.570; the lowest chunk that actually
    contained an answer scored 0.683 (10 of 10 questions). The floor must fall
    strictly between, or it either admits off-topic queries or discards
    answers. The original 0.62 sat above the then-worst answer-bearing chunk
    (0.614) and silently dropped a correct passage.
    """
    assert 0.570 < MIN_COSINE_SCORE < 0.683
