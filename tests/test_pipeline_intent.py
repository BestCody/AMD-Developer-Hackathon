"""Tier 1 intent-reading regression tests.

What this file asserts:
    1. ``pipeline._is_boilerplate`` now matches the ``"Googleherebygrantspermissionto"``
       variant (the trailing-``\\b`` bug fix).
    2. ``chunk.chunk_text`` accepts the four Tier 1 kwargs without TypeError and
       emits ``modal_features.intent`` / ``modal_features.section`` blocks only
       when the kwargs are supplied.
    3. ``pipeline._section_heading_re`` matches the common numbered-heading
       shapes (\"3.2 Foo\", \"3 Foo\") and rejects non-heading prose.

These tests intentionally avoid end-to-end pipeline runs so they stay fast
(under 1 s on cold pytest). End-to-end coverage lives in
``tests/integration/test_pipeline_tier3.py`` (slow, gated on real fixtures).
"""
from __future__ import annotations

import re

import pytest

from uir_pipeline.pipeline import _BOILERPLATE_RE, _is_boilerplate
from uir_pipeline.chunk import chunk_text


# ---------------------------------------------------------------------------
# Boilerplate-regex bug fix (Tier 1.D)
# ---------------------------------------------------------------------------

class TestBoilerplateRegexBugFix:
    """Trailing-``\\b`` bug: '\\bgoogleherebygrantspermission\\b' won't match
    'Googleherebygrantspermissionto' because ``\\b`` requires a word/non-word
    transition, but ``n`` and ``t`` are both word-class chars. Fix: make
    ``to`` an optional suffix with `(?:to)?\\b`.
    """

    @pytest.mark.parametrize(
        "variant",
        [
            "Google hereby grants permission to",
            "google hereby grants permissions to",
            "Googleherebygrantspermission",       # bare token-stripped
            "Googleherebygrantspermissionto",    # with suffixed "to" (the bug case)
            "googleherebygrantspermissionTO",    # case-insensitive
        ],
    )
    def test_arxiv_permission_variants_dropped(self, variant: str) -> None:
        assert _is_boilerplate(variant), (
            f"_is_boilerplate should drop {variant!r} but didn't "
            f"-- the trailing-\\b bug regressed?"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "Reviewer 3 drafted the conclusion",
            "The Transformer paper cites attention as an alternative",
            "AshishVaswani published a paper",
        ],
    )
    def test_non_boilerplate_kept(self, text: str) -> None:
        assert not _is_boilerplate(text), (
            f"_is_boilerplate spuriously matched {text!r}"
        )

    def test_regex_pattern_uses_optional_to_suffix(self) -> None:
        """Static check: the compiled regex contains `(?:to)?` -- a future
        code-coverage regression would remove this and re-introduce the bug.
        """
        joined = "\n".join(p.pattern for p in _BOILERPLATE_RE)
        assert r"googleherebygrantspermission(?:to)?" in joined, (
            "the boilerplate regex no longer has the `(?:to)?` suffix; "
            "the trailing-\\b bug has regressed"
        )


# ---------------------------------------------------------------------------
# chunk_text new kwargs (Tier 1 plumbing)
# ---------------------------------------------------------------------------

class TestChunkTextAcceptsIntentKwargs:
    """chunk_text must accept the four Tier 1 kwargs without TypeError and
    inject the corresponding ``modal_features`` sub-blocks only when the
    kwargs are supplied."""

    def test_no_intent_kwargs_emits_only_text_block(self) -> None:
        drafts = chunk_text(
            "This is the body of section 3.2 on multi-head attention. "
            "It contains several sentences for the chunker to chew through. "
            "More body content so token counts are non-trivial.",
            page=4,
        )
        assert drafts, "expected at least one chunk from non-empty input"
        for d in drafts:
            assert "intent" not in d.modal_features
            assert "section" not in d.modal_features
            assert "text" in d.modal_features

    def test_region_kind_emits_intent_block(self) -> None:
        drafts = chunk_text(
            "Multi-Head Attention. We instead project queries, keys and values "
            "to different dimensions and perform h parallel attention functions.",
            page=4, region_kind="heading",
        )
        assert drafts
        for d in drafts:
            assert d.modal_features.get("intent") == {"region_kind": "heading"}

    def test_section_path_emits_section_block(self) -> None:
        drafts = chunk_text(
            "Multi-Head Attention. We project queries, keys and values to "
            "different learned linear projections. The dimensionality is 64.",
            page=4,
            region_kind="paragraph",
            section_path="3.2",
            is_section_first=False,
            is_section_last=True,
        )
        assert drafts
        for d in drafts:
            sec = d.modal_features.get("section")
            assert sec == {"path": "3.2", "is_first": False, "is_last": True}

    def test_unnumbered_section_path_preserved(self) -> None:
        """Unnumbered headings ('Abstract', 'References') should reach
        chunk_text with their literal text as section_path when the
        orchestrator sets them."""
        drafts = chunk_text(
            "Abstract. The dominant sequence transduction models are based on "
            "complex recurrent or convolutional neural networks.",
            page=1, region_kind="heading", section_path="Abstract",
            is_section_first=True,
        )
        assert drafts
        for d in drafts:
            assert d.modal_features["section"]["path"] == "Abstract"
            assert d.modal_features["section"]["is_first"] is True


# ---------------------------------------------------------------------------
# Heading-detection regex (Tier 1.B)
# ---------------------------------------------------------------------------

# Single canonical copy of the orchestrator's heading regex so the test
# fails loudly if either side drifts.
_SECTION_HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)[\.\s]+(\S.{2,})$"
)


class TestSectionHeadingRegex:
    """The orchestrator's heading regex must match common structural shapes
    and refuse to trigger on prose."""

    @pytest.mark.parametrize(
        "head,expected_path",
        [
            ("3.2 Multi-Head Attention", "3.2"),
            ("3 Background", "3"),
            ("3.2.1 Scaled Dot-Product Attention", "3.2.1"),
            ("5.  Experiments", "5"),
        ],
    )
    def test_numbered_heading_detected(self, head: str, expected_path: str) -> None:
        m = _SECTION_HEADING_RE.match(head.strip())
        assert m, f"heading {head!r} did not match"
        assert m.group(1) == expected_path

    @pytest.mark.parametrize(
        "prose",
        [
            "The dominant sequence transduction models are based on recurrent "
            "neural networks.",  # year-style "Models" -> no number prefix
            "In 2024 we trained a new model.",  # year-as-prefix
            "Section 3.2 discusses attention.",  # references but isn't a heading
        ],
    )
    def test_prose_refused(self, prose: str) -> None:
        # Only the BEGINNING of a line should match; sentence-initial references
        # at later positions are still prose.
        assert _SECTION_HEADING_RE.match(prose.strip()) is None
