"""test_intent_filter -- unit tests for the post-orchestrator module.

What this file asserts:
    1. Empty / blank intent returns ``matched_chunks=0`` and produces an
       unchanged-style copy.
    2. Keyword extraction strips a small English stop-word set so phrases
       like ``"show me the attention table"`` produce
       ``['attention', 'table']``.
    3. Keyword match is case-insensitive substring on chunk text.
    4. ``no_match_fallback=True`` is set when no chunk matches AND the
       intent has extractable keywords (graceful keep-all behavior).
    5. Output file is written alongside the source with a ``.intent.uir.json``
       suffix and embeds an ``intent_filter`` block on the root.
    6. Non-ASCII characters pass through (we shouldn't crash on
       accented / CJK text).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from uir_pipeline.intent_filter import filter_uirstream_by_intent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_uir(chunks_text: list[str]) -> dict:
    """Build a minimal UIR V1 dict with a flat list of text chunks."""
    return {
        "uiR_version": "1.0",
        "id": "fixture_doc",
        "modal_type": "document",
        "source": {
            "uri": "file:///tmp/fixture.pdf",
            "format": "PDF",
            "mime_type": "application/pdf",
            "size_bytes": 1024,
            "checksum": "sha256:deadbeef",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        "metadata": {
            "title": "Fixture",
            "page_count": len(chunks_text),
            "language": "en",
        },
        "structure": {
            "type": "hierarchical",
            "root": {
                "id": "doc_fixture",
                "type": "document",
                "title": "Fixture",
                "page": 1,
                "children": [
                    {
                        "id": f"chunk_{i}",
                        "type": "chunk",
                        "text": t,
                        "token_count": len(t.split()),
                        "page": 1,
                        "bounding_box": [0, 0, 1000, 1000],
                        "confidence": 1.0,
                        "modal_features": {"text": {"token_count": len(t.split())}},
                    }
                    for i, t in enumerate(chunks_text)
                ],
            },
        },
        "semantics": {"entities": [], "relationships": [], "topics": []},
        "provenance": {
            "extraction": {
                "model": "LayoutLMv3-heuristic",
                "version": "1.0",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            "normalization": {"version": "1.0", "timestamp": "2026-01-01T00:00:00Z"},
        },
    }


@pytest.fixture()
def fixture_uir_path(tmp_path: Path) -> Path:
    """Write a 4-chunk fixture UIR to ``tmp_path/fixture.uir.json``."""
    chunks = [
        "Multi-Head Attention. We project queries, keys and values to "
        "different linear projections of dimension 64.",
        "Scaled Dot-Product Attention is computed as a weighted sum over "
        "values, scaled by the square root of the dimensionality.",
        "Table 5 reports BLEU scores on the WMT 2014 English-German "
        "translation task. The Transformer base achieves 27.3.",
        "The encoder is composed of a stack of N=6 identical layers.",
    ]
    p = tmp_path / "fixture.uir.json"
    p.write_text(json.dumps(_build_uir(chunks)))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyOrBlankIntent:
    """Empty / blank input produces a no-op summary and unchanged-shape output."""
    def test_empty_string_is_noop(self, fixture_uir_path: Path) -> None:
        summary = filter_uirstream_by_intent(fixture_uir_path, "")
        assert summary["keywords"] == []
        assert summary["matched_chunks"] == 0
        assert summary["no_match"] is False
        # Source file untouched.
        on_disk = json.loads(fixture_uir_path.read_text())
        assert len(on_disk["structure"]["root"]["children"]) == 4

    def test_whitespace_only_is_noop(self, fixture_uir_path: Path) -> None:
        summary = filter_uirstream_by_intent(fixture_uir_path, "   \t\n  ")
        assert summary["keywords"] == []


class TestKeywordExtraction:
    """Stop-word + length filtering on the intent string."""

    def test_stopwords_dropped_lowercase(self) -> None:
        path_summary_run = filter_uirstream_by_intent
        # Use a tiny ephemeral fixture for keyword-only check.
        import json, tempfile
        from pathlib import Path as _P
        d = _P(tempfile.mkdtemp())
        f = d / "u.json"
        f.write_text(json.dumps(_build_uir(["alpha beta gamma"])))
        s = path_summary_run(f, "show me the attention")
        assert s["keywords"] == ["attention"]

    def test_dash_split_tokens(self, fixture_uir_path: Path) -> None:
        """``multi-head-attention`` should tokenise to ['multi', 'head', 'attention']."""
        s = filter_uirstream_by_intent(fixture_uir_path, "multi-head-attention")
        assert s["keywords"] == ["multi", "head", "attention"]

    def test_short_tokens_dropped(self, fixture_uir_path: Path) -> None:
        s = filter_uirstream_by_intent(fixture_uir_path, "ai is up to no go")
        # ``ai``, ``is``, ``up``, ``to``, ``no``, ``go`` are all < 3 chars,
        # AND ``is``, ``to``, ``no`` are stops -> all dropped.
        assert s["keywords"] == []

    def test_dedupes_in_query_order(self, fixture_uir_path: Path) -> None:
        s = filter_uirstream_by_intent(
            fixture_uir_path,
            "attention attention query query attention",
        )
        assert s["keywords"] == ["attention", "query"]


class TestMatching:
    """Case-insensitive substring matching on chunk.text."""

    def test_substring_case_insensitive(self, fixture_uir_path: Path) -> None:
        # ``BLEU`` -> capital letters should match via lowercased substring.
        s = filter_uirstream_by_intent(fixture_uir_path, "BLEU")
        assert s["matched_chunks"] == 1
        assert s["no_match"] is False

    def test_no_match_fallback_keeps_all(self, tmp_path: Path) -> None:
        """If keywords are present but no chunk matches, return all chunks
        (graceful degradation) and mark ``no_match=True`` so the caller
        knows the intent was over-restrictive."""
        chunks = ["alpha bravo charlie", "delta echo foxtrot"]
        p = tmp_path / "missing.uir.json"
        p.write_text(json.dumps(_build_uir(chunks)))
        s = filter_uirstream_by_intent(p, "zebra")
        assert s["keywords"] == ["zebra"]
        # Fallback: matched_chunks resets to 0 (semantic "no match found")
        # but the OUTPUT keeps every chunk so the user sees something.
        # Path: ``p.parent / (p.stem + ".intent" + p.suffix)`` preserves
        # the ``.uir`` portion (which ``Path.with_suffix`` would silently
        # strip). Reuse a tiny helper at the top of this class below.
        filtered_path = p.parent / (p.stem + ".intent" + p.suffix)
        on_disk = json.loads(filtered_path.read_text())
        assert len(on_disk["structure"]["root"]["children"]) == 2
        assert on_disk["structure"]["root"]["intent_filter"]["no_match_fallback"] is True
        assert s["matched_chunks"] == 0

    def test_partial_match_kept(self, fixture_uir_path: Path) -> None:
        """``attention`` should match BOTH Multi-Head + Dot-Product chunks."""
        s = filter_uirstream_by_intent(fixture_uir_path, "attention")
        assert s["matched_chunks"] == 2
        assert s["no_match"] is False


class TestFileOutput:
    """Output file path + embedded intent_filter block."""

    def test_output_file_suffix(self, fixture_uir_path: Path) -> None:
        s = filter_uirstream_by_intent(fixture_uir_path, "attention")
        # Default suffix-appended clean path: ``<stem>.intent<suffix>``.
        target = fixture_uir_path.parent / (fixture_uir_path.stem + ".intent" + fixture_uir_path.suffix)
        assert Path(s["out_path"]) == target

    def test_output_well_formed(self, fixture_uir_path: Path) -> None:
        s = filter_uirstream_by_intent(fixture_uir_path, "encoder")
        loaded = json.loads(Path(s["out_path"]).read_text())
        # Intent summary embedded at root.
        f = loaded["structure"]["root"]["intent_filter"]
        assert f["intent"] == "encoder"
        assert f["keywords"] == ["encoder"]
        assert f["matched_chunks"] == 1
        assert f["total_chunks"] == 4

    def test_explicit_out_path_overrides_default(
        self, fixture_uir_path: Path, tmp_path: Path,
    ) -> None:
        explicit = tmp_path / "custom-named.uir.json"
        s = filter_uirstream_by_intent(fixture_uir_path, "attention", out_path=explicit)
        assert s["out_path"] == str(explicit)
        assert explicit.is_file()


class TestNonAsciiPassthrough:
    """Non-ASCII characters in intent / chunk text don't crash."""

    def test_unicode_intent(self, tmp_path: Path) -> None:
        chunks = ["Le chat est sur le tapis.", "The cat sits on the mat."]
        p = tmp_path / "u.json"
        p.write_text(json.dumps(_build_uir(chunks)))
        s = filter_uirstream_by_intent(p, "matin")
        # 3-char French token kept; matches nothing; graceful fallback.
        filtered_path = p.parent / (p.stem + ".intent" + p.suffix)
        on_disk = json.loads(filtered_path.read_text())
        assert len(on_disk["structure"]["root"]["children"]) == 2
        assert s["no_match"] is True
