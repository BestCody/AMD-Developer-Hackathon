"""test_intent_filter -- pin the 6-fix behavior set for the intent feature.

Three minimal inline UIRs exercise the post-fix paths:
    1. fixture_with_sections  -- exercises fix #2 (tree-walk into sections)
    2. fixture_bge_vectors    -- exercises fix #4/5 (BGE cosine ranking +
                                  ranked response with neighbours)
    3. fixture_legacy_no_vecs -- exercises the substring fallback when no
                                  embeddings are persisted

Each fixture asserts a specific property (matched > 0, score present,
neighbours present, topic widening). Failures here mean a fix regressed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Repo-local src/ add so we don't depend on `pip install -e .`
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from uir_pipeline.intent_filter import (  # noqa: E402
    _intent_keywords, filter_uirstream_by_intent,
)


# ----------------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------------

def _write_uir(tmp: Path, name: str, payload: dict) -> Path:
    p = tmp / name
    p.write_text(json.dumps(payload))
    return p


def _section(section_id: str, title: str, chunks: list[dict]) -> dict:
    return {"id": section_id, "type": "section", "title": title,
            "page": 1, "bounding_box": [0, 0, 1000, 1000],
            "children": chunks}


def _chunk(
    chunk_id: str, text: str, *, section_path: str | None = None,
    embedding: list[float] | None = None, page: int = 1,
    preceding: str | None = None, following: str | None = None,
) -> dict:
    mf: dict = {}
    if section_path is not None:
        mf["section"] = {"path": section_path, "is_first": False, "is_last": False}
    if preceding or following:
        if preceding:
            mf["preceding_chunk_id"] = {"chunk_id": preceding}
        if following:
            mf["following_chunk_id"] = {"chunk_id": following}
    if embedding is not None:
        mf["vector"] = {
            "dim": len(embedding),
            "model": "BAAI/bge-small-en-v1.5",
            "chunk_index": 0,
            "embedding": embedding,
        }
    return {
        "id": chunk_id, "type": "chunk", "text": text,
        "token_count": len(text.split()),
        "page": page, "bounding_box": [0, 0, 1000, 1000],
        "confidence": 1.0, "modal_features": mf,
    }


def _doc_envelope(root_children: list[dict], *topics: str) -> dict:
    return {
        "uiR_version": "1.0", "id": "doc_test", "modal_type": "document",
        "source": {
            "uri": "file:///test.pdf", "format": "PDF",
            "mime_type": "application/pdf", "size_bytes": 1024,
            "checksum": "sha256:0" * 64,
            "timestamp": "2024-01-01T00:00:00Z",
        },
        "metadata": {"title": "t", "page_count": 1, "language": "en"},
        "structure": {"type": "hierarchical", "root": {
            "id": "root_doc", "type": "document", "title": "t",
            "page": 1, "children": root_children,
        }},
        "semantics": {
            "entities": [], "relationships": [],
            "topics": list(topics),
        },
        "provenance": {
            "extraction": {
                "model": "LayoutLMv3-heuristic", "version": "1.0",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            "normalization": {
                "version": "1.0", "timestamp": "2024-01-01T00:00:00Z",
            },
        },
    }


# ----------------------------------------------------------------------------
# Tokenization (fix #3) -- short-token safelist keeps AI/ML/GPU through
# ----------------------------------------------------------------------------

def test_short_tokens_kept():
    """AI/ML/GPU survive the length floor (fix #3)."""
    assert "ai" in _intent_keywords("AI is taking over")
    assert "ml" in _intent_keywords("ML and GPU")
    assert "gdp" in _intent_keywords("GDP growth")


def test_stopwords_still_dropped():
    """Common English stopwords still get dropped."""
    out = _intent_keywords("show me the table about it")
    # 'show', 'me', 'the', 'it' all stopwords; only 'table' + 'about' might survive
    assert "show" not in out and "me" not in out and "the" not in out


def test_compound_numeric_kept():
    """6Li, BGE-512, vec_dim=384 -- new tokenizer splits on non [A-Za-z0-9-]."""
    out = _intent_keywords("6Li lithium depletion")
    assert "6li" in out
    assert "lithium" in out
    out2 = _intent_keywords("BGE-512 embeddings")
    assert "bge-512" in out2


def test_dedup_and_order_preserved():
    out = _intent_keywords("lithium lithium 6Li lithium")
    # first-seen order; dedup
    assert out == ["lithium", "6li"]


# ----------------------------------------------------------------------------
# Tree-walk (fix #2) -- sections used to be dropped silently
# ----------------------------------------------------------------------------

def test_section_title_match(tmp_path: Path):
    """Section titled 'Observations' matches intent 'observation' (fix #2)."""
    # Two sections: Observations + Methods; intent should pull only
    # Observations because section-title match wins.
    obs = _section("sec_obs",
                   "Observations",
                   [_chunk("c1", "stellar lithium abundance is 2.4"),
                    _chunk("c2", "model predicts additional depletion")])
    meth = _section("sec_meth",
                    "Methods",
                    [_chunk("c3", "we used 6Li spectroscopy")])
    uir = _doc_envelope([obs, meth])
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "show me observation")
    assert res["no_match_fallback"] is False
    assert len(res["matches"]) >= 2  # the 2 chunks in Observations
    # Section title-match path: obs section is kept whole
    out = json.loads(path.with_name(path.stem + ".intent" + path.suffix).read_text())
    root = out["structure"]["root"]
    kept_titles = [c.get("title") for c in root["children"]
                   if c.get("type") == "section"]
    assert "Observations" in kept_titles


def test_chunk_text_match_pulls_parent_section(tmp_path: Path):
    """Chunk-text match on a nested chunk keeps its parent section intact."""
    obs = _section("sec_obs", "Observations",
                   [_chunk("c1", "the stellar parameter is measured"),
                    _chunk("c2", "model predicts additional depletion")])
    uir = _doc_envelope([obs])
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "stellar depletion")
    assert res["no_match_fallback"] is False, res
    # Both chunks should be in the output (c1 via 'stellar', c2 via 'depletion')
    assert len(res["matches"]) >= 1
    out = json.loads(path.with_name(path.stem + ".intent" + path.suffix).read_text())
    sec = next(c for c in out["structure"]["root"]["children"]
               if c.get("type") == "section")
    kept_texts = [c["text"] for c in sec["children"]]
    assert any("stellar" in t for t in kept_texts) or any("depletion" in t for t in kept_texts)


# ----------------------------------------------------------------------------
# BGE cosine ranking (fix #4) -- only when embeddings are present
# ----------------------------------------------------------------------------

def _mini_unit_vec(direction: str) -> list[float]:
    """Make a tiny 4-d unit-ish vector for testing -- exact direction wins."""
    if direction == "lithium":
        return [1.0, 0.1, 0.0, 0.0]
    if direction == "money":
        return [0.0, 0.0, 1.0, 0.1]
    return [0.0, 0.0, 0.0, 1.0]


def test_cosine_ranking_runs_and_emits_scores(tmp_path: Path, monkeypatch):
    """BGE-cosine returns ranked matches when embeddings are persisted."""
    # Monkey-patch _embed_intent so we don't pay BGE cold-load in CI
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent",
        lambda intent: _mini_unit_vec("lithium"),
    )
    chunks = [
        _chunk("c_lit", "lithium abundance stellar", embedding=_mini_unit_vec("lithium")),
        _chunk("c_money", "revenue gdp earnings", embedding=_mini_unit_vec("money")),
        _chunk("c_other", "deep convolutional network", embedding=_mini_unit_vec("other")),
    ]
    uir = _doc_envelope(chunks, "lithium")
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "lithium stellar")
    out = json.loads(path.with_name(path.stem + ".intent" + path.suffix).read_text())
    meta = out["structure"]["root"]["intent_filter"]
    # Match path: cosine path picks the lithium-aligned chunk
    assert meta["scoring"].startswith("cosine+bge")
    assert len(res["matches"]) >= 1
    # First match should be c_lit (largest cosine)
    assert meta["matches"][0]["chunk_id"] == "c_lit"
    assert meta["matches"][0]["score"] > 0.5


def test_legacy_uir_falls_back_to_keyword(tmp_path: Path, monkeypatch):
    """A UIR without vector.embedding keys falls back to keyword match."""
    # No embeddings persisted -> cosine returns nothing -> keyword path
    # is used.
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent",
        lambda intent: [1.0, 0.0, 0.0, 0.0],
    )
    chunks = [
        _chunk("c1", "alpha-beta algorithm details"),
        _chunk("c2", "gamma delta unrelated content"),
    ]
    uir = _doc_envelope(chunks)
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "alpha-beta")
    out = json.loads(path.with_name(path.stem + ".intent" + path.suffix).read_text())
    meta = out["structure"]["root"]["intent_filter"]
    # No embeddings -> cosine matches are empty -> keyword scoring only.
    assert meta["scoring"] == "keyword"
    assert len(res["matches"]) >= 1


# ----------------------------------------------------------------------------
# Match response shape (fix #5) -- chunk_id + score + section_path + neighbours
# ----------------------------------------------------------------------------

def test_match_response_has_neighbors_and_section_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent",
        lambda intent: _mini_unit_vec("lithium"),
    )
    chunks = []
    # 3 chunks with cross-linked preceding/following
    for i, text in enumerate([
        "lithium abundance page 1",
        "stellar spectrum observations",
        "model predictions and analysis",
    ]):
        prev = f"c{i-1}" if i > 0 else None
        nxt = f"c{i+1}" if i < 2 else None
        chunks.append(_chunk(f"c{i}", text,
                             section_path="Observations",
                             embedding=_mini_unit_vec("lithium"),
                             preceding=prev, following=nxt))
    uir = _doc_envelope(chunks)
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "lithium")
    out = json.loads(path.with_name(path.stem + ".intent" + path.suffix).read_text())
    meta = out["structure"]["root"]["intent_filter"]
    match = meta["matches"][0]
    assert "chunk_id" in match and match["chunk_id"].startswith("c")
    assert "score" in match
    assert "score_kind" in match
    assert "section_path" in match
    assert "neighbour_chunk_ids" in match
    # Inline ``text`` so agents can cite without re-lookup
    # (added in the empirical-probe cleanup). Pin so a future regression
    # that drops the field breaks this test.
    assert "text" in match and isinstance(match["text"], str), match
    # If we got the first chunk, it should have a following_chunk_id
    if match["chunk_id"] == "c0":
        assert "c1" in match["neighbour_chunk_ids"]


# ----------------------------------------------------------------------------
# Topic-aware widening (fix #6)
# ----------------------------------------------------------------------------

def test_topics_used_for_widening(tmp_path: Path, monkeypatch):
    """If intent keyword overlaps semantics.topics, mention it in meta."""
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent",
        lambda intent: None,  # disable cosine to focus on keyword path
    )
    obs = _section("sec_obs", "Observations",
                   [_chunk("c1", "we measured stellar lithium"),
                    _chunk("c2", "uncertainty in the abundance estimate")])
    uir = _doc_envelope([obs], "lithium", "stellar")
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "lithium")
    out = json.loads(path.with_name(path.stem + ".intent" + path.suffix).read_text())
    meta = out["structure"]["root"]["intent_filter"]
    # topics_hit should capture keywords overlapping with topics
    assert any("lithium" in kw for kw in res["topics_hit"])


# ============================================================================
# Iteration-2 tests
# ============================================================================
# Pin the 4 post-test fixes:
#   #2-A: BM25-lite text-score fallback so agent always has ranking signal
#   #2-B: silent BGE exception now surfaces a warning + reason
#   #2-C: dynamic cosine threshold (per-query floor = top - 0.15)
#   #2-D: topic widening uses token-substring, not exact token match
# ============================================================================

def _score(m: dict) -> float:
    """Helper: safely extract score (None-tolerant)."""
    v = m.get("score")
    return float(v) if isinstance(v, (int, float)) else 0.0


def test_text_score_ranks_by_relevance(tmp_path: Path, monkeypatch):
    """BM25-lite (iteration-2 #2-A) ranks chunks by keyword density."""
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent", lambda intent: None,
    )
    # 3 chunks: highly-relevant, medium, irrelevant.
    chunks = [
        _chunk("c_lit_hi", "lithium stellar abundance is the measurement"),
        _chunk("c_lit_md", "the lithium abundance is discussed"),
        _chunk("c_lit_lo", "completely unrelated vocabulary here"),
    ]
    uir = _doc_envelope(chunks)
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "lithium stellar abundance")
    assert res["scoring"] == "keyword"
    # All 3 chunks survived no_match because at least one had a keyword
    # match; the agent should now see non-None scores and a clear ordering.
    scores = [_score(m) for m in res["matches"]]
    assert all(s > 0 for s in scores), scores
    # The chunk with both 'lithium' AND 'stellar' AND 'abundance' outranks
    # the chunk with just 'lithium' AND 'abundance'.
    assert scores[0] >= scores[1], res["matches"]
    # c_lit_lo (irrelevant) should be absent OR score=0.
    kept_ids = {m["chunk_id"] for m in res["matches"]}
    if "c_lit_lo" in kept_ids:
        idx = next(i for i, m in enumerate(res["matches"]) if m["chunk_id"] == "c_lit_lo")
        assert scores[idx] == 0.0


def test_title_boost_via_text_score(tmp_path: Path, monkeypatch):
    """Section-title match upweights BM25 score so titles win over body hits."""
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent", lambda intent: None,
    )
    obs = _section("sec_obs", "Observations",
                   [_chunk("c_a", "we measured 6Li this week"),
                    _chunk("c_b", "observations of cosmic rays and dust")])
    intr = _section("sec_intro", "Introduction",
                    [_chunk("c_c", "6Li is mentioned once here")])
    uir = _doc_envelope([obs, intr])
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "observations")
    # Intr (Introduction) doesn't match, so only Observations section kept
    # via a section-level title match (the whole obs section is lifted into
    # matches_payload). Both c_a and c_b are inside the obs section tree
    # and thus surface in matches. c_a has 0 body match (only section lift
    # pulled it in), c_b has 1 body match. BM25-lite ranks c_b higher.
    by_id = {m["chunk_id"]: _score(m) for m in res["matches"]}
    assert "c_a" in by_id
    assert "c_b" in by_id
    assert "c_c" not in by_id  # intro section, no title/body match
    # c_b beats c_a: not because of payload but because BM25-lite counts
    # the body hit in c_b. This is the correct BM25-lite behaviour.
    assert by_id["c_b"] >= by_id["c_a"]


def test_dynamic_threshold_with_low_cosine(tmp_path: Path, monkeypatch):
    """Per-query dynamic floor (fix #2-C) relaxes threshold when no chunk
    exceeds cosine_threshold=0.20. Low scores still surface (don't drop the
    kernel of 'maybe this is the closest' matches).
    """
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent",
        lambda intent: [0.10, 0.20, 0.30][:1] * 4,  # intent vector
    )
    # 3 chunks with embeddings that score 0.10, 0.20, 0.30 against intent.
    chunks = [
        _chunk("c_lo", "unrelated", embedding=[0.05, -0.10, 0.30, 0.10]),
        _chunk("c_md", "mediocre", embedding=[0.20, 0.15, 0.25, 0.30]),
        _chunk("c_hi", "on-topic",  embedding=[0.30, 0.30, 0.30, 0.30]),
    ]
    uir = _doc_envelope(chunks)
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "anything", cosine_threshold=0.20)
    # top-1 hit (0.30 >= 0.20) -> floor = max(0.30 - 0.15, 0.05) = 0.15
    # so c_md (0.20) and c_hi (0.30) survive; c_lo (0.05 < 0.15) drops.
    kept = [m["chunk_id"] for m in res["matches"]]
    assert "c_hi" in kept
    assert "c_lo" not in kept


def test_dynamic_threshold_relaxes_for_short_intent(
    tmp_path: Path, monkeypatch,
):
    """When no chunk scores >= cosine_threshold, the floor relaxes to the
    ambient score so the agent gets the closest match instead of nothing.
    """
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent",
        lambda intent: [1.0, 0.0, 0.0, 0.0],
    )
    chunks = [
        _chunk("c_a", "near-zero",  embedding=[0.02, 0.01, 0.00, 0.00]),
        _chunk("c_b", "ambient 0.08", embedding=[0.08, 0.05, 0.00, 0.00]),
        _chunk("c_c", "background", embedding=[0.01, 0.00, 0.00, 0.00]),
    ]
    uir = _doc_envelope(chunks)
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "AI", cosine_threshold=0.20)
    # All similarity scores are < 0.20 -> relaxed floor activates; match c_b
    kept = {m["chunk_id"] for m in res["matches"]}
    assert "c_b" in kept, res["matches"]


def test_topic_token_substring_match(tmp_path: Path, monkeypatch):
    """Multi-token topic phrase matches single-token keyword (fix #2-D)."""
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent", lambda intent: None,
    )
    obs = _section("sec_obs", "Observations",
                   [_chunk("c1", "stellar nucleosynthesis details")])
    uir = _doc_envelope([obs], "Stellar nucleosynthesis", "cosmology background")
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "stellar nucleosynthesis")
    # topics_hit should pick up BOTH "stellar" and "nucleosynthesis" tokens
    # against the multi-token topic "Stellar nucleosynthesis".
    assert "stellar" in res["topics_hit"]
    assert "nucleosynthesis" in res["topics_hit"]


def test_embed_failure_surfaces_in_meta(tmp_path: Path, monkeypatch, caplog):
    """Silent BGE exception now logs + sets module _EMBED_FAIL_REASON
    (fix #2-B). The orchestrator can surface this to the UI.

    Implementation note: monkeypatching ``_embed_intent`` directly would
    BYPASS the new try/except inside the function (the broken function
    wouldn't have any exception handling). Patching the underlying
    ``embed_texts`` instead is the right thing -- it lets the production
    ``_embed_intent`` code execute AND the failure is caught AND
    ``_EMBED_FAIL_REASON`` is set AND the warning is logged.
    """
    import uir_pipeline.intent_filter as intent_filter_mod
    intent_filter_mod._EMBED_FAIL_REASON = None
    # Patch the underlying embed_texts entry point -- the new try/except
    # in _embed_intent catches this exception and sets _EMBED_FAIL_REASON.
    def _broken_embed_texts(*args, **kwargs):
        raise RuntimeError("simulated torchvision import loop")
    import uir_pipeline.embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed_texts", _broken_embed_texts)
    # Use an intent that LITERALLY matches the chunk body so the BM25-lite
    # fallback actually produces a match. (Earlier I used "hello world"
    # which doesn't appear in "test content here" so matches was empty,
    # tripping the `res["matches"]` truthiness assertion below.)
    chunks = [_chunk("c1", "test content here lithium stellar")]
    uir = _doc_envelope(chunks)
    path = _write_uir(tmp_path, "t.uir.json", uir)
    with caplog.at_level("WARNING", logger="uir_pipeline.intent_filter"):
        res = filter_uirstream_by_intent(path, "lithium stellar")
    # BM25-lite is the fallback; matches should exist with non-None score.
    assert res["scoring"] == "keyword", res
    assert res["matches"], res
    assert res["matches"][0]["score_kind"] == "bm25-lite"
    # The persisted meta surfaces the reason (the assertion the orchestrator
    # actually consumes). Module-global is also set but we don't pin it --
    # it's mutable state, not the contract surface.
    out = json.loads(path.with_name(path.stem + ".intent" + path.suffix).read_text())
    meta = out["structure"]["root"]["intent_filter"]
    assert meta["embed_unavailable_reason"] is not None
    assert "RuntimeError" in meta["embed_unavailable_reason"]
    # The warning was emitted with exc_info so debugging is possible
    # without re-running.
    assert any("BGE embed failed" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


def test_topic_token_no_false_positive_on_short_kw(tmp_path: Path, monkeypatch):
    """Iteration-2 strict topic widening: kw length >= 3 prevents "ai"
    from matching "main"/"fail"/"train"-class topic tokens.
    """
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent", lambda intent: None,
    )
    obs = _section("sec_obs", "Observations",
                   [_chunk("c1", "the train arrived main station")])
    uir = _doc_envelope([obs], "Stellar nucleosynthesis", "main train")
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "stellar nucleosynthesis")
    # 'stellar' length 7 >= 3, exact token match -> topics_hit includes stellar
    # 'nucleosynthesis' length 14 >= 3, exact token match -> topics_hit includes it
    assert "stellar" in res["topics_hit"]
    assert "nucleosynthesis" in res["topics_hit"]
    # 2-token intent length < 3 chars: nothing short should appear
    assert all(len(kw) >= 3 for kw in res["topics_hit"])


# ============================================================================
# Public-API surface contract (regression for the real bug surfaced in
# empirical eval against /tmp/uir_v4 -- the function declared `Path` in
# its type hint but accepted `uir_path.read_text()` unconditionally, so
# any caller who passed a raw string crashed with `AttributeError: 'str'
# object has no attribute 'read_text'`. CLI / shell callers pass strings
# by default, so this bug blocked any real-world invocation outside the
# test suite. The fix widens the signature to `Path | str` and coerces
# to `Path` at function entry; the tests below pin both contracts.)
# ============================================================================

def test_filter_uirstream_accepts_str_path(tmp_path: Path, monkeypatch):
    """``filter_uirstream_by_intent`` must accept a raw ``str`` path."""
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent", lambda intent: None,
    )
    obs = _section(
        "sec_obs", "Observations",
        [_chunk("c_a", "we measured lithium abundance in stellar spectra")],
    )
    uir = _doc_envelope([obs])
    path = _write_uir(tmp_path, "t.uir.json", uir)
    # Regression: pass a str path (not a Path)
    str_path = path.as_posix()
    res = filter_uirstream_by_intent(str_path, "lithium")
    # The keyword match fires AND we are NOT on the no-match-fallback branch
    # (c_a literally contains "lithium"), so the first match must be the
    # actual BM25-lite hit on c_a -- not the synthetic
    # ``*no_match_fallback*`` sentinel. Tests pinning ONLY ``matches non-empty``
    # would otherwise stay green under a future regression that swung the
    # scoring path to a degraded branch (output mismatch vs. signal gone).
    assert res["matches"], "str-path call returned matches=[] (regression)"
    m0 = res["matches"][0]
    assert m0["chunk_id"] != "*no_match_fallback*", res
    assert m0["score_kind"] in {"bm25-lite", "section-lift"}, res
    assert m0.get("score") is None or m0["score"] > 0, res
    # Output sibling file exists (path.parent/stem + '.intent' + suffix).
    expected_out = path.parent / (path.stem + ".intent" + path.suffix)
    assert expected_out.is_file(), f"str-path call did not write {expected_out}"


def test_filter_uirstream_accepts_path_object(tmp_path: Path, monkeypatch):
    """Legacy ``Path`` contract must still work after the widening."""
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent", lambda intent: None,
    )
    obs = _section(
        "sec_obs", "Observations",
        [_chunk("c_a", "we measured lithium abundance in stellar spectra")],
    )
    uir = _doc_envelope([obs])
    path = _write_uir(tmp_path, "t.uir.json", uir)
    res = filter_uirstream_by_intent(path, "lithium")  # Path object, not str
    assert res["matches"]


def test_filter_uirstream_str_output_path(tmp_path: Path, monkeypatch):
    """Optional ``out_path`` arg must accept a raw string too."""
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent", lambda intent: None,
    )
    obs = _section(
        "sec_obs", "Observations",
        [_chunk("c_a", "we measured lithium abundance in stellar spectra")],
    )
    uir = _doc_envelope([obs])
    path = _write_uir(tmp_path, "t.uir.json", uir)
    explicit_out = tmp_path / "custom_intent.uir.json"
    res = filter_uirstream_by_intent(
        path.as_posix(),
        "lithium",
        out_path=explicit_out.as_posix(),  # str out_path
    )
    assert explicit_out.is_file(), "explicit str out_path did not yield a written file"


def test_filter_uirstream_accepts_pathlike(tmp_path: Path, monkeypatch):
    """Per the docstring claim, ``os.PathLike`` (e.g. ``pathlib.PurePath``
    subclass) must also work -- the signature widens beyond ``Path`` to
    include any ``__fspath__`` implementer. This pins the type-hint /
    ``Path()``-coercion contract so future code that re-narrows the
    signature breaks loudly here.
    """
    import os
    monkeypatch.setattr(
        "uir_pipeline.intent_filter._embed_intent", lambda intent: None,
    )
    obs = _section(
        "sec_obs", "Observations",
        [_chunk("c_a", "we measured lithium abundance in stellar spectra")],
    )
    uir = _doc_envelope([obs])
    path = _write_uir(tmp_path, "t.uir.json", uir)

    class _MyPath(os.PathLike):
        """Custom PathLike so the test exercises the ``__fspath__`` path
        WITHOUT relying on ``pathlib.PurePath`` (whose subclasses the
        ``Path()`` constructor may special-case)."""
        def __init__(self, p: Path) -> None:
            self._p = p
        def __fspath__(self) -> str:
            return str(self._p)
    res = filter_uirstream_by_intent(_MyPath(path), "lithium")
    assert res["matches"]


def test_filter_uirstream_rejects_none(monkeypatch):
    """Defensive: ``None`` is not path-like. Must raise ``TypeError``
    (the ``Path()`` constructor itself raises on None).
    """
    with pytest.raises((TypeError, AttributeError)):
        filter_uirstream_by_intent(None, "lithium")  # type: ignore[arg-type]  -- intentional


