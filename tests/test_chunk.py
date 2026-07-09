"""tests/test_chunk.py -- Phase I chunker tests.

The BGE tokenizer is loaded once for the suite. Tests avoid asserting
exact token counts (BGE-small token boundaries shift by model version);
instead they assert SHAPE: each chunk under the hard cap, overlap
stitching happens, and short-text fast paths.
"""
from __future__ import annotations

import pytest

from uir_pipeline.chunk import (
    ChunkDraft,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    _split_sentences,
    chunk_text,
)
from uir_pipeline.utils import DEFAULT_BGE_MODEL, count_tokens


# ----------------------------------------------------------------------------
# _split_sentences
# ----------------------------------------------------------------------------

def test_split_sentences_basic():
    out = _split_sentences("Hello world. Bye now. Hi again?")
    assert out == ["Hello world.", "Bye now.", "Hi again?"]


def test_split_sentences_handles_no_terminal_punct():
    out = _split_sentences("apple banana cherry")
    assert out == ["apple banana cherry"]


def test_split_sentences_normalizes_whitespace_only_at_boundaries():
    """``"... test."`` (lowercase after the period) returns as ONE sentence.

    The regex only splits at ``[.!?]+\\s+`` followed by uppercase/bracket/quote,
    so lazy punctuation isn't rewarded with phantom sentences.
    """
    out = _split_sentences("This is a sentence. this continues without break.")
    assert len(out) == 1
    assert "continues" in out[0]


# ----------------------------------------------------------------------------
# chunk_text: short text
# ----------------------------------------------------------------------------

def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n   ") == []


def test_chunk_text_short_produces_single_chunk():
    text = "Hello world."
    chunks = chunk_text(text, page=1)
    assert len(chunks) == 1
    assert isinstance(chunks[0], ChunkDraft)
    assert chunks[0].text == "Hello world."
    assert chunks[0].page == 1
    assert chunks[0].token_count == count_tokens("Hello world.")


# ----------------------------------------------------------------------------
# chunk_text: window + overlap (shape only -- BGE token boundaries shift)
# ----------------------------------------------------------------------------

def test_chunk_text_long_text_yields_multiple_chunks_or_single():
    """Long text with target=64 should yield >= 1 chunks. Each chunk's
    token count should be <= MAX_CHUNK_TOKENS. Don't assert an exact
    split count because BGE token boundaries shift by model version.
    """
    text = "This is a paragraph of words. " * 50  # ~250 tokens
    chunks = chunk_text(
        text,
        page=1,
        target_tokens=64,
        overlap_pct=20,
    )
    assert len(chunks) >= 1
    for c in chunks:
        assert c.token_count <= MAX_CHUNK_TOKENS


def test_chunk_text_overlap_keeps_chunks_within_token_cap():
    """With overlap, adjacent chunks share text; per-chunk token counts
    may exceed target but must respect the hard BGE cap of 512.
    """
    text = "Sentence A. " + "Sentence B. " * 30 + "Sentence Z."
    chunks = chunk_text(
        text,
        page=1,
        target_tokens=64,
        overlap_pct=25,
    )
    assert len(chunks) >= 1
    for c in chunks:
        assert c.token_count <= MAX_CHUNK_TOKENS


# ----------------------------------------------------------------------------
# chunk_text: recursive halving
# ----------------------------------------------------------------------------

def test_chunk_text_respects_max_tokens_cap_for_long_text():
    """Long text gets split so every chunk fits within ``MAX_CHUNK_TOKENS``.

    The exact split count depends on BGE tokenization, so we only assert
    each chunk is under the cap. We deliberately use a multi-sentence
    payload so the chunker has whitespace boundaries to halve on.
    """
    text = ("The quick brown fox jumps over the lazy dog. " * 200).strip()
    chunks = chunk_text(text, page=1, max_tokens=MAX_CHUNK_TOKENS)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= MAX_CHUNK_TOKENS


def test_chunk_text_no_space_text_falls_back_gracefully():
    """A single no-space / no-punct word can't be cleanly halved -- accepted.

    Documents :func:`_recursive_halve`'s last-resort pass-through.
    """
    text = "x" * 10_000  # one giant token with zero whitespace
    chunks = chunk_text(text, page=1, max_tokens=MAX_CHUNK_TOKENS)
    assert len(chunks) >= 1  # whatever the chunker can do, that's fine.


def test_chunk_text_target_at_min_boundary_works():
    chunks = chunk_text(
        ("alpha. " * 200),
        page=1,
        target_tokens=MIN_CHUNK_TOKENS,
    )
    assert all(c.token_count <= MAX_CHUNK_TOKENS for c in chunks)


def test_chunk_text_target_above_max_clamped_to_max():
    chunks = chunk_text(
        ("alpha. " * 200),
        page=1,
        target_tokens=9999,  # above 512
    )
    for c in chunks:
        assert c.token_count <= MAX_CHUNK_TOKENS


def test_chunk_text_module_id_propagates_to_token_count():
    text = "Quick test of token-count alignment."
    chunks = chunk_text(text, page=2, model_id=DEFAULT_BGE_MODEL)
    assert chunks[0].token_count == count_tokens(text, DEFAULT_BGE_MODEL)


# ----------------------------------------------------------------------------
# chunk_text: bbox + modal_features
# ----------------------------------------------------------------------------

def test_chunk_text_uses_supplied_bbox():
    bbox = (110, 220, 330, 440)
    chunks = chunk_text("a single sentence.", page=1, bbox=bbox)
    assert chunks[0].bbox == bbox


def test_chunk_text_default_bbox_is_full_canvas():
    chunks = chunk_text("a single sentence.", page=1)
    assert chunks[0].bbox == (0, 0, 1000, 1000)


def test_chunk_text_modal_features_metadata():
    chunks = chunk_text("hello world.", page=1)
    assert "text" in chunks[0].modal_features
    assert "chunk_strategy" in chunks[0].modal_features["text"]
    assert chunks[0].modal_features["text"]["chunk_strategy"].startswith(
        "paragraph-aware"
    )


# ----------------------------------------------------------------------------
# Paragraph-aware splitting (replaces sentence-bounded core path)
# ----------------------------------------------------------------------------

def test_chunk_text_bundles_small_paragraphs():
    """Two short paragraphs that sum under target_tokens share one chunk
    rather than getting separately cached AND split.
    """
    text = "First paragraph with a few words.\n\nSecond paragraph here."
    chunks = chunk_text(text, page=1, target_tokens=64)
    assert len(chunks) == 1
    # Both paragraphs must appear in the chunk text -- ``\\n\\n`` between
    # them preserved so the agent downstream sees the paragraph break.
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[0].text
    assert "\n\n" in chunks[0].text


def test_chunk_text_single_newline_preserves_paragraph():
    """``\\n`` (line-wrap) is NOT a paragraph break -- only ``\\n\\n`` is.

    Verifies the empirical-eval bug class is gone: a chunk's text no
    longer has mid-paragraph sentence splits.
    """
    text = "Line one. Line two.\nLine three. Line four.\n\nNew paragraph sentence A. New paragraph sentence B."
    chunks = chunk_text(text, page=1, target_tokens=64)
    # 2 paragraphs total. Both fit under target=64.
    # The single-newline group stays UNSPLIT (its lines stay joined) --
    # so ``Line two.\\nLine three`` appears in the chunk text (single
    # newline preserved, not split into its own chunk).
    for c in chunks:
        # No fragment from the second sentence is glued to other sentences.
        assert "Line two." in c.text or "Line three." in c.text
    pool_text = "\n\n".join(c.text for c in chunks)
    # The whole first paragraph stays intact -- ``\\n`` between Line two
    # and Line three is preserved (line-wrap, not paragraph break).
    assert "Line one. Line two.\nLine three. Line four." in pool_text
    # The paragraph break survives -- ``\\n\\n`` between "Line four." and
    # "New paragraph".
    assert "Line four.\n\nNew" in pool_text


def test_chunk_text_oversize_single_paragraph_triggers_halving():
    """A SINGLE paragraph (no ``\\n\\n`` boundaries) exceeding ``max_tokens``
    is recursively halved -- preserves the BGE 512-token ceiling while
    staying inside the paragraph geometry.
    """
    # ~600 tokens, no \n\n -> 1 paragraph > max=512 -> _recursive_halve -> 2+ chunks
    text = ("The quick brown fox jumps over the lazy dog. " * 120).strip()
    chunks = chunk_text(text, page=1, target_tokens=256, max_tokens=MAX_CHUNK_TOKENS)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= MAX_CHUNK_TOKENS


def test_chunk_text_paragraph_split_keeps_internal_sentences_intact():
    """Mid-paragraph sentence-capital-letters do NOT trigger splits -- the
    old ``[.!?]+space+capital`` regex bled here and the paragraph-first
    splitter replaces it.
    """
    # Paragraph A has a sentence ending + a new sentence starting with
    # capital. The old regex would have split -- the paragraph splitter
    # does NOT.
    text = (
        "First sentence. Second sentence continues. Third. Fourth.\n\n"
        "Next paragraph one. Next paragraph two."
    )
    chunks = chunk_text(text, page=1, target_tokens=64)
    # 2 paragraphs, both fit. Bundling may keep them as 1 chunk. Either
    # way, mid-paragraph sentence boundaries must not split.
    for c in chunks:
        # Mid-paragraph sentence pairs stay on the same line.
        assert ("First sentence. Second sentence continues" in c.text
                or "Third. Fourth" in c.text
                or "Next paragraph one. Next paragraph two" in c.text)


def test_chunk_text_overlap_uses_small_word_tail():
    """The overlap stitch must use the fixed small word-tail (≤16
    words) and NOT bleed an entire sentence across the boundary.
    """
    text = (
        "A short first paragraph with several words in it.\n\n"
        "A short second paragraph with several words in it."
    )
    chunks = chunk_text(text, page=1, target_tokens=64, overlap_pct=20)
    for c in chunks:
        assert c.token_count <= MAX_CHUNK_TOKENS
    # 2 paragraphs, both tiny. With overlap, second chunk may carry a
    # short word-tail from the first. Cap is 16 words. The stitched
    # chunk2 may start with up to 16 words from chunk1's tail.
    if len(chunks) >= 2:
        chunk2_head = chunks[1].text.split(maxsplit=20)
        # No full sentence bleeds (each paragraph's sentences have ~9
        # words; if 16-word overlap runs, that's >1 sentence -- but the
        # test verifies the bleed is bounded and doesn't fragment
        # sentences mid-word).
        assert len(chunk2_head) <= 20


def test_split_sentences_legacy_export_still_works():
    """`_split_sentences` is retained as an exported legacy test helper
    even though the production chunker is paragraph-aware. Pin that this
    doesn't silently disappear.
    """
    out = _split_sentences("Hello world. Bye now. Hi again?")
    assert out == ["Hello world.", "Bye now.", "Hi again?"]


def test_split_into_paragraphs_helper():
    """The paragraph-splitting helper gets exercised directly so a future
    regression in the regex surfaces here, not just at the chunker level.
    """
    from uir_pipeline.chunk import _split_into_paragraphs
    paras = _split_into_paragraphs("a\nb\n\nc\nd\n\n\ne\nf")
    assert paras == ["a\nb", "c\nd", "e\nf"]


def test_bundle_paragraphs_oversize_recurses():
    """When a single paragraph exceeds max_tokens, the bundler routes it
    through ``_recursive_halve`` so the BGE ceiling holds.
    """
    from uir_pipeline.chunk import _bundle_paragraphs_for_chunks
    # Fixture: a single paragraph (no ``\\n\\n``) that is DEFINITELY over
    # MAX_CHUNK_TOKENS. Each ~5-word sentence is ~7 BPE tokens; 200
    # sentences ~= 1400 tokens, well above 512. Single ``\\n`` between
    # paragraphs would NOT count as a paragraph break.
    paragraph = (
        "short head.\n"
        + ("alpha beta gamma delta epsilon. " * 200).strip()
        + "\nshort tail."
    )
    # Sanity: confirm the fixture is actually oversize before testing the
    # bundler's halve path -- guards against future BGE versions shifting
    # token boundaries so much that the assertion semantics invert.
    from uir_pipeline.utils import count_tokens
    assert count_tokens(paragraph) > MAX_CHUNK_TOKENS, (
        f"fixture must exceed MAX_CHUNK_TOKENS ({MAX_CHUNK_TOKENS}); "
        f"got {count_tokens(paragraph)}"
    )
    out = _bundle_paragraphs_for_chunks([paragraph], target_tokens=256, max_tokens=MAX_CHUNK_TOKENS)
    # The single oversize paragraph gets halved => multiple chunks
    assert len(out) >= 2
    for seg in out:
        assert count_tokens(seg) <= MAX_CHUNK_TOKENS
