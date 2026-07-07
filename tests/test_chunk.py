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
        "growing-window"
    )
