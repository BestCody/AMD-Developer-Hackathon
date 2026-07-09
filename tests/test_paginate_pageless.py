"""Tests for ``paginate_pageless`` (chunk.py)."""
from __future__ import annotations

from uir_pipeline.chunk import paginate_pageless
from uir_pipeline.utils import count_tokens


# ----------------------------------------------------------------------------
# Basic shape: empty input -> empty list
# ----------------------------------------------------------------------------

def test_paginate_pageless_empty():
    assert paginate_pageless("") == []
    assert paginate_pageless("   \n   ") == []


def test_paginate_pageless_whitespace_only():
    assert paginate_pageless("\n\n\n\n") == []


# ----------------------------------------------------------------------------
# Single short paragraph fits one page
# ----------------------------------------------------------------------------

def test_paginate_pageless_single_paragraph_one_page():
    text = "Hello world. This is a small paragraph."
    pages = paginate_pageless(text, max_tokens=2000)
    assert len(pages) == 1
    page_no, page_text = pages[0]
    assert page_no == 1
    assert page_text == text


# ----------------------------------------------------------------------------
# Multi-paragraph below threshold -> one page
# ----------------------------------------------------------------------------

def test_paginate_pageless_three_small_paragraphs_one_page():
    text = (
        "First paragraph with a few words here.\n\n"
        "Second paragraph also brief.\n\n"
        "Third paragraph closes the example."
    )
    pages = paginate_pageless(text, max_tokens=2000)
    assert len(pages) == 1
    assert "First paragraph" in pages[0][1]
    assert "Second paragraph" in pages[0][1]
    assert "Third paragraph" in pages[0][1]


# ----------------------------------------------------------------------------
# Multi-paragraph above threshold -> multiple pages, sequential numbers
# ----------------------------------------------------------------------------

def test_paginate_pageless_long_text_splits_into_sequential_pages():
    # Build a text that is well above 2000 tokens even with one paragraph
    # equal to ``max_tokens``.
    para = "Sentence alpha beta gamma. " * 800  # ~10k tokens
    text = (para + "\n\n") * 3  # 3 paragraphs joined by \n\n
    pages = paginate_pageless(text, max_tokens=2000)
    assert len(pages) >= 2
    # Sequential page numbers starting at 1.
    assert [p for p, _ in pages] == list(range(1, len(pages) + 1))
    # No empty pages.
    for _, page_text in pages:
        assert page_text.strip()


# ----------------------------------------------------------------------------
# Boundary: a single oversized paragraph always consumed (no infinite loop)
# ----------------------------------------------------------------------------

def test_paginate_pageless_single_oversized_paragraph_does_not_loop():
    # ``alpha beta gamma. `` is 19 chars; *500 keeps the input under
    # ~10 KB so the BGE tokenizer doesn't dominate the test runtime.
    # The trailing space is normalised by ``_split_into_paragraphs``
    # which strips whitespace, so the round-trip assert uses .strip().
    para = "alpha beta gamma. " * 500
    pages = paginate_pageless(para, max_tokens=500)
    assert len(pages) == 1  # absorbed whole because no prior content
    assert pages[0][1] == para.strip()


# ----------------------------------------------------------------------------
# Token-boundary: each page <= max_tokens (within BGE ceiling)
# ----------------------------------------------------------------------------

def test_paginate_pageless_respects_max_tokens_approximately():
    para = "alpha beta gamma. " * 800
    text = (para + "\n\n") * 4
    pages = paginate_pageless(text, max_tokens=2000)
    # Each page should have token count roughly under max_tokens + 1 para's worth.
    for _, page_text in pages:
        tokens = count_tokens(page_text)
        # allow ~1 paragraph of slack (we never split a single paragraph
        # even if it would exceed the cap).
        assert tokens <= 2000 + count_tokens(para)


# ----------------------------------------------------------------------------
# Page numbering starts at 1 (matches existing PDF contract)
# ----------------------------------------------------------------------------

def test_paginate_pageless_page_numbers_start_at_1():
    text = "alpha. " * 4500  # enough to force 2+ pages
    pages = paginate_pageless(text, max_tokens=2000)
    assert pages[0][0] == 1
