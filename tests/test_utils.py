"""tests/test_utils.py -- shared helper ground-truth tests.

Critical regression coverage:
    -- ``deterministic_node_id`` is idempotent across runs (same inputs -> same id).
    -- Stripped-prefix round-trip for ``strip_uir_prefix``.
    -- ``bbox_from_pixel`` clamps + canonical-orders coordinates.
    -- ``bbox_union`` encloses both inputs tightly.
    -- ``get_bge_tokenizer`` returns a stable instance across calls
       (and lazy-loads ``transformers`` if needed).
"""
from __future__ import annotations

import pytest

from uir_pipeline.utils import (
    BBOX_CANVAS,
    DEFAULT_BGE_MODEL,
    bbox_from_pixel,
    bbox_union,
    count_tokens,
    deterministic_node_id,
    strip_uir_prefix,
)


# ----------------------------------------------------------------------------
# deterministic_node_id
# ----------------------------------------------------------------------------

def test_node_id_is_stable_across_calls():
    id1 = deterministic_node_id("chunk", "src.pdf", 1, 2, "abc")
    id2 = deterministic_node_id("chunk", "src.pdf", 1, 2, "abc")
    assert id1 == id2


def test_node_id_prefix_separation_different_seeds():
    a = deterministic_node_id("chunk", "src.pdf", 1, 1, "x")
    b = deterministic_node_id("chunk", "src.pdf", 1, 2, "x")  # diff page==2
    assert a != b


def test_node_id_unknown_prefix_raises():
    with pytest.raises(ValueError, match="unknown UIR id prefix"):
        deterministic_node_id("definitely-not-a-prefix", "x")


def test_node_id_matches_schema_pattern():
    """The produced id must satisfy ``uir_schema.NODE_ID_PATTERN``."""
    import re
    from uir_pipeline.uir_schema import NODE_ID_PATTERN
    id_str = deterministic_node_id("entity", "Alice", 0.9)
    assert re.match(NODE_ID_PATTERN, id_str), f"{id_str!r} failed pattern"


# ----------------------------------------------------------------------------
# strip_uir_prefix
# ----------------------------------------------------------------------------

def test_strip_uir_prefix_basic():
    full = deterministic_node_id("chunk", "a", "b")
    assert strip_uir_prefix(full) == full.split("_", 1)[1]


def test_strip_uir_prefix_no_prefix_is_no_op():
    """A bare UUID without a prefix is left alone (defensive)."""
    bare = "00000000-0000-0000-0000-000000000000"
    assert strip_uir_prefix(bare) == bare


# ----------------------------------------------------------------------------
# bbox_from_pixel
# ----------------------------------------------------------------------------

def test_bbox_from_pixel_canonical_ordering():
    """Coordinates given bottom-up must be swapped to canonical (x1<=x2,y1<=y2)."""
    bbox = bbox_from_pixel((500, 500, 100, 100), page_width_px=1000, page_height_px=1000)
    assert bbox == (100, 100, 500, 500)


def test_bbox_from_pixel_corners_to_canvas():
    """Pixel (0,0,1000,1000) maps to canvas (0,0,1000,1000)."""
    bbox = bbox_from_pixel((0, 0, 1000, 1000), page_width_px=1000, page_height_px=1000)
    assert bbox == (0, 0, 1000, 1000)


def test_bbox_from_pixel_clamps_oversize():
    """Coordinates larger than page_dim are clamped, not silently truncated."""
    bbox = bbox_from_pixel((1500, 1500, 100, 100), page_width_px=1000, page_height_px=1000)
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 <= BBOX_CANVAS
    assert 0 <= y1 <= BBOX_CANVAS
    assert 0 <= x2 <= BBOX_CANVAS
    assert 0 <= y2 <= BBOX_CANVAS
    assert x2 == 1000   # clamped


def test_bbox_from_pixel_clamps_negative():
    bbox = bbox_from_pixel((-100, -100, 100, 100), page_width_px=1000, page_height_px=1000)
    assert bbox == (0, 0, 100, 100)


def test_bbox_from_pixel_rejects_zero_dim():
    with pytest.raises(ValueError, match="page dimensions"):
        bbox_from_pixel((0, 0, 100, 100), page_width_px=0, page_height_px=1000)


# ----------------------------------------------------------------------------
# bbox_union
# ----------------------------------------------------------------------------

def test_bbox_union_encloses_both():
    a = (100, 100, 200, 200)
    b = (300, 50, 400, 250)
    assert bbox_union(a, b) == (100, 50, 400, 250)


def test_bbox_union_overlapping():
    a = (100, 100, 200, 200)
    b = (150, 150, 180, 180)
    assert bbox_union(a, b) == a


# ----------------------------------------------------------------------------
# BGE tokenizer
# ----------------------------------------------------------------------------

def test_get_bge_tokenizer_lazy_loads_and_caches(monkeypatch):
    """Two calls with the same model_id return the same tokenizer instance."""
    t1 = count_tokens.__module__  # silence linter for unused reference
    _ = t1
    # First call: triggers transformers import + tokenizer load (~133MB on cold,
    # but Python's @lru_cache of the module-level dict is what we're testing).
    from uir_pipeline.utils import get_bge_tokenizer
    tok1 = get_bge_tokenizer(DEFAULT_BGE_MODEL)
    tok2 = get_bge_tokenizer(DEFAULT_BGE_MODEL)
    assert tok1 is tok2  # same instance


def test_count_tokens_returns_positive_int():
    n = count_tokens("Hello, world!")
    assert isinstance(n, int) and n > 0


def test_count_tokens_long_text_counts_many():
    n_short = count_tokens("one two three")
    n_long = count_tokens("one two three " * 50)
    assert n_long > n_short
