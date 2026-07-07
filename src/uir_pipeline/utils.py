"""utils -- shared helpers (deterministic UIR IDs, bbox normalization, BGE token counting).

PLAN.md \u00a78 mandates that UIR node IDs are deterministic strings of the form
``<prefix>_<uuid>`` where the UUID is ``uuid5(NAMESPACE_URL, <stable key>)``.
Native ``pydantic.UUID5`` rejects the prefix; the schema is regex-validated
instead (see ``uir_schema.NODE_ID_PATTERN``). This module centralizes the
``uuid5`` recipe so every producer (chunk / table / layout / pipeline)
shares one prefix->namespace table.

Bbox convention follows PLAN.md \u00a78: UIR stores a pixel rectangle
``(x1, y1, x2, y2)`` with all coordinates normalized to a 0-1000 virtual
canvas. We never store raw pixels; ``bbox_from_pixel`` is the only
public conversion entry point.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Final

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Per PLAN.md \u00a78 we deterministically derive node IDs via ``uuid5``.
# Python's stdlib ships a canonical URL namespace UUID (``uuid.NAMESPACE_URL``)
# which we re-use per prefix instead of inventing per-prefix URLs (the
# latter would be invalid input to ``uuid.UUID(...)``).
#
# To keep IDs distinct across prefixes we namespace the *seed* (not the
# UUID namespace): ``f"{prefix}|{parts...}"`` so the same source key
# under different prefixes yields different ids.

# Default BGE embedding model id (matches sentence-transformers hub id).
# Overrideable via ``$EMBEDDING_MODEL`` per .env.example.
DEFAULT_BGE_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"

# BGE (BERT-family) hard limit on input tokens (PLAN.md \u00a79 Phase I).
DEFAULT_BGE_MAX_TOKENS: Final[int] = 512

# Default chunk target-window tokens. Within [256, 512]; the Plan
# specifies 256-512 with 10-20% overlap.
DEFAULT_CHUNK_TARGET_TOKENS: Final[int] = 384

# Default overlap percentage between adjacent chunks.
DEFAULT_CHUNK_OVERLAP_PCT: Final[int] = 15

# Virtual canvas for bbox normalization (PLAN.md \u00a78).
BBOX_CANVAS: Final[int] = 1000

# Per-process state for the lazily-loaded BGE tokenizer (so chunk/auth paths
# share one instance instead of reloading per call).
_TOKENIZER_CACHE: dict[str, Any] = {}
_TOKENIZER_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Deterministic UIR IDs
# ----------------------------------------------------------------------------

# Allowed prefixes per PLAN.md \u00a78. Looked up by ``deterministic_node_id``
# so a typo fails loud (``ValueError``).
_VALID_PREFIXES: Final[frozenset[str]] = frozenset({
    "doc", "section", "table", "figure", "list", "chunk", "entity",
})


def deterministic_node_id(prefix: str, *parts: Any) -> str:
    """Return a UIR id ``f"{prefix}_{uuid5(NAMESPACE_URL, seed)}"``.

    ``prefix`` must be one of ``{doc, section, table, figure, list, chunk, entity}``
    per PLAN.md \u00a78. ``parts`` are stringified with ``str(...)`` and joined
    by ``"|"``. We use ``uuid.NAMESPACE_URL`` (Python's standard URL namespace)
    as the uuid5 namespace, and prepend the prefix to the seed so IDs under
    different prefixes never collide.

    Raises ``ValueError`` for unknown prefixes so a typo fails loudly.
    """
    if prefix not in _VALID_PREFIXES:
        raise ValueError(
            f"unknown UIR id prefix {prefix!r}; expected one of {sorted(_VALID_PREFIXES)}"
        )

    seed = "|".join((prefix, *(str(p) for p in parts)))
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, seed)}"


def strip_uir_prefix(uir_id: str) -> str:
    """Strip the ``<prefix>_`` from a UIR id, returning the bare UUID hex.

    PLAN.md \u00a79 Phase K: Weaviate's primary node ID requires a plain
    UUID with no prefix; full prefixed ids are stored separately as a
    BM25-indexed ``uir_id`` property.
    """
    if "_" not in uir_id:
        return uir_id
    return uir_id.split("_", 1)[1]


# ----------------------------------------------------------------------------
# Bounding-box helpers
# ----------------------------------------------------------------------------

def bbox_from_pixel(
    pixel_bbox: tuple[int, int, int, int],
    page_width_px: int,
    page_height_px: int,
) -> tuple[int, int, int, int]:
    """Normalize a pixel bbox to the 0-1000 virtual canvas.

    Coordinates outside the page bounds are clamped (PLAN.md \u00a78 invariant:
    bbox must be ``0 <= coord <= 1000``). Negative or oversize inputs
    typically arise from OCR caches that store raw image coords.
    """
    if page_width_px <= 0 or page_height_px <= 0:
        raise ValueError(
            f"page dimensions must be positive integers; "
            f"got width={page_width_px}, height={page_height_px}"
        )
    x1, y1, x2, y2 = pixel_bbox
    x1n = _scale(x1, page_width_px)
    y1n = _scale(y1, page_height_px)
    x2n = _scale(x2, page_width_px)
    y2n = _scale(y2, page_height_px)
    # Canonical ordering -- clamp before swap.
    return (min(x1n, x2n), min(y1n, y2n), max(x1n, x2n), max(y1n, y2n))


def _scale(coord_px: int, page_dim_px: int) -> int:
    """Linear scale a single pixel coord into the 0-1000 canvas."""
    clamped = max(0, min(coord_px, page_dim_px))
    return round((clamped * BBOX_CANVAS) / page_dim_px)


def bbox_union(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the tight bbox enclosing both ``a`` and ``b``.

    Assumes both inputs are already on the 0-1000 canvas with
    canonical ordering (``x1 <= x2, y1 <= y2``).
    """
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


# ----------------------------------------------------------------------------
# BGE tokenizer (lazy, cached, thread-safe)
# ----------------------------------------------------------------------------

def get_bge_tokenizer(model_id: str = DEFAULT_BGE_MODEL):
    """Return the cached BGE :class:`AutoTokenizer` for ``model_id``.

    Lazy-imports ``transformers`` so cold CLI startup stays cheap. The
    tokenizer is loaded once per ``(process, model_id)`` pair and reused
    across all chunker calls -- this is critical for the MVP because
    each chunk() runs ``tokenize()`` thousands of times.
    """
    cached = _TOKENIZER_CACHE.get(model_id)
    if cached is not None:
        return cached
    with _TOKENIZER_LOCK:
        cached = _TOKENIZER_CACHE.get(model_id)
        if cached is not None:
            return cached
        from transformers import AutoTokenizer  # lazy
        logger.debug("loading BGE tokenizer %s (first use; cached after)", model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        _TOKENIZER_CACHE[model_id] = tok
        return tok


def count_tokens(text: str, model_id: str = DEFAULT_BGE_MODEL) -> int:
    """Return the BGE token count for ``text`` (no special tokens, like chunking)."""
    tok = get_bge_tokenizer(model_id)
    # ``add_special_tokens=False`` matches how chunk boundaries are computed.
    return len(tok.encode(text, add_special_tokens=False))


__all__ = [
    "BBOX_CANVAS",
    "DEFAULT_BGE_MAX_TOKENS",
    "DEFAULT_BGE_MODEL",
    "DEFAULT_CHUNK_OVERLAP_PCT",
    "DEFAULT_CHUNK_TARGET_TOKENS",
    "UID_NAMESPACE",
    "bbox_from_pixel",
    "bbox_union",
    "count_tokens",
    "deterministic_node_id",
    "get_bge_tokenizer",
    "strip_uir_prefix",
]
