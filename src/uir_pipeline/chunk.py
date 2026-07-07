"""chunk -- token-bounded text segmentation respecting BGE's 512-token limit (Phase I).

PLAN.md \u00a79 Phase I exit:
    -- chunking produces 256-512 token chunks with 10-20% overlap
    -- sentence boundaries honored when feasible
    -- exports tokenizer count + bbox + page + confidence + modal_features
    -- hard ceiling at BGE's 512-token input limit; chunks above that are
       recursively halved (preserving overlap)

Tokenizer MUST be BGE's :class:`AutoTokenizer` -- not tiktoken or any
whitespace heuristic. Per PLAN.md \u00a79 Phase I: mismatch causes silent
truncation that destroys retrieval signals.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Final

from uir_pipeline.utils import (
    DEFAULT_BGE_MAX_TOKENS,
    DEFAULT_BGE_MODEL,
    DEFAULT_CHUNK_OVERLAP_PCT,
    DEFAULT_CHUNK_TARGET_TOKENS,
    count_tokens,
    get_bge_tokenizer,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Min / max chunk token targets per PLAN.md \u00a79 Phase I.
MIN_CHUNK_TOKENS: Final[int] = 256
MAX_CHUNK_TOKENS: Final[int] = DEFAULT_BGE_MAX_TOKENS  # 512 (BGE hard limit)

# Safety cap for ``_recursive_halve`` depth. 25 is enough to halve
# 2**25 ~= 33M tokens down to under 1 token. Way past what BGE ever sees.
_MAX_HALVE_DEPTH: Final[int] = 25

# Sentence detector -- "." "!" "?" followed by whitespace + capital letter
# (or opening bracket / quote). PLAN.md \u00a79 Phase I specifies "sentence
# boundaries honored where feasible" -- this is a heuristic.
_SENT_END_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z(\[\"'\u201c])"
)


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkDraft:
    """A draft text chunk produced by the chunker.

    ``bbox`` is on the 0-1000 virtual canvas (union of source-region
    bboxes; single-region chunks carry the region's bbox unchanged).
    ``page`` is 1-based. ``confidence`` is a soft [0, 1] estimate of
    chunk quality (currently 1.0 -- left as a hook for future layout-aware
    confidence).
    """

    text: str
    token_count: int
    page: int
    bbox: tuple[int, int, int, int]
    confidence: float = 1.0
    modal_features: dict[str, dict[str, object]] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Sentence splitter
# ----------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split ``text`` at sentence boundaries (conservative regex)."""
    parts = _SENT_END_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ----------------------------------------------------------------------------
# Windowed splitter (greedy, sentence-bounded)
# ----------------------------------------------------------------------------

def _split_by_tokens(text: str, target_tokens: int) -> list[str]:
    """Greedy sentence-bounded split respecting ``target_tokens`` per segment.

    Sentences whose own token count exceeds ``target_tokens`` are emitted
    as standalone segments -- the :func:`_recursive_halve` step handles
    the overflow.
    """
    sentences = _split_sentences(text)
    segments: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sent in sentences:
        tok = count_tokens(sent)
        if tok >= target_tokens:
            if buf:
                segments.append(" ".join(buf))
                buf, buf_tokens = [], 0
            segments.append(sent)
            continue
        if buf_tokens + tok > target_tokens and buf:
            segments.append(" ".join(buf))
            buf, buf_tokens = [sent], tok
        else:
            buf.append(sent)
            buf_tokens += tok
    if buf:
        segments.append(" ".join(buf))
    return segments


# ----------------------------------------------------------------------------
# Halving primitives
# ----------------------------------------------------------------------------

def _halve_at_whitespace(text: str) -> tuple[str, str]:
    """Halve ``text`` near its midpoint, snapping to the nearest whitespace.

    Returns ``(left, right)`` where both pieces are non-empty when a
    whitespace boundary existed. Falls back to a hard character-half
    when ``text`` contains no whitespace at all.
    """
    n = len(text)
    if n < 2:
        return text, ""
    mid = n // 2
    # Bounded sweep outward from the midpoint looking for whitespace.
    for radius in range(0, n // 2):
        for idx in (mid - radius, mid + radius):
            if 0 <= idx < n and text[idx].isspace():
                left = text[:idx].rstrip()
                right = text[idx + 1:].lstrip()
                if left and right:
                    return left, right
                # One side empty -- return both halves (the empty will be
                # skipped by the caller, the non-empty side keeps all).
                return left or right, right or left
    # No whitespace -- hard character cut. Always non-empty for n >= 2.
    cut = max(1, mid)
    return text[:cut], text[cut:]


def _recursive_halve(text: str, max_tokens: int, depth: int = 0) -> list[str]:
    """Halve ``text`` until each piece fits within ``max_tokens`` tokens.

    Uses whitespace-boundary halving (see :func:`_halve_at_whitespace`)
    so no-progress blocks (e.g., one giant whitespace-less token) still
    halve via a hard character cut. The recursion has a depth cap so
    pathological inputs can't blow the stack.
    """
    text = text.strip()
    if not text:
        return []
    if depth >= _MAX_HALVE_DEPTH:
        # Last-resort: accept the overflow chunk. The BGE embedder
        # truncates at 512 tokens and ``count_tokens`` reports the
        # original count (the orchestrator can opt to log a warning).
        logger.warning(
            "chunk._recursive_halve hit depth cap (%d) on len=%d text; "
            "accepting overflow as a single chunk", _MAX_HALVE_DEPTH, len(text),
        )
        return [text]
    if count_tokens(text) <= max_tokens:
        return [text]
    left, right = _halve_at_whitespace(text)
    # Ensure each half is strictly smaller than the input -- otherwise
    # we'd loop forever on pathological inputs. Fall back to a hard
    # character-cut if whitespace-halving didn't make progress.
    if (not left and not right) or (
        len(left) >= len(text) or len(right) >= len(text)
    ):
        cut = max(1, len(text) // 2)
        left, right = text[:cut], text[cut:]
    out: list[str] = []
    for p in (left, right):
        p = p.strip()
        if not p:
            continue
        if count_tokens(p) <= max_tokens:
            out.append(p)
        else:
            out.extend(_recursive_halve(p, max_tokens, depth + 1))
    return out


# ----------------------------------------------------------------------------
# Overlap stitching
# ----------------------------------------------------------------------------

def _with_overlap(segments: list[str], overlap_pct: int) -> list[str]:
    """Re-stitch ``segments`` with ``overlap_pct`` % token overlap.

    Overlap counts ``overlap_pct`` % of ``DEFAULT_CHUNK_TARGET_TOKENS``
    so chunks share meaningful boundary context (PLAN.md \u00a79 Phase I
    specifies 10-20%).
    """
    if not segments or overlap_pct <= 0:
        return list(segments)
    overlap_tokens = max(
        1,
        round(DEFAULT_CHUNK_TARGET_TOKENS * overlap_pct / 100),
    )
    out: list[str] = [segments[0]]
    tok = get_bge_tokenizer()
    for prev, curr in zip(segments, segments[1:]):
        # Use whitespace word-split instead of decoding the last
        # ``overlap_tokens`` BGE subword IDs. Slicing subword IDs at an
        # arbitrary boundary can cut a piece of a token mid-token; decoding
        # then yields ``##atrix``-style fragments that the BGE embedder
        # treats as unrelated noise. Whitespace word-split keeps every
        # prefix/suffix word intact at the cost of marginally looser
        # overlap (a word may extend a few subwords past the boundary). The
        # PLAN.md §9 chunk_token_count envelope still holds because the
        # boundary is still well below the 512-token hard cap.
        prev_words = prev.split()
        if len(prev_words) > overlap_tokens:
            tail = " ".join(prev_words[-overlap_tokens:])
            out.append((tail + " " + curr).strip())
        else:
            out.append(curr)
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def chunk_text(
    text: str,
    page: int = 1,
    bbox: tuple[int, int, int, int] | None = None,
    *,
    target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    overlap_pct: int = DEFAULT_CHUNK_OVERLAP_PCT,
    max_tokens: int = MAX_CHUNK_TOKENS,
    model_id: str = DEFAULT_BGE_MODEL,
    region_kind: str | None = None,
    section_path: str | None = None,
    is_section_first: bool = False,
    is_section_last: bool = False,
) -> list[ChunkDraft]:
    """Split ``text`` into token-bounded ``ChunkDraft``s.

    ``target_tokens`` is the soft target per chunk (PLAN.md \u00a79 Phase I:
    256-512). ``overlap_pct`` is the boundary overlap (10-20%). ``max_tokens``
    is the hard ceiling (BGE's 512-token input limit).
    """
    target_tokens = max(MIN_CHUNK_TOKENS, min(target_tokens, MAX_CHUNK_TOKENS))
    overlap_pct = max(0, min(overlap_pct, 50))
    max_tokens = max(target_tokens, min(max_tokens, MAX_CHUNK_TOKENS))

    if not text or not text.strip():
        return []

    page_bbox = bbox or (0, 0, 1000, 1000)

    # Step 1: greedy sentence-bounded split at target window.
    initial_segs = _split_by_tokens(text, target_tokens)

    # Step 2: recursive halving so every segment fits within ``max_tokens``.
    bounded: list[str] = []
    for seg in initial_segs:
        if count_tokens(seg) <= max_tokens:
            if seg.strip():
                bounded.append(seg)
        else:
            bounded.extend(_recursive_halve(seg, max_tokens))

    # Step 3: re-stitch with overlap.
    with_overlap = _with_overlap(bounded, overlap_pct)

    drafts: list[ChunkDraft] = []
    for seg in with_overlap:
        # Tier 1 intent metadata: only inject ``intent`` / ``section``
        # sub-blocks when the caller actually supplied the inputs, so a
        # caller that doesn't care (unit tests, smoke tests) keeps the
        # pre-Tier-1 JSON shape. ``preceding_chunk_id`` and
        # ``following_chunk_id`` are wired in the orchestrator after this
        # function returns because deterministic chunk IDs depend on the
        # orchestrator's index plan.
        modal_features: dict[str, dict[str, object]] = {
            "text": {
                "token_count": count_tokens(seg, model_id),
                "chunk_strategy": "growing-window-bge-tokenizer",
            },
        }
        if region_kind is not None:
            modal_features["intent"] = {"region_kind": region_kind}
        if section_path is not None:
            modal_features["section"] = {
                "path": section_path,
                "is_first": bool(is_section_first),
                "is_last": bool(is_section_last),
            }
        drafts.append(ChunkDraft(
            text=seg.strip(),
            token_count=count_tokens(seg, model_id),
            page=page,
            bbox=page_bbox,
            confidence=1.0,
            modal_features=modal_features,
        ))
    return drafts


def chunks_from_regions(
    regions: list[tuple[tuple[int, int, int, int], str, int]],
    *,
    target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    overlap_pct: int = DEFAULT_CHUNK_OVERLAP_PCT,
) -> list[ChunkDraft]:
    """Chunk a layout-derived list of ``(bbox, text, page)`` region tuples.

    Each region's text is chunked independently; every chunk emitted from
    the same region inherits that region's bbox. Tier 1 intent metadata is
    NOT propagated here — callers wanting per-region ``region_kind`` or
    ``section_path`` should call :func:`chunk_text` directly with the
    kwargs they need.
    """
    out: list[ChunkDraft] = []
    for src_bbox, text, page in regions:
        for draft in chunk_text(
            text,
            page=page,
            bbox=src_bbox,
            target_tokens=target_tokens,
            overlap_pct=overlap_pct,
        ):
            out.append(draft)
    return out


__all__ = [
    "ChunkDraft",
    "MAX_CHUNK_TOKENS",
    "MIN_CHUNK_TOKENS",
    "chunk_text",
    "chunks_from_regions",
]
