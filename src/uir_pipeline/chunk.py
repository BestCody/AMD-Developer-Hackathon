"""chunk -- paragraph-aware token-bounded text segmentation.

PLAN.md \u00a79 Phase I exit + empirical-eval follow-up:
    -- chunking produces 256-512 token chunks with 10-20% overlap
    -- paragraph boundaries are the PRIMARY split (replaces the previous
       sentence-bounded splitter that mid-paragraph spliced text the agent
       received as "mangled")
    -- exports tokenizer count + bbox + page + confidence + modal_features
    -- hard ceiling at BGE's 512-token input limit; single paragraphs that
       overflow are recursively halved (preserving the safety mechanism)
    -- overlap stitching uses a fixed small word-tail (≤16 words) so the
       preceding chunk's tail does NOT bleed mid-sentence into the next

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
# (or opening bracket / quote). Kept exported for :func:`_split_sentences`
# backward-compat tests; the new core path is paragraph-aware.
_SENT_END_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z(\[\"'\u201c])"
)

# Paragraph break detector. ``\n`` followed by any whitespace, then 1+
# newlines (and any whitespace on the second line). Handles ``\n\n``,
# ``\n\n\t``, ``\r\n\r\n`` etc. Single ``\n`` (a line-wrap inside a
# paragraph) is NOT a paragraph break.
_PARAGRAPH_BREAK_RE: Final[re.Pattern[str]] = re.compile(r"\n\s*\n+")

# Fixed small word-tail overlap. The previous algorithm blew the overlap
# budget into 20%+ of the target window AND clipped subword IDs, producing
# fragments like "``##atrix``" that the BGE embedder treated as noise.
# A 16-word tail keeps boundary retrieval recall while guaranteeing no
# mid-sentence bleed (whitespace-split preserves every overlapping word).
_OVERLAP_TAIL_WORDS: Final[int] = 16


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
# Sentence splitter (legacy / retained)
# ----------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split ``text`` at sentence boundaries (conservative regex).

    Kept exported so legacy tests in :mod:`tests.test_chunk` continue to
    pass without churn. NOT used in the production chunker path --
    paragraph-aware bundling replaced sentence-bounded splitting to fix
    the "mangled and spliced" agent-output bug surfaced in the empirical
    eval.
    """
    parts = _SENT_END_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ----------------------------------------------------------------------------
# Paragraph splitter + bundler
# ----------------------------------------------------------------------------

def _split_into_paragraphs(text: str) -> list[str]:
    """Split ``text`` on ``\\n\\s*\\n+`` paragraph breaks.

    Returns trimmed non-empty paragraphs in source order. Single ``\\n``
    line wraps are preserved inside each paragraph (NOT used as a split
    boundary). ``\\r\\n\\r\\n`` (Windows page-break-style) is also a
    paragraph break via the ``\\n\\s*\\n+`` rule.
    """
    blocks = _PARAGRAPH_BREAK_RE.split(text)
    return [b.strip() for b in blocks if b.strip()]


def _bundle_paragraphs_for_chunks(
    paragraphs: list[str],
    target_tokens: int,
    max_tokens: int,
) -> list[str]:
    """Greedy bundle ``paragraphs`` into chunks respecting token envelopes.

    Rules:
        * Adding a paragraph that fits within ``max_tokens`` accumulates.
        * When the buffer reaches ``target_tokens`` after accumulation,
          flush the buffer as one chunk.
        * When adding a paragraph would overflow ``max_tokens``, flush
          the buffer first then start a new buffer.
        * A SINGLE paragraph larger than ``max_tokens`` is recursively
          halved (preserves the BGE 512-token ceiling while staying
          inside the paragraph geometry).
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def _flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            # Re-join paragraphs with ``\\n\\n`` so a chunk's text preserves
            # the original paragraph breaks. The agent downstream sees
            # distinct paragraphs rather than a single joined line.
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_tokens = 0

    for par in paragraphs:
        par = par.strip()
        if not par:
            continue
        par_tok = count_tokens(par)
        if par_tok > max_tokens:
            # Genuinely oversized single paragraph -- recurse-halve.
            # The bleed into other middle-oversized paragraphs is rare;
            # flushing the buffer first keeps the small-paragraph output
            # contiguous.
            _flush()
            chunks.extend(_recursive_halve(par, max_tokens))
            continue
        if buf_tokens + par_tok <= max_tokens:
            buf.append(par)
            buf_tokens += par_tok
            if buf_tokens >= target_tokens:
                _flush()
        else:
            _flush()
            buf.append(par)
            buf_tokens = par_tok
            if buf_tokens >= target_tokens:
                _flush()
    _flush()
    return chunks


# ----------------------------------------------------------------------------
# Halving primitives
# ----------------------------------------------------------------------------

def _halve_at_whitespace(text: str) -> tuple[str, str]:
    """Halve ``text`` near its midpoint, snapping to the nearest whitespace.

    Both halves are GUARANTEED to contain at least ``_MIN_HALVE_FRACTION``
    (25%) of the input length -- this prevents :func:`_recursive_halve`
    from looping forever on degenerate inputs (e.g., a giant whitespace-
    less token, or a near-midpoint whitespace that produces an empty
    left half, which then gets dropped by the caller and re-stitches the
    right half back to the original -- the no-progress bug that
    previously caused ``test_bundle_paragraphs_oversize_recurses`` to
    return a single chunk for an actually-oversize paragraph).
    """
    n = len(text)
    if n < 2:
        return text, ""
    mid = n // 2
    min_half = max(1, n // 4)  # both halves must be >= 25% of ``n``.
    for radius in range(0, n // 2):
        for idx in (mid - radius, mid + radius):
            if not (0 <= idx < n) or not text[idx].isspace():
                continue
            left = text[:idx].rstrip()
            right = text[idx + 1:].lstrip()
            if len(left) >= min_half and len(right) >= min_half:
                return left, right
    # No whitespace split produced two meaningfully smaller halves. Hard
    # character cut -- always yields non-empty halves for ``n >= 2``.
    cut = max(1, mid)
    return text[:cut].rstrip(), text[cut:].lstrip()


def _recursive_halve(text: str, max_tokens: int, depth: int = 0) -> list[str]:
    """Halve ``text`` until each piece fits within ``max_tokens`` tokens.

    Uses whitespace-boundary halving (see :func:`_halve_at_whitespace`)
    with the ``_MIN_HALVE_FRACTION`` guarantee so each recursion
    strictly shrinks the input -- no infinite loops on degenerate
    whitespace layouts. Falls back to a hard character cut when no
    whitespace boundary yields two meaningful halves. The recursion
    has a depth cap so pathological inputs can't blow the stack.
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
    """Re-stitch ``segments`` with a small fixed word-tail overlap.

    Replaces the previous overlap strategy whose decoded-subword tail
    ("``tail + space + curr``" with N BGE-token subwords) bled mid-sentence
    text into the next chunk. The new rule bounds the overlap to at most
    :data:`_OVERLAP_TAIL_WORDS` (16) trailing words; previous chunks whose
    tail is shorter than the limit are not extended. Words are split on
    whitespace only -- no subword clipping -- so the overlap tail is
    always made of complete words.

    The overlap is also **cap-aware**: if naively stitching the full
    word-tail would push the resulting chunk past BGE's :data:`MAX_CHUNK_TOKENS`
    ceiling, the tail is trimmed word-by-word until the stitched chunk
    fits. This guarantees per-chunk token counts respect the cap even
    when ``_recursive_halve`` halved near (but not under) the cap.
    """
    if not segments or overlap_pct <= 0:
        return list(segments)
    pct_yield = max(
        1,
        round(DEFAULT_CHUNK_TARGET_TOKENS * overlap_pct / 100),
    )
    word_tail = max(1, min(pct_yield, _OVERLAP_TAIL_WORDS))
    # Minimum overlap word count -- keeps at least a couple of words of
    # context even under aggressive cap-trimming. Falls through to
    # no-overlap (just ``curr``) when ``curr`` alone is already >= cap.
    _MIN_OVERLAP_WORDS: Final[int] = 2
    out: list[str] = [segments[0]]
    for prev, curr in zip(segments, segments[1:]):
        prev_words = prev.split()
        if len(prev_words) < _MIN_OVERLAP_WORDS:
            out.append(curr)
            continue
        n_tail = min(word_tail, len(prev_words))
        trimmed = False
        while n_tail >= _MIN_OVERLAP_WORDS:
            tail = " ".join(prev_words[-n_tail:])
            stitched = (tail + " " + curr).strip()
            if count_tokens(stitched) <= MAX_CHUNK_TOKENS:
                out.append(stitched)
                trimmed = True
                break
            n_tail -= 1
        if not trimmed:
            # ``curr`` alone was already >= cap -- emit as-is so the
            # downstream ``count_tokens`` call can surface the overflow.
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
    """Split ``text`` into paragraph-aware token-bounded ``ChunkDraft``s.

    Pipeline (replaces the previous sentence-bounded pipeline):

    1. **Paragraph split** -- :func:`_split_into_paragraphs` on ``\\n\\s*\\n+``.
       Single ``\\n`` line wraps stay inside the paragraph.
    2. **Greedy bundling** -- :func:`_bundle_paragraphs_for_chunks` packs
       paragraphs into chunks of ``target_tokens`` size, splitting only
       when the next paragraph would exceed ``max_tokens``.
    3. **Cap-size enforcement** -- any segment that exceeded ``max_tokens``
       (a real path: a single 1k-token paragraph in a dense PDF) flows
       through :func:`_recursive_halve`.
    4. **Overlap stitching** -- :func:`_with_overlap` re-stitches with a
       fixed ``_OVERLAP_TAIL_WORDS`` (16) word-tail overlap so the agent
       downstream doesn't see subword-clipped fragments.

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

    # Step 1: paragraph split on \n\s*\n+.
    paragraphs = _split_into_paragraphs(text)

    # Step 2: greedy bundle paragraphs into target_tokens windows, with
    # the cap-size safety valve for genuinely oversized single paragraphs.
    bounded = _bundle_paragraphs_for_chunks(
        paragraphs,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )

    # Step 3: re-stitch with a small fixed word-tail overlap. Paragraph
    # geometry is preserved because the join step kept ``\n\n`` between
    # paragraphs.
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
                "chunk_strategy": "paragraph-aware-bge-tokenizer",
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
    NOT propagated here -- callers wanting per-region ``region_kind`` or
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


# ----------------------------------------------------------------------------
# Pageless pagination (PLAN §17 §Multi-format)
# ----------------------------------------------------------------------------
# Pageless formats (TXT/MD/code/CSV/RTF) have no native page concept but
# downstream Stages 4-11 key on ``page=`` for position semantics, BGE
# chunk overlap stitching, and section_path tracing. We synthesise pages
# here at a ``DEFAULT_PAGELESS_PAGE_TOKENS`` token budget, splitting
# strictly on paragraph boundaries so a structural heading never ends up
# stranded in a window with one orphan line.
DEFAULT_PAGELESS_PAGE_TOKENS: Final[int] = 2000


def paginate_pageless(
    text: str,
    *,
    max_tokens: int = DEFAULT_PAGELESS_PAGE_TOKENS,
    model_id: str = DEFAULT_BGE_MODEL,
) -> list[tuple[int, str]]:
    """Split ``text`` into ``[(page_number, joined_text), ...]``.

    Each window holds roughly ``max_tokens`` BGE tokens, sliced on
    paragraph breaks (``\\n\\n``) so single-paragraph ``\\n`` wraps stay
    intact. A single oversized paragraph is always absorbed whole
    (``len(pages) >= 1``); splitting mid-paragraph would break the
    downstream chunker's overlap-stitch logic. Page numbers are
    1-based to match the existing PDF contract; an empty / whitespace
    input returns ``[]``.

    Used by:
        -- :mod:`src.uir_pipeline.ingest_rtf` (RTF after striprtf decode)
        -- the new ``pipeline._run_text_route`` branch for TXT/MD/code/CSV
        -- direct callers in tests (unit-tested via
           ``tests/test_paginate_pageless.py``).
    """
    if not text or not text.strip():
        return []
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []
    pages: list[tuple[int, str]] = []
    cur_paras: list[str] = []
    cur_tokens = 0
    page_idx = 1
    for para in paragraphs:
        para_tokens = count_tokens(para, model_id=model_id)
        # Roll a new page when adding this paragraph would exceed
        # ``max_tokens`` AND we already have content on the current
        # page. Empty pages absorb any single oversized paragraph so
        # we never infinite-loop on a 50k-token blob.
        if cur_paras and (cur_tokens + para_tokens) > max_tokens:
            pages.append((page_idx, "\n\n".join(cur_paras)))
            page_idx += 1
            cur_paras = []
            cur_tokens = 0
        cur_paras.append(para)
        cur_tokens += para_tokens
    if cur_paras:
        pages.append((page_idx, "\n\n".join(cur_paras)))
    return pages


__all__ = [
    "DEFAULT_PAGELESS_PAGE_TOKENS",
    "ChunkDraft",
    "MAX_CHUNK_TOKENS",
    "MIN_CHUNK_TOKENS",
    "chunk_text",
    "chunks_from_regions",
    "paginate_pageless",
    "_split_sentences",
]
