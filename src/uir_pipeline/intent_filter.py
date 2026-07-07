"""intent_filter -- post-orchestrator reader-mode chunk selection.

The orchestrator's standard output is the *full* UIR document. For
LLM ``reader-mode`` workloads (the eventual goal: reduce token cost by
issuing narrowly-shaped queries), the calling agent doesn't need every
chunk -- ``"show me the bounding box / algorithm table for section 3.2"``
should yield 4 chunks instead of the whole 54.

This module is a pure-Python post-processor that:

    1. Reads the produced full UIR JSON from disk.
    2. Tokenises ``intent`` into lowercase keywords (with a small
       English stopword set so "show me the table" produces ``["table"]``).
    3. Keeps chunks whose ``text`` contains any keyword -- case-
       insensitive substring match. (Tier 2 work: extend to match
       ``section.path`` / ``region_kind`` for structured intents.)
    4. Persists a filtered UIR JSON alongside the full one and emits
       ``matched_chunks`` / ``total_chunks`` counts so the LAN UI can
       surface ``"X of Y chunks (intent: ...)"`` without re-scanning.

Implementation is intentionally simple and synchronous: runs after the
orchestrator's main :func:`pipeline.run` so an orchestrator change does
not disturb this layer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Tiny English stop-word set. We don't pull in NLTK / spaCy: a hard-coded
# list is enough for a single-document intent filter and costs zero bytes.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "of", "on", "or",
    "show", "tell", "that", "the", "their", "them", "these", "they",
    "this", "to", "us", "was", "we", "what", "when", "where", "which",
    "who", "why", "with", "you", "your", "me", "my", "give", "about",
    "find", "list", "all", "any", "only", "some",
})


def _intent_keywords(intent: str) -> list[str]:
    """Split ``intent`` into lowercase keywords (drops stops + < 3 chars).

    Stable first-seen order preserved so the front-end renders the
    matched-keywords list in the user's own query order.
    """
    out: list[str] = []
    if not intent:
        return out
    seen: set[str] = set()
    for tok in re.split(r"[^A-Za-z0-9]+", intent.lower()):
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _chunk_text_lower(c: dict[str, Any]) -> str:
    """Lowercased ``text`` field of a UIR ChunkNode-ish dict.

    Lower-casing once is cheaper than per-keyword ``.lower()`` when
    scanning many chunks.
    """
    return (c.get("text") or "").lower()


def filter_uirstream_by_intent(
    uir_path: Path,
    intent: str,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Apply intent-filter to ``uir_path`` and write the filtered JSON.

    Returns a summary dict::

        {
            "intent": str,
            "total_chunks": int,
            "matched_chunks": int,
            "keywords": list[str],
            "out_path": str,
            "no_match": bool,
        }

    Parameters:
        uir_path: Path to the full UIR JSON (output of ``pipeline.run``).
        intent:   Natural-language reader intent (e.g. "show me section 3.2").
        out_path: Where to write the filtered JSON.  If ``None``, write
            to ``<uir_path>.intent.uir.json`` next to the source file.

    Behaviour notes:
        * If intent is empty / blank, returns ``{"matched_chunks": 0,
          "no_match": False}`` and produces an unchanged copy.
        * If keywords are extracted but no chunk matches any of them,
          we keep the **full** chunk set and emit
          ``no_match=True`` so the LAN UI can warn the user ("expanded
          to full document -- intent was over-restrictive"). This is
          better than returning an empty result that 200-OKs.
        * The filtered JSON keeps the full UIR envelope (``source``,
          ``metadata``, ``semantics``, ``provenance``) intact so downstream
          consumers (Weaviate upsert, BGE retriever) still see consistent
          state. Only ``structure.root.children`` is reduced.
    """
    keywords = _intent_keywords(intent)
    src = json.loads(uir_path.read_text())
    chunks = src.get("structure", {}).get("root", {}).get("children", [])

    matched: list[dict[str, Any]] = []
    for c in chunks:
        text_lo = _chunk_text_lower(c)
        if any(kw in text_lo for kw in keywords):
            matched.append(c)

    no_match = bool(keywords) and not matched
    if no_match:
        # Graceful fallback: keep everything so the user sees SOMETHING.
        matched = list(chunks)

    src["structure"]["root"]["children"] = matched
    src["structure"]["root"]["intent_filter"] = {
        "intent": intent,
        "keywords": keywords,
        "matched_chunks": len(matched) if not no_match else 0,
        "total_chunks": len(chunks),
        "no_match_fallback": no_match,
    }

    # Output path: explicit caller-supplied wins; otherwise a sibling
    # with ``.intent`` inserted before the final suffix. We do NOT use
    # ``Path.with_suffix`` here -- on ``fixture.uir.json`` that swaps
    # the WHOLE ``.json`` suffix (silently losing the ``.uir`` part).
    # ``stem = "fixture.uir"`` + ``suffix = ".json"`` concat is unambiguous.
    if out_path is None:
        target = uir_path.parent / (uir_path.stem + ".intent" + uir_path.suffix)
    else:
        target = out_path
    target.write_text(json.dumps(src, indent=2))

    return {
        "intent": intent,
        "total_chunks": len(chunks),
        "matched_chunks": len(matched) if not no_match else 0,
        "keywords": keywords,
        "out_path": str(target),
        "no_match": no_match,
    }


__all__ = ["filter_uirstream_by_intent"]
