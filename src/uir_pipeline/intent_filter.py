"""intent_filter -- post-orchestrator reader-mode chunk selection.

The orchestrator's standard output is the *full* UIR document. For
LLM ``reader-mode`` workloads (the eventual goal: reduce token cost by
issuing narrowly-shaped queries), the calling agent doesn't need every
chunk -- ``"show me lithium isotope abundance"`` should yield the 3-5
relevant chunks (plus their enclosing section) instead of all 133.

This module is a pure-Python post-processor that:

    1. Reads the produced full UIR JSON from disk.
    2. Tokenises ``intent`` into lowercase keywords (with a small
       English stopword set AND a tech-term safelist so "AI / ML / GPU"
       survive the de-noising pass).
    3. **Tree-walks** the document hierarchy (post-fix-#3 sections are
       nested under root), scoring each chunk by:
           a) keyword substring match on chunk.text
           b) keyword substring match on the enclosing section's title
           c) cosine similarity against the chunk's persisted BGE
              embedding (skip if no vector)
    4. Persists a filtered UIR JSON alongside the full one and emits a
       ``matches`` array with ``chunk_id / score / section_path /
       neighbour_chunk_ids`` so the LAN UI can show ranked, contextual
       retrieval results without re-scanning.

Implementation is intentionally simple and synchronous: runs after the
orchestrator's main :func:`pipeline.run` so an orchestrator change does
not disturb this layer.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# Module logger so the silent BGE-import / embed failures surface in the
# orchestrator's per-doc log instead of vanishing into a `return None`.
# Anything below filters down to ``WARNING`` for the user-facing web layer.
logger = logging.getLogger(__name__)

# Tracks the most-recent BGE failure reason (set inside ``_embed_intent``).
# The filter reads it after the embed attempt so ``intent_filter_meta`` can
# surface ``embed_unavailable_reason`` -- the agent then knows whether to
# suspect (a) model install issue, (b) OOM, (c) missing model checkpoint.
_EMBED_FAIL_REASON: str | None = None


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

# Technical / scientific 2-3 char tokens that a strict stopword + length
# filter would drop but that carry real retrieval signal in arxiv / bio /
# econ PDFs. Lowercased for case-insensitive match. Extend here as new
# vocabulary appears; the test fixtures pin the canonical set.
_TECH_TERMS: frozenset[str] = frozenset({
    # ML / AI acronyms
    "ai", "ml", "dl", "rl", "nlp", "cv", "rag", "llm", "vae", "gan",
    "bert", "gpt", "gpu", "cpu", "ram", "hbm", "ssd", "nic", "vpn",
    # Econ / bio / med / chem
    "gdp", "ct", "mri", "xr", "dna", "rna", "crispr",
    # Compound-symbol tokens commonly seen
    "ph",
})


# ----------------------------------------------------------------------------
# Tokenization (fix #3: short-token safelist + broader regex)
# ----------------------------------------------------------------------------

def _intent_keywords(intent: str) -> list[str]:
    """Split ``intent`` into lowercase keywords (drops stops except safelist).

    Keeps tokens >= 2 chars (was >= 3). Splits on any NON
    [A-Za-z0-9-] run so `6Li`, `BGE-512`, `vec_dim=384`-style tokens all
    survive. Stable first-seen order preserved so the front-end renders
    the matched-keywords list in the user's own query order.
    """
    out: list[str] = []
    if not intent:
        return out
    seen: set[str] = set()
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9]|[A-Za-z0-9]", intent.lower()):
        if len(tok) < 2:
            continue
        if tok in _STOPWORDS and tok not in _TECH_TERMS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


# ----------------------------------------------------------------------------
# Tree-walk helpers (fix #2: recurse into sections)
# ----------------------------------------------------------------------------

def _walk_chunks(root: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every chunk in the document tree (depth-first)."""
    out: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = [root]
    while stack:
        n = stack.pop()
        if n.get("type") == "chunk":
            out.append(n)
            continue
        stack.extend(n.get("children") or [])
    return out


def _walk_sections(root: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every section in the document tree."""
    out: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = [root]
    while stack:
        n = stack.pop()
        if n.get("type") == "section":
            out.append(n)
        stack.extend(n.get("children") or [])
    return out


def _chunk_text_lower(c: dict[str, Any]) -> str:
    """Lowercased ``text`` field of a UIR ChunkNode-ish dict.

    Lower-casing once is cheaper than per-keyword ``.lower()`` when
    scanning many chunks.
    """
    return (c.get("text") or "").lower()


def _section_title_lower(s: dict[str, Any]) -> str:
    return (s.get("title") or "").lower()


def _chunk_embedding(c: dict[str, Any]) -> list[float] | None:
    """Return the persisted BGE embedding for ``c`` or None if absent."""
    return (
        c.get("modal_features", {}).get("vector", {}).get("embedding")
        or None
    )


def _neighbours(c: dict[str, Any]) -> list[str]:
    mf = c.get("modal_features", {})
    return [
        x for x in (
            mf.get("preceding_chunk_id", {}).get("chunk_id"),
            mf.get("following_chunk_id", {}).get("chunk_id"),
        ) if x
    ]


# ----------------------------------------------------------------------------
# BGE cosine ranking (fix #4)
# ----------------------------------------------------------------------------

def _cosine_score(a: list[float], b: list[float]) -> float:
    """Dot product is cosine for L2-normalized vectors (BGE outputs are)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(ai * bi for ai, bi in zip(a[:n], b[:n]))


def _embed_intent(intent: str) -> list[float] | None:
    """Compute the BGE embedding for ``intent`` text (cold-loads on first use).

    Returns ``None`` if the embed call fails (model not installed, OOM,
    torchvision import loop, etc.). The orchestrator already lazy-imports
    the model -- first call within a process pays the load cost, all
    subsequent calls hit the module-level cache. Failures are LOGGED with
    full exc_info so the failure surface is visible in orchestrator logs;
    the most-recent reason is captured in the module-level
    :data:`_EMBED_FAIL_REASON` for downstream surfacing.
    """
    global _EMBED_FAIL_REASON
    if not intent or not intent.strip():
        _EMBED_FAIL_REASON = None
        return None
    try:
        from uir_pipeline.embed import embed_texts
        out = embed_texts([intent])
        if out.vectors and out.vectors[0]:
            _EMBED_FAIL_REASON = None
            return [float(v) for v in out.vectors[0]]
        # Wholly-empty embedding -- sentence-transformers sometimes does
        # this when the model checkpoint is empty / mid-write. Treat as
        # failure and surface.
        _EMBED_FAIL_REASON = "embed_texts returned empty vectors (model checkpoint or batch issue)"
        logger.warning("BGE returned empty vectors for intent=%r", intent)
        return None
    except Exception as exc:
        _EMBED_FAIL_REASON = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "BGE embed failed for intent=%r; falling back to text-only scoring (%s)",
            intent, _EMBED_FAIL_REASON, exc_info=True,
        )
        return None


def _cosine_ranked(
    intent_vec: list[float],
    chunks: list[dict[str, Any]],
    top_k: int = 20,
) -> list[tuple[float, dict[str, Any]]]:
    """Cosine-rank ``chunks`` against ``intent_vec``; return [(score, chunk)].

    Drops chunks without an embedding or with a mismatched dimensionality.
    Identity equal/dimensionally-incompatible inputs are silently skipped
    rather than raised so the filter stays best-effort on partial UIRs.
    """
    if intent_vec is None or not chunks:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    dim = len(intent_vec)
    for c in chunks:
        cvec = _chunk_embedding(c)
        if cvec is None or len(cvec) != dim:
            continue
        scored.append((_cosine_score(intent_vec, cvec), c))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


# ----------------------------------------------------------------------------
# BM25-lite text-only scoring (iteration-2 fix #2)
# ----------------------------------------------------------------------------
# Used when the BGE embed failed (e.g. torchvision import loop, model
# checkpoint missing). Pure-Python, no scipy/numpy dep, ~O(N*K) where
# N = #chunks and K = #intent-tokens. Returns a score in roughly the same
# range as BGE cosine so the agent sees comparable ordering across docs
# (the front-end renders them in the same `score` field).
#
# The score is sum-of per-token BM25-like terms with a token-in-title
# boost so a chunk whose section-title matches an intent token outranks a
# chunk that merely contains the same token in its body text.

_BM25_K1: float = 1.2
_BM25_B: float = 0.75
_TITLE_BOOST: float = 0.30


def _text_score(
    intent_tokens: list[str],
    chunk: dict[str, Any],
    avgdl: float,
) -> float:
    """BM25-lite score of ``chunk`` against ``intent_tokens``.

    ``avgdl`` is the document-average chunk-text-length used in the BM25
    length-normalisation. Returns 0.0 when there is no match signal.
    """
    if not intent_tokens:
        return 0.0
    text_lo: str = _chunk_text_lower(chunk)
    title_lo: str = ""
    spath = (chunk.get("modal_features", {}).get("section", {}) or {}).get("path")
    if spath:
        title_lo = spath.lower()
    if not text_lo and not title_lo:
        return 0.0
    dl: float = max(1.0, len(text_lo.split()))
    norm: float = (1.0 - _BM25_B) + _BM25_B * (dl / max(1.0, avgdl))
    score: float = 0.0
    for tok in intent_tokens:
        if not tok:
            continue
        tf_body = text_lo.count(tok)
        tf_title = title_lo.count(tok) if title_lo else 0
        tf = tf_body + tf_title
        if tf == 0:
            continue
        term: float = (tf * (1.0 + _BM25_K1)) / (tf + _BM25_K1 * norm)
        if tf_title:
            term += _TITLE_BOOST * tf_title
        score += term
    return score


# ----------------------------------------------------------------------------
# Section-walked keyword match (fix #2) with topic-aware widening (fix #6)
# ----------------------------------------------------------------------------

def _build_keyword_matched_set(
    root: dict[str, Any],
    keywords: list[str],
) -> tuple[set[str], set[str], set[str]]:
    """Tree-walk and return ``(title_matched_section_ids, text_matched_chunk_ids, section_id_set)``.

    ``section_id_set`` is the set of ALL section IDs in the tree (used to
    recognise section boundaries when re-assembling the output below).
    """
    title_sections: set[str] = set()
    text_chunks: set[str] = set()
    all_section_ids: set[str] = set()
    if not keywords:
        return title_sections, text_chunks, all_section_ids
    stack: list[dict[str, Any]] = [root]
    while stack:
        n = stack.pop()
        n_type = n.get("type")
        if n_type == "section":
            n_id = n.get("id", "")
            all_section_ids.add(n_id)
            title_lo = _section_title_lower(n)
            if any(kw in title_lo for kw in keywords):
                title_sections.add(n_id)
        elif n_type == "chunk":
            text_lo = _chunk_text_lower(n)
            if any(kw in text_lo for kw in keywords):
                text_chunks.add(n.get("id", ""))
        stack.extend(n.get("children") or [])
    return title_sections, text_chunks, all_section_ids


def _compose_keyword_output(
    root: dict[str, Any],
    title_matched_sections: set[str],
    text_matched_chunks: set[str],
) -> list[dict[str, Any]]:
    """Re-assemble the matched subtree, keeping enclosing sections intact.

    Rules:
    - If a section's title matched -> keep the whole section (all its chunks).
    - If a section's SOME chunks matched but title didn't -> keep ONLY the
      matching chunks inside that section (preserves the section label).
    - Top-level (root) chunks -> kept if they matched; not lifted into a
      synthetic section, since root-level chunks don't have a parent.
    """
    out: list[dict[str, Any]] = []
    for child in (root.get("children") or []):
        ctype = child.get("type")
        cid = child.get("id", "")
        if ctype == "chunk":
            if cid in text_matched_chunks:
                out.append(child)
        elif ctype == "section":
            if cid in title_matched_sections:
                # Title match -> whole section
                out.append(child)
            else:
                # Filter children: keep only those whose chunk id matched.
                kept = [c for c in (child.get("children") or [])
                        if c.get("id", "") in text_matched_chunks]
                if kept:
                    out.append({**child, "children": kept})
        else:
            # Unknown node type -> pass through unchanged.
            out.append(child)
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def filter_uirstream_by_intent(
    uir_path: Path | os.PathLike[str] | str,
    intent: str,
    out_path: Path | os.PathLike[str] | str | None = None,
    *, top_k: int = 20,
    cosine_threshold: float = 0.20,
) -> dict[str, Any]:
    """Apply intent-filter to ``uir_path`` and write the filtered JSON.

    ``uir_path`` and (optional) ``out_path`` accept either :class:`Path`
    or a raw ``str`` (or any :class:`os.PathLike`) -- common when callers
    come from a CLI / shell / shell-piped URL. Internally coerced to
    :class:`Path` so the rest of this module can use ``.read_text()`` /
    ``.write_text()`` uniformly. Regression coverage:
    :func:`tests.test_intent_filter.test_filter_uirstream_accepts_str_path`
    pins the string contract -- a real bug surfaced here when this
    function was called with a ``str`` from the LAN UI shell.

    Returns a summary dict with the ranked matches, scoring metadata, and
    counts so the LAN UI / LLM agent can surface retrieval quality::

        {
            "intent": str,
            "keywords": list[str],
            "scoring": "cosine+bge" | "keyword" | "keyword+topic-widen",
            "matches": [
                {
                    "chunk_id": str,           # uuid-prefixed UIR id
                    "score": float | None,     # cosine/BGE = float; section-lift / no-match synthetic = None
                    "score_kind": str,         # "cosine" | "bm25-lite" | "section-lift" | "no_match_fallback"
                    "section_path": str | None,
                    "section_title": str | None,
                    "neighbour_chunk_ids": [str, ...],
                    "text": str | None,        # raw chunk body -- inline so agents can cite without re-lookup. ``None`` ONLY on the no_match_fallback synthetic; real matches always carry text or ``""``.
                }
            ],
            "topics_hit": [str, ...],
            "no_match_fallback": bool,
            "out_path": str,
        }

    Behaviour notes:
        * If intent is empty / blank, returns ``matched=[]``, ``no_match_fallback=False`` and emits an unchanged copy.
        * If BGE vectors are present we score by cosine similarity; if not, we fall back to keyword match.
        * Topic-widening (fix #6): if any extracted keyword overlaps with ``semantics.topics``, lifted matches also include the entire sections that contain the cosine hit.
        * The filtered JSON keeps the full UIR envelope (``source``, ``metadata``, ``semantics``, ``provenance``) intact. Only ``structure.root.children`` is replaced by the matched subtree; ``structure.root.intent_filter`` carries the ranking metadata.
    """
    # Coerce path-like inputs to ``Path()`` so CLI / shell callers can pass
    # raw strings. ``Path(Path(...))`` is idempotent so existing Path
    # callers pay no cost; the type hint is also widened so static analysis
    # doesn't false-fail on str inputs. Round-trips through ``pathlib`` are
    # the codebase convention -- ``pipeline.run`` does the same.
    uir_path = Path(uir_path)
    if out_path is not None:
        out_path = Path(out_path)
    keywords = _intent_keywords(intent)
    src = json.loads(uir_path.read_text(encoding="utf-8"))
    root = src.get("structure", {}).get("root", {})

    # Compute document-average chunk length once so _text_score stays O(1)
    # per (chunk, keyword) inside the loop below. avgdl is required for the
    # BM25 length-normalisation term.
    all_chunks = _walk_chunks(root)
    if all_chunks:
        total_dl: float = float(sum(
            len((c.get("text") or "").split()) for c in all_chunks
        ))
        avgdl: float = max(1.0, total_dl / len(all_chunks))
    else:
        avgdl = 1.0    # Tree-walked keyword match (fix #2) -- finds title-matched sections AND
    # text-matched chunks anywhere in the tree.
    title_sections, text_chunks, _ = _build_keyword_matched_set(
        root, keywords,
    )

    # BGE cosine ranking (fix #4) -- runs whenever intent is non-empty AND
    # at least one chunk in the tree carries an embedding. Skips silently
    # on legacy UIRs (pre-fix-#1) so we never crash on missing vectors.
    # Iteration-2 #2-B fix: capture the failure reason INTO A LOCAL var
    # immediately after the call so we don't depend on the module-global
    # state (thread-safety under concurrent /api/run requests).
    intent_vec = _embed_intent(intent)
    embed_fail_local: str | None = _EMBED_FAIL_REASON
    cosine_scored: list[tuple[float, dict[str, Any]]] = (
        _cosine_ranked(intent_vec, all_chunks, top_k=top_k) if intent_vec else []
    )
    # Filter by threshold; iteration-2 fix #3 sets a per-query-floor:
    # single helper that consolidates the two thresholding branches.
    # If any chunk scores >= cosine_threshold, floor = top - 0.15. If no
    # chunk reaches that trigger, floor = max(0.05, trigger) so the
    # ambient score (still meaningful) surfaces.
    matched_scored: list[tuple[float, dict[str, Any]]] = []
    if cosine_scored:
        trigger = max((s for s, _ in cosine_scored), default=0.0)
        floor = (
            max(trigger - 0.15, 0.05)
            if trigger >= cosine_threshold
            else max(0.05, trigger)
        )
        matched_scored = [(s, c) for s, c in cosine_scored if s >= floor]
    using_cosine = bool(matched_scored)
    scoring = "cosine+bge" if using_cosine else "keyword"

    # Topic-aware widening (iteration-2 fix #4): a topic is usually a
    # multi-token phrase ("Stellar nucleosynthesis", "dark matter halos"),
    # so an exact kw==topic match rarely fires. Match each keyword against
    # the SET of single tokens from all topic phrases ("stellar" hits
    # "Stellar nucleosynthesis"). Iter-2 stricter check: require
    # keyword length >= 3 OR substring containment against a topic token
    # at least 4 chars long, so ""/"ai" don't false-positive against
    # "main"/"fail"/"train"-class topic tokens.
    topics = [t for t in (src.get("semantics", {}).get("topics") or []) if t]
    topic_set: set[str] = {t.lower().strip() for t in topics if t.strip()}
    topic_token_set: set[str] = set()
    for t in topics:
        for tw in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", t.lower()):
            if len(tw) >= 2:
                topic_token_set.add(tw)
    topic_hits: list[str] = []
    seen_hit: set[str] = set()
    for kw in keywords:
        if kw in seen_hit:
            continue
        if len(kw) >= 3 and (kw in topic_token_set or any(
            tt.startswith(kw) or tt.endswith(kw) for tt in topic_token_set
            if len(tt) >= 4
        )):
            topic_hits.append(kw)
            seen_hit.add(kw)


# Resolve matched SETS: start with cosine picks, merge keyword matches.
    matched_chunk_ids_in_subtree: set[str] = set()
    if using_cosine:
        for _score, c in matched_scored:
            matched_chunk_ids_in_subtree.add(c.get("id", ""))
        # Cosine-only neighbours: pull in +-1 chunk IDs so the lifted
        # section gets in-context chunks, not just the lone cosine hit.
        for _score, c in matched_scored:
            for nb in _neighbours(c):
                matched_chunk_ids_in_subtree.add(nb)
    matched_chunk_ids_in_subtree |= text_chunks
    # A title-matched section implies its chunks are also "matched":
    # surface them in matches_payload so the agent sees the same set
    # that's in the output subtree (consistency rule).
    if title_sections:
        for s in _walk_sections(root):
            if s.get("id", "") in title_sections:
                for ch in (s.get("children") or []):
                    if ch.get("id", ""):
                        matched_chunk_ids_in_subtree.add(ch["id"])

    # Build the output subtree via the keyword composer (which handles
    # title-matched sections + per-section filtered chunks).
    out_children = _compose_keyword_output(
        root, title_sections, matched_chunk_ids_in_subtree,
    )

    no_match = (
        bool(keywords)
        and not title_sections
        and not matched_chunk_ids_in_subtree
    )
    if no_match:
        out_children = list(root.get("children") or [])

    # Build the ranked response (fix #5) -- ordered by score desc.
    # Iteration-2 fix 2A: keyword-only path gets a BM25-lite score so the
    # agent receives ranking signal even when BGE didn't load. Each
    # match carries a ``score_kind`` discriminator so UI/tooling callers
    # know whether to threshold on cosine logic (-1, +1) OR BM25 logic
    # (~[0, K]). Without this discriminator the two ranges get conflated
    # and the agent silently misranks BM25 hits below cosine hits.
    matches_payload: list[dict[str, Any]] = []
    # ``text`` is included in the payload so downstream agents can render
    # an inline snippet WITHOUT a second lookup against the output UIR
    # subtree. Earlier rounds omitted this and probe agents fetching the
    # ``chunk_text`` field got ``None`` -- this was discovered in the
    # empirical eval against real-arXiv UIRs. The value carries the
    # pre-chunking text body verbatim; for the no-match-fallback synthetic
    # the field is intentionally absent (the synthetic has no underlying
    # chunk).
    if using_cosine:
        for score, c in matched_scored:
            matches_payload.append({
                "chunk_id": c.get("id", ""),
                "score": round(float(score), 4),
                "score_kind": "cosine",
                "section_path": (
                    c.get("modal_features", {}).get("section", {}).get("path")
                ),
                "section_title": None,  # resolved by id lookup below if needed
                "neighbour_chunk_ids": _neighbours(c),
                "text": c.get("text", ""),
            })
    else:
        for c in all_chunks:
            if c.get("id", "") not in matched_chunk_ids_in_subtree:
                continue
            text_sc: float = _text_score(keywords, c, avgdl)
            if text_sc > 0.0:
                # BM25-lite hit: chunk body / section title contains a
                # keyword with measurable tf. Honest score + confident kind.
                kind = "bm25-lite"
                score_out: float = round(text_sc, 4)
            else:
                # Section-tree title lift pulled this chunk in but its
                # own chunk-level section_path didn't fire _text_score.
                # Use a distinct score_kind so the agent doesn't conflate
                # "no scoring signal" with "scored zero by BM25-lite".
                kind = "section-lift"
                score_out = None
            matches_payload.append({
                "chunk_id": c.get("id", ""),
                "score": score_out,
                "score_kind": kind,
                "section_path": (
                    c.get("modal_features", {}).get("section", {}).get("path")
                ),
                "section_title": None,
                "neighbour_chunk_ids": _neighbours(c),
                "text": c.get("text", ""),
            })
        # Sort by score desc (None last) so the agent reads best first.
        matches_payload.sort(
            key=lambda m: (
                -(m["score"] if isinstance(m["score"], (int, float)) else -1.0)
            ),
        )

    # Resolve section_title for each match (does a tiny id lookup, O(|sec|))
    sec_by_id: dict[str, dict[str, Any]] = {
        s.get("id", ""): s for s in _walk_sections(root)
    }
    for m in matches_payload:
        spath = m.get("section_path")
        for s in sec_by_id.values():
            if s.get("title") == spath:
                m["section_title"] = s.get("title")
                break

    # Iteration-2 bug fix #2: when ``no_match`` is true the output subtree
    # preserved the full document, but ``matches_payload`` was empty --
    # which is a state the agent cannot reason about (``matches=[]`` with
    # ``output=full-tree``). Emit ONE synthetic match so the routing
    # signal is unambiguous: the agent sees ``matches=[synthetic]`` AND
    # ``score_kind="no_match_fallback"`` rather than guessing why a
    # no-match intent returned a full document.
    if no_match and not matches_payload:
        matches_payload.append({
            "chunk_id": "*no_match_fallback*",
            "score": None,
            "score_kind": "no_match_fallback",
            "section_path": None,
            "section_title": None,
            "neighbour_chunk_ids": [],
            "fallback_reason": (
                "intent did not match any chunk; full document preserved"
                f" ({len(all_chunks)} chunks total)"
            ),
        })

    intent_filter_meta = {
        "intent": intent,
        "keywords": keywords,
        "scoring": scoring + ("+topic-widen" if topic_hits and scoring == "cosine+bge" else ""),
        "matches": matches_payload,
        "topics_hit": topic_hits,
        "topic_set": sorted(topic_set),
        "total_chunks": len(all_chunks),
        "matched_chunks": len(matches_payload),
        "no_match_fallback": no_match,
        # Iteration-2 fix #2-B (thread-safe): surface the BGE failure reason
        # via the LOCAL snapshot taken right after the `_embed_intent` call
        # so concurrent /api/run requests don't pollute each other's meta.
        # NULL = BGE succeeded or wasn't attempted. NOT the module-global
        # `_EMBED_FAIL_REASON` because that's mutable cross-request state.
        "embed_unavailable_reason": embed_fail_local,
    }

    new_src = dict(src)
    new_struct = dict(new_src.get("structure", {}))
    new_root = dict(new_struct.get("root", {}))
    new_root["children"] = out_children
    new_root["intent_filter"] = intent_filter_meta
    new_struct["root"] = new_root
    new_src["structure"] = new_struct

    # Output path: explicit caller-supplied wins; otherwise a sibling with
    # ``.intent`` inserted before the final suffix.
    if out_path is None:
        target = uir_path.parent / (uir_path.stem + ".intent" + uir_path.suffix)
    else:
        target = out_path
    target.write_text(json.dumps(new_src, indent=2), encoding="utf-8")

    return {
        "intent": intent,
        "keywords": keywords,
        "scoring": intent_filter_meta["scoring"],
        "matches": matches_payload,
        "topics_hit": topic_hits,
        "no_match_fallback": no_match,
        "out_path": str(target),
    }


__all__ = ["filter_uirstream_by_intent"]
