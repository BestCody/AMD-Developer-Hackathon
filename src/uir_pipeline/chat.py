"""chat -- grounded Q&A over documents the user has already converted.

Two halves:

``retrieve()``
    Ranks chunks across one or more UIR JSON documents against a query.
    This deliberately reuses :mod:`uir_pipeline.intent_filter`'s scoring
    internals rather than reimplementing them -- the BGE cosine path and
    the BM25-lite text fallback already exist, are tested, and must not
    drift from what ``/api/run?intent=`` does.

``answer()``
    Sends the retrieved chunks to a Fireworks-hosted chat model with a
    grounding instruction, and returns the reply plus the citations it
    was given. Sibling of :mod:`uir_pipeline.fireworks_vision` -- same
    OpenAI-compatible endpoint, same fail-soft contract (return an error
    dict, never raise at the caller).

Environment variables:
    ``FIREWORKS_API_KEY``     (required) Fireworks AI API token.
    ``FIREWORKS_CHAT_MODEL``  (optional) Model ID override.
    ``FIREWORKS_BASE_URL``    (optional) API base URL override.

Grounding is enforced by prompt, not by construction. A language model
can still ignore the instruction and answer from parametric memory. The
citations we return are the chunks we *supplied*, not a proof that the
model used them. Treat the answer as attributable-to-context, not
guaranteed-from-context.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

_DEFAULT_CHAT_MODEL: Final[str] = "accounts/fireworks/models/minimax-m3"
_DEFAULT_BASE_URL: Final[str] = "https://api.fireworks.ai/inference/v1"
_DEFAULT_MAX_TOKENS: Final[int] = 1024

#: How many chunks to put in front of the model. Each chunk is ~256
#: tokens, so 6 keeps the grounding block under ~1.5k tokens and leaves
#: the model plenty of room to answer.
DEFAULT_TOP_K: Final[int] = 6

#: Chunks scoring below this against the query are dropped before they
#: reach the model. Without a floor, an off-topic question retrieves the
#: six *least bad* chunks and the model confabulates an answer around them.
#:
#: Measured, not guessed. Swept over a 267-chunk UIR of "Attention Is All
#: You Need" with BAAI/bge-small-en-v1.5, using 10 questions the paper
#: answers and 6 it does not:
#:
#:     out-of-domain top-1 cosine:      max 0.570
#:     answer-bearing chunk cosine:     min 0.683   (10 of 10 retrieved)
#:
#: The populations separate cleanly, with no overlap. 0.58 sits below the
#: worst answer-bearing chunk and above every off-topic query: it rejects
#: all 6 off-topic queries and drops none of the 10 answers. The original
#: 0.62 sat *above* the worst answer-bearing chunk, discarding a passage
#: that contained the answer while rejecting no additional off-topic query.
#:
#: (The first sweep put the worst answer-bearing chunk at 0.614 and found
#: only 9 of 10, because PDF extraction had split `0.1` into `0 . 1`; see
#: `docling_extract.normalize_extracted_text`. Fixing that raised the floor
#: of the answer population, which widened the gap rather than moving it.)
#:
#: Caveat: one document, one embedding model, one topic. Re-run the sweep
#: if either the model or the corpus changes; the gap is what matters, not
#: the constant.
MIN_COSINE_SCORE: Final[float] = 0.58

_SYSTEM_PROMPT: Final[str] = (
    "You answer questions about the user's documents using ONLY the "
    "numbered context passages provided.\n\n"
    "Rules:\n"
    "- Ground every claim in a passage and cite it inline as [1], [2], etc.\n"
    "- If the passages do not contain the answer, say exactly: "
    "\"I can't answer that from the documents you've converted.\" "
    "Do not guess, and do not fall back on general knowledge.\n"
    "- Never invent a citation number that wasn't given to you.\n"
    "- Be concise and factual. Quote figures and names exactly as written."
)


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------

def _load_doc(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- a bad doc must not kill the query
        logger.warning("chat: could not read UIR at %s: %s", path, exc)
        return None


def retrieve(
    uir_paths: list[Path],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Rank chunks across ``uir_paths`` against ``query``.

    Returns a list of context dicts sorted best-first::

        {"doc_id", "doc_title", "chunk_id", "page", "text", "score"}

    Empty when nothing clears :data:`MIN_COSINE_SCORE` (semantic path) or
    scores zero (text-fallback path). An empty return is meaningful and
    the caller must not send an empty grounding block to the model.
    """
    # Private imports are intentional: these are the *same* ranking
    # functions /api/run's intent filter uses. Duplicating them here
    # would let the two paths silently diverge.
    from uir_pipeline.intent_filter import (
        _chunk_embedding,
        _cosine_score,
        _embed_intent,
        _intent_keywords,
        _text_score,
        _walk_chunks,
    )

    query = (query or "").strip()
    if not query:
        return []

    # Flatten every chunk from every document, remembering its origin.
    pool: list[tuple[dict[str, Any], dict[str, Any]]] = []  # (chunk, doc)
    for p in uir_paths:
        doc = _load_doc(p)
        if not doc:
            continue
        root = ((doc.get("structure") or {}).get("root")) or {}
        for chunk in _walk_chunks(root):
            if (chunk.get("text") or "").strip():
                pool.append((chunk, doc))

    if not pool:
        return []

    def _ctx(chunk: dict[str, Any], doc: dict[str, Any], score: float) -> dict[str, Any]:
        meta = doc.get("metadata") or {}
        return {
            "doc_id": doc.get("id") or "unknown",
            "doc_title": meta.get("title") or "Untitled document",
            "chunk_id": chunk.get("id") or "",
            "page": chunk.get("page"),
            "text": (chunk.get("text") or "").strip(),
            "score": round(float(score), 4),
        }

    query_vec = _embed_intent(query)

    if query_vec is not None:
        dim = len(query_vec)
        scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for chunk, doc in pool:
            cvec = _chunk_embedding(chunk)
            if cvec is None or len(cvec) != dim:
                continue
            scored.append((_cosine_score(query_vec, cvec), chunk, doc))
        if scored:
            scored.sort(key=lambda t: -t[0])
            return [
                _ctx(c, d, s)
                for s, c, d in scored[:top_k]
                if s >= MIN_COSINE_SCORE
            ]
        logger.info("chat: no chunk embeddings usable; falling back to text scoring")

    # Text fallback: BGE unavailable, or the corpus predates embeddings.
    tokens = _intent_keywords(query)
    if not tokens:
        return []
    lengths = [len((c.get("text") or "").split()) for c, _ in pool]
    avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
    text_scored = [
        (_text_score(tokens, chunk, avgdl), chunk, doc) for chunk, doc in pool
    ]
    text_scored.sort(key=lambda t: -t[0])
    return [_ctx(c, d, s) for s, c, d in text_scored[:top_k] if s > 0.0]


# ----------------------------------------------------------------------------
# Fireworks chat completion
# ----------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "FIREWORKS_API_KEY is not set. Set it in your .env file or environment."
        )
    return key.strip()


def _get_chat_model() -> str:
    return os.environ.get("FIREWORKS_CHAT_MODEL", _DEFAULT_CHAT_MODEL).strip()


def _get_base_url() -> str:
    return os.environ.get("FIREWORKS_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/")


def _format_context_block(contexts: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, c in enumerate(contexts, start=1):
        loc = f"{c['doc_title']}"
        if c.get("page") is not None:
            loc += f", p. {c['page']}"
        parts.append(f"[{i}] ({loc})\n{c['text']}")
    return "\n\n".join(parts)


#: Matches an inline citation marker: [1], [12]. Deliberately not matching
#: [1, 2] or [a] -- the prompt asks for one integer per bracket, and a marker
#: we don't recognise is left alone rather than mangled.
#:
#: The lookbehind keeps ``array[0]`` and ``x[1]`` -- subscripts in quoted code,
#: which this corpus is full of -- from being read as citations and deleted. A
#: real marker follows a space or punctuation, never an identifier character.
_CITATION_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\w\]])\[(\d+)\]")


def _validate_citations(reply: str, n_contexts: int) -> tuple[str, list[int]]:
    """Strip citation markers that point at passages we never supplied.

    The system prompt says "never invent a citation number", but a prompt is
    a request, not a constraint. A model that answers with "[4]" when three
    passages were given produces a claim the reader cannot check and that
    *looks* sourced -- strictly worse than an uncited claim.

    Returns the cleaned reply and the sorted invalid numbers found. Only the
    marker is removed; the surrounding sentence is left untouched, because we
    can't know which passage (if any) the model meant.
    """
    invalid: set[int] = set()

    def _sub(match: re.Match[str]) -> str:
        num = int(match.group(1))
        if 1 <= num <= n_contexts:
            return match.group(0)
        invalid.add(num)
        return ""

    cleaned = _CITATION_RE.sub(_sub, reply)
    if invalid:
        # Removing " [4]" leaves a double space or a space before punctuation.
        cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, sorted(invalid)


def _cited_indices(reply: str, n_contexts: int) -> list[int]:
    """1-based passage numbers the reply actually cites, in order of appearance."""
    seen: list[int] = []
    for m in _CITATION_RE.finditer(reply):
        num = int(m.group(1))
        if 1 <= num <= n_contexts and num not in seen:
            seen.append(num)
    return seen


def answer(
    query: str,
    contexts: list[dict[str, Any]],
    *,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    model: str | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Answer ``query`` from ``contexts`` via a Fireworks chat model.

    Returns ``{success, answer, citations, model, usage, error}``. Never
    raises: transport and API failures come back as ``success=False``
    with a human-readable ``error``, matching
    :func:`uir_pipeline.fireworks_vision.describe_image`.
    """
    if not contexts:
        # Short-circuit: with nothing retrieved there is nothing to ground
        # against, and asking the model anyway invites a hallucinated
        # answer that *looks* sourced.
        return {
            "success": True,
            "answer": "I can't answer that from the documents you've converted.",
            "citations": [],
            "cited": [],
            "invalid_citations": [],
            "model": None,
            "usage": {},
            "grounded": False,
        }

    resolved_model = model or _get_chat_model()
    try:
        api_key = _get_api_key()
    except ValueError as exc:
        return {
            "success": False,
            "error": str(exc),
            "answer": "",
            "citations": [],
            "model": resolved_model,
            "usage": {},
        }
    base_url = _get_base_url()

    user_content = (
        f"Context passages:\n\n{_format_context_block(contexts)}\n\n"
        f"Question: {query}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    # Prior turns give the model pronoun/topic continuity. Truncated to the
    # last few so a long session can't push the grounding block out of the
    # context window -- the passages matter more than the backchat.
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": user_content})

    request_body = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    logger.info(
        "fireworks chat call: model=%s contexts=%d query=%r",
        resolved_model, len(contexts), query[:80],
    )

    try:
        import requests as _requests

        response = _requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=request_body,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            return {
                "success": False,
                "error": "empty choices in Fireworks API response",
                "answer": "",
                "citations": contexts,
                "model": resolved_model,
                "usage": {},
            }

        reply = choices[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        logger.info("fireworks chat response: model=%s tokens=%s", resolved_model, usage)

        reply, invalid = _validate_citations(reply.strip(), len(contexts))
        if invalid:
            logger.warning(
                "chat model cited %d passage(s) that were never supplied "
                "(%s; only %d given); markers stripped from the answer",
                len(invalid), invalid, len(contexts),
            )

        return {
            "success": True,
            "answer": reply,
            "citations": contexts,
            "cited": _cited_indices(reply, len(contexts)),
            "invalid_citations": invalid,
            "model": resolved_model,
            "usage": usage,
            "grounded": True,
        }

    except Exception as exc:  # noqa: BLE001 -- fail-soft, mirrors fireworks_vision
        # Surface the status line but not the raw body: Fireworks echoes the
        # request on some 4xx, and the request contains document text.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f"HTTP {status}" if status else f"{type(exc).__name__}: {exc}"
        logger.exception("fireworks chat call failed")
        return {
            "success": False,
            "error": f"Chat model call failed ({detail}).",
            "answer": "",
            "citations": contexts,
            "model": resolved_model,
            "usage": {},
        }


__all__ = [
    "DEFAULT_TOP_K",
    "MIN_COSINE_SCORE",
    "answer",
    "retrieve",
]
