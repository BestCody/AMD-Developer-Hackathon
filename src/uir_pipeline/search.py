"""search -- semantic + title-priority passage search over a user's documents.

A single :func:`search` function backs both the ``/api/search`` endpoint and
the chat agent's ``search`` / ``get_more_sources`` tools. It ranks passages by
BGE cosine similarity to the query, with a constant boost added to every
passage of a document whose **title** matches the query -- so title matches
surface above content-only matches (title priority), while content cosine
orders within each group. Falls back to BM25-lite when embeddings are
unavailable.

Storage stays on-disk UIR JSON (the web path skips Weaviate), so this module
only reads + scores; it owns no database. It reuses the proven primitives in
:mod:`uir_pipeline.intent_filter` (chunk walk, cosine, keyword tokenize,
BM25-lite text score) and :mod:`uir_pipeline.embed` (BGE query embed) so the
ranking stays consistent with the chat's existing retrieval.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

from uir_pipeline.intent_filter import (
    _chunk_embedding,
    _cosine_score,
    _embed_intent,
    _intent_keywords,
    _text_score,
    _walk_chunks,
)

logger = logging.getLogger(__name__)

#: Boost added to a passage's score when its document's TITLE matches the
#: query. BGE cosine for unrelated passages sits near 0 and strong matches
#: rarely exceed ~0.6, so +0.30 reliably lifts title-matching documents above
#: content-only hits while leaving content cosine to order within each group.
TITLE_BOOST: Final[float] = 0.30

DEFAULT_TOP_K: Final[int] = 8


def _doc_title(doc: dict[str, Any], filename: str) -> str:
    meta = doc.get("metadata") or {}
    return (str(meta.get("title") or "").strip() or str(filename or "").strip()
            or "Untitled document")


def _title_matches(query_tokens: set[str], title: str) -> bool:
    """True iff any non-stopword query token appears in the document title."""
    if not query_tokens or not title:
        return False
    return bool(query_tokens & set(_intent_keywords(title)))


def search(
    docs: list[dict[str, Any]],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Rank passages across ``docs`` for ``query`` (title matches prioritized).

    ``docs`` is a list of ``{"job_id", "uir_path", "filename"}`` already
    filtered to the caller's DONE jobs. Returns up to ``top_k`` results sorted
    by descending combined score, each::

        {"job_id", "doc_id", "doc_title", "chunk_id", "page", "text",
         "score", "title_match"}

    ``job_id`` lets the UI open the source document; ``title_match`` flags
    title-priority hits so the search bar can badge them.
    """
    q = (query or "").strip()
    if not q or not docs:
        return []
    query_tokens = set(_intent_keywords(q))
    qvec = _embed_intent(q)  # None if BGE unavailable -> BM25-lite fallback

    results: list[dict[str, Any]] = []
    for d in docs:
        uir_path = d.get("uir_path")
        if not uir_path or not Path(uir_path).is_file():
            continue
        try:
            doc = json.loads(Path(uir_path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 -- skip unreadable docs
            logger.warning("search: could not read %s: %s", uir_path, exc)
            continue
        title = _doc_title(doc, d.get("filename") or "")
        title_match = _title_matches(query_tokens, title)
        root = (doc.get("structure") or {}).get("root") or doc.get("structure")
        chunks = _walk_chunks(root) if root else []
        if not chunks:
            continue
        avgdl = sum(len((c.get("text") or "").split()) for c in chunks) / max(1, len(chunks))
        doc_id = doc.get("id") or d.get("job_id")
        tok_list = list(query_tokens)
        for c in chunks:
            emb = _chunk_embedding(c)
            if qvec is not None and emb is not None:
                content = _cosine_score(qvec, emb)
            else:
                content = _text_score(tok_list, c, avgdl)
            # Drop passages with no signal at all (keeps the result list tight).
            if content == 0.0 and not title_match:
                continue
            score = content + (TITLE_BOOST if title_match else 0.0)
            if score < min_score:
                continue
            results.append({
                "job_id": d.get("job_id"),
                "doc_id": doc_id,
                "doc_title": title,
                "chunk_id": c.get("id"),
                "page": c.get("page"),
                "text": c.get("text") or "",
                "score": round(float(score), 6),
                "title_match": bool(title_match),
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[: max(0, int(top_k))]


__all__ = ["search", "TITLE_BOOST", "DEFAULT_TOP_K"]
