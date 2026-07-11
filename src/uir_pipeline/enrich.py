"""enrich -- spaCy NER + co-occurrence relationships (Phase J).

PLAN.md Section 9 Phase J exit:
    -- spaCy NER produces >=1 entity on a fixture with known entities
    -- co-occurrence relationships within chunks
    -- topics stub returns [] (LDA deferred to Phase 2)

``en_core_web_sm`` is the canonical small English model. We load once
per process (lazy import + per-thread safe-lazy-cache).
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Final

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityDraft:
    """A single named-entity hit (NOT yet a UIR Entity -- no id; orchestrator assigns)."""
    text: str
    type: str  # spaCy label_ (e.g. "PERSON", "ORG", "GPE")
    confidence: float  # synthetic 1.0 (spaCy doesn't emit probabilities for en_core_web_sm)


@dataclass(frozen=True)
class RelationshipDraft:
    """A co-occurrence relationship between two entities. ``from_`` and
    ``to`` carry the canonical text; the orchestrator maps to UIR ids.
    """
    from_text: str
    to_text: str
    type: str = "co-occurrence"
    confidence: float = 1.0


@dataclass(frozen=True)
class EnrichmentResult:
    """Result of enriching a chunk list."""
    entities: list[EntityDraft] = field(default_factory=list)
    relationships: list[RelationshipDraft] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Lazy singleton for the spaCy pipeline
# ----------------------------------------------------------------------------

_NLP_CACHE: dict[str, Any] = {}
_NLP_LOCK = threading.Lock()
DEFAULT_SPACY_MODEL: Final[str] = "en_core_web_sm"


class _DummyNLP:
    """No-op stand-in for a :class:`spacy.language.Language` when the model
    is unavailable.

    Surface area intentionally minimal: :func:`enrich_chunks` only reads
    ``doc.ents``, so ``__call__(text)`` returns a plain object with an
    empty ``ents`` tuple. Every downstream consumer
    (chunks / UIR / UMR / grounded chat retrieval) treats the
    entity and relationship lists in :class:`EnrichmentResult` as additive
    metadata, so a model-down :class:`_DummyNLP` lets the whole job
    succeed with zero NER hits instead of aborting at
    ``spacy.load(model_id)``. Mirrors the best-effort enrichment pattern
    in :mod:`uir_pipeline.caption`.
    """

    # Cached at module-import time -- one allocation for the entire process
    # is enough; every call returns the same empty doc. Callers only
    # iterate ``ents``, so the doc stays effectively immutable in practice;
    # assertion follows up with ``is`` to verify the cache hit.
    _EMPTY_DOC = SimpleNamespace(ents=())

    def __call__(self, text: str) -> Any:
        return self._EMPTY_DOC


def _get_nlp(model_id: str = DEFAULT_SPACY_MODEL):
    """Return the cached :class:`spacy.language.Language` for ``model_id``.

    If the requested model is not installed (``spacy.load`` raises
    ``OSError [E050]`` -- this happens when ``python -m spacy download
    <model>`` was skipped), substitute :class:`_DummyNLP` and emit a
    single WARNING. The job still completes; only NER hits are lost.
    Cached alongside a real pipeline under ``_NLP_CACHE`` so the warning
    fires exactly once per process (matching the lock-guarded cache
    pattern below).
    """
    cached = _NLP_CACHE.get(model_id)
    if cached is not None:
        return cached
    with _NLP_LOCK:
        cached = _NLP_CACHE.get(model_id)
        if cached is not None:
            return cached
        import spacy  # lazy
        logger.debug("loading spaCy pipeline %s (first use; cached after)", model_id)
        try:
            nlp = spacy.load(model_id)
        except OSError as exc:
            # spaCy raises ``OSError`` with ``[E050]`` for the missing-model
            # case (also covers ``[E104]`` for broken packages and ``[E097]``
            # for invalid model arg). Specific on purpose, not
            # ``Exception`` -- a real bug in this code path should still
            # propagate so the worker crash-isolation path can flag it.
            logger.warning(
                "spaCy model %r not found; NER enrichment will be empty. "
                "Run `python -m spacy download %s` to fix. (Underlying: %s)",
                model_id, model_id, exc,
            )
            nlp = _DummyNLP()
        _NLP_CACHE[model_id] = nlp
        return nlp


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _entity_confidence_label_map() -> dict[str, float]:
    """Heuristic per-label confidence weighting.

    ``en_core_web_sm`` does not emit a softmax probability per entity,
    so we assign a stable per-label confidence. ORG / PERSON / GPE are
    typically the highest-quality categories; DATE / TIME / MONEY are
    noisier but still useful.
    """
    return {
        "PERSON":       0.92,
        "ORG":          0.90,
        "GPE":          0.88,
        "LOC":          0.85,
        "FAC":          0.85,
        "EVENT":        0.80,
        "WORK_OF_ART":  0.78,
        "LAW":          0.78,
        "LANGUAGE":     0.85,
        "NORP":         0.82,
        "PRODUCT":      0.72,
        "DATE":         0.70,
        "TIME":         0.70,
        "MONEY":        0.78,
        "PERCENT":      0.78,
        "QUANTITY":     0.72,
        "ORDINAL":      0.65,
        "CARDINAL":     0.65,
    }


def _spacy_entity_to_draft(ent) -> EntityDraft:
    """Convert a :class:`spacy.tokens.Span` into a stable :class:`EntityDraft`."""
    label = ent.label_ or "ENTITY"
    conf_map = _entity_confidence_label_map()
    confidence = conf_map.get(label, 0.6)
    return EntityDraft(
        text=ent.text.strip(),
        type=label,
        confidence=confidence,
    )


def _cooccurrence_relationships(
    entities_in_chunk: list[EntityDraft],
) -> list[RelationshipDraft]:
    """Pairwise co-occurrence relationships among entities in the same chunk.

    We avoid self-loops and dedup pair order (a-b == b-a). With >20
    entities per chunk, this would explode combinatorially; we cap at
    20 pairs per chunk to keep the output bounded.
    """
    rels: list[RelationshipDraft] = []
    seen: set[tuple[str, str]] = set()
    max_pairs = 20
    for i, a in enumerate(entities_in_chunk):
        for b in entities_in_chunk[i + 1:]:
            if a.text == b.text:
                continue
            # An explicit 2-tuple: `tuple(sorted(...))` widens to
            # `tuple[str, ...]`, which is not `seen`'s element type.
            key = (a.text, b.text) if a.text < b.text else (b.text, a.text)
            if key in seen:
                continue
            seen.add(key)
            rels.append(RelationshipDraft(
                from_text=a.text,
                to_text=b.text,
                type="co-occurrence",
                confidence=round(min(a.confidence, b.confidence), 3),
            ))
            if len(rels) >= max_pairs:
                return rels
    return rels


def _dedupe_entities(entities: list[EntityDraft]) -> list[EntityDraft]:
    """Drop surface-form duplicates by ``(text.lower(), type)`` keeping max confidence."""
    by_key: dict[tuple[str, str], EntityDraft] = {}
    for ent in entities:
        k = (ent.text.lower(), ent.type)
        cur = by_key.get(k)
        if cur is None or cur.confidence < ent.confidence:
            by_key[k] = ent
    return list(by_key.values())


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def enrich_chunks(
    chunk_texts: list[str],
    *,
    model_id: str = DEFAULT_SPACY_MODEL,
) -> EnrichmentResult:
    """Run spaCy NER per chunk + co-occurrence relationships, dedup entities.

    ``chunk_texts`` is a flat list of per-chunk text strings. Topics are
    populated from the top-5 most-frequent entity surface-forms (Fix Plan
    item #5). The LDA path remains a Phase 2 stub per PLAN.md Section 3.
    """
    if not chunk_texts:
        return EnrichmentResult(entities=[], relationships=[], topics=[])

    nlp = _get_nlp(model_id)
    all_entities: list[EntityDraft] = []
    all_relationships: list[RelationshipDraft] = []
    for text in chunk_texts:
        if not text or not text.strip():
            continue
        # Disable parser for speed (we only need NER) -- Phase 2 may
        # re-enable for relation extraction.
        doc = nlp(text)
        chunk_entities = [_spacy_entity_to_draft(ent) for ent in doc.ents]
        all_entities.extend(chunk_entities)
        all_relationships.extend(_cooccurrence_relationships(chunk_entities))

    deduped = _dedupe_entities(all_entities)
    # Topics stand-in for the LDA stack (PLAN.md Section 3, deferred to
    # Phase 2). The top-5 most-frequent entity surface-forms in a paper
    # (Fix Plan item #5) are a reasonable proxy for "what this paper is
    # about" until the LDA path lands. We surface-form-lower-case before
    # counting so duplicates (BERT vs Bert) collapse into one topic. A
    # 0-count Counter maps to topics=[] which is identical to the prior
    # behaviour for entity-free inputs.
    topic_counter = Counter(
        e.text.strip().lower() for e in deduped if e.text.strip()
    )
    topics = [t for t, _ in topic_counter.most_common(5)]
    return EnrichmentResult(
        entities=deduped,
        relationships=all_relationships,
        topics=topics,
    )


__all__ = [
    "DEFAULT_SPACY_MODEL",
    "EnrichmentResult",
    "EntityDraft",
    "RelationshipDraft",
    "enrich_chunks",
]
