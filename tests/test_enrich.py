"""tests/test_enrich.py -- Phase J enrich module tests.

``spaCy`` is mocked: the tests use a stub ``Language`` whose ``__call__``
returns a ``Doc``-like with a customizable ``.ents`` list. This keeps
``en_core_web_sm`` (and its 40 MB download) out of unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import pytest

from uir_pipeline.enrich import (
    DEFAULT_SPACY_MODEL,
    EnrichmentResult,
    EntityDraft,
    _cooccurrence_relationships,
    _dedupe_entities,
    _spacy_entity_to_draft,
    enrich_chunks,
)


# ----------------------------------------------------------------------------
# _spacy_entity_to_draft
# ----------------------------------------------------------------------------

@dataclass
class _FakeSpan:
    text: str
    label_: str


def test_spacy_entity_to_draft_uses_label_map_for_known_labels():
    ent = _FakeSpan(text="Alice", label_="PERSON")
    d = _spacy_entity_to_draft(ent)
    assert isinstance(d, EntityDraft)
    assert d.text == "Alice"
    assert d.type == "PERSON"
    assert 0.0 <= d.confidence <= 1.0


def test_spacy_entity_to_draft_falls_back_for_unknown_label():
    d = _spacy_entity_to_draft(_FakeSpan(text="x", label_="WAT"))
    assert d.confidence == 0.6  # unknown-label fallback


def test_spacy_entity_to_draft_strips_whitespace():
    d = _spacy_entity_to_draft(_FakeSpan(text="   Alice   ", label_="PERSON"))
    assert d.text == "Alice"


# ----------------------------------------------------------------------------
# _dedupe_entities
# ----------------------------------------------------------------------------

def test_dedupe_entities_keeps_max_confidence():
    a = EntityDraft(text="Alice", type="PERSON", confidence=0.9)
    b = EntityDraft(text="ALICE", type="PERSON", confidence=0.85)
    out = _dedupe_entities([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.9


def test_dedupe_entities_keeps_distinct_labels():
    a = EntityDraft(text="Apple", type="ORG", confidence=0.95)
    b = EntityDraft(text="Apple", type="PRODUCT", confidence=0.7)
    out = _dedupe_entities([a, b])
    assert len(out) == 2


# ----------------------------------------------------------------------------
# _cooccurrence_relationships
# ----------------------------------------------------------------------------

def test_cooccurrence_no_pairs():
    rels = _cooccurrence_relationships([EntityDraft("a", "PERSON", 0.9)])
    assert rels == []


def test_cooccurrence_pairwise_dedup():
    a = EntityDraft("Apple", "ORG", 0.9)
    b = EntityDraft("Alice", "PERSON", 0.85)
    rels = _cooccurrence_relationships([a, b, a])  # duplicate on purpose
    assert len(rels) == 1
    assert rels[0].from_text in ("Apple", "Alice")
    assert rels[0].to_text in ("Apple", "Alice")


def test_cooccurrence_caps_at_20():
    entities = [EntityDraft(f"E{i}", "PERSON", 0.9) for i in range(20)]
    rels = _cooccurrence_relationships(entities)
    assert len(rels) == 20


def test_cooccurrence_skips_self_loops():
    a = EntityDraft("Alice", "PERSON", 0.9)
    rels = _cooccurrence_relationships([a, a])
    assert rels == []


# ----------------------------------------------------------------------------
# enrich_chunks (with mocked spaCy)
# ----------------------------------------------------------------------------

class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNlp:
    """Stub for spaCy Language."""
    def __init__(self, docs_by_text: dict[str, list[_FakeSpan]]):
        self._docs = docs_by_text
    def __call__(self, text: str):
        spans = self._docs.get(text, [])
        return _FakeDoc(spans)


@pytest.fixture
def stub_spacy(monkeypatch):
    """Patch the cache + the lazy-loading helper so spaCy isn't loaded."""
    import uir_pipeline.enrich as enrich_mod
    # Pre-populate the cache so enrich._get_nlp doesn't import spacy.
    nlp = _FakeNlp({})
    enrich_mod._NLP_CACHE[DEFAULT_SPACY_MODEL] = nlp
    yield nlp
    enrich_mod._NLP_CACHE.clear()


def test_enrich_chunks_empty_input_returns_empty_result(stub_spacy):
    res = enrich_chunks([])
    assert res == EnrichmentResult(entities=[], relationships=[], topics=[])


def test_enrich_chunks_skips_empty_strings(stub_spacy):
    res = enrich_chunks(["", "   ", "\n"])
    assert res.entities == []


def test_enrich_chunks_returns_entities_per_chunk(stub_spacy):
    stub_spacy._docs["Alice works at Acme Corp in Paris."] = [
        _FakeSpan("Alice", "PERSON"),
        _FakeSpan("Acme Corp", "ORG"),
        _FakeSpan("Paris", "GPE"),
    ]
    res = enrich_chunks(["Alice works at Acme Corp in Paris."])
    texts = [e.text for e in res.entities]
    assert "Alice" in texts
    assert "Acme Corp" in texts
    assert "Paris" in texts


def test_enrich_chunks_dedupes_entities_across_chunks(stub_spacy):
    stub_spacy._docs["chunk A"] = [_FakeSpan("Alice", "PERSON")]
    stub_spacy._docs["chunk B"] = [
        _FakeSpan("ALICE", "PERSON"),  # lowercase-keyed dedup
        _FakeSpan("Apple", "ORG"),
    ]
    res = enrich_chunks(["chunk A", "chunk B"])
    persons = [e for e in res.entities if e.type == "PERSON"]
    assert len(persons) == 1


def test_enrich_chunks_emits_cooccurrence_relationships(stub_spacy):
    stub_spacy._docs["x"] = [
        _FakeSpan("Alice", "PERSON"),
        _FakeSpan("Acme", "ORG"),
    ]
    res = enrich_chunks(["x"])
    assert len(res.relationships) == 1
    assert res.relationships[0].type == "co-occurrence"


def test_enrich_chunks_topics_stub_returns_empty(stub_spacy):
    """Topics are out of MVP scope per PLAN.md \u00a73 -- always []."""
    res = enrich_chunks(["x" ])
    assert res.topics == []
