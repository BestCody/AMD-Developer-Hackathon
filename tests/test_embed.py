"""tests/test_embed.py -- Phase K embed module tests.

We mock ``sentence_transformers.SentenceTransformer`` so the unit tests
don't pay the 133 MB cold-load. The fake returns a numpy ndarray so
production code's ``raw.tolist()`` works without bypass.

Weaviate is also mocked (``collections.exists``, ``collections.create``,
``collections.get(...).batch.dynamic()``, ``coll.data.insert``) since
unit tests run without a live cluster.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from uir_pipeline.embed import (
    COLLECTION_CHUNKS,
    COLLECTION_PARENT_DOCS,
    EmbeddingOutput,
    derive_doc_id,
    embed_texts,
    ensure_collections,
    mean_pool_vectors,
    upsert_chunks,
    upsert_parent_doc,
)
from uir_pipeline.utils import DEFAULT_BGE_MODEL, strip_uir_prefix


# ----------------------------------------------------------------------------
# Fake stub sentence-transformers model (returns numpy.ndarray)
# ----------------------------------------------------------------------------

class _FakeSentenceTransformer:
    """Returns (N, 384) numpy arrays whose rows are deterministic seeded-noise.

    The fake mirrors the real ``SentenceTransformer.encode(..., convert_to_
    numpy=True)`` contract so ``raw.tolist()`` works without bypass.
    """

    dim = 384  # matches BAAI/bge-small-en-v1.5

    def __init__(self, model_id=None):
        self.last_texts: list[str] = []

    def encode(self, texts, **kwargs):
        self.last_texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = sum(ord(c) for c in t)
            for j in range(self.dim):
                out[i, j] = round(math.sin((h + j) * 0.0001), 6)
        return out


@pytest.fixture
def stub_sentence_transformer(monkeypatch):
    """Install a fake ST model into the module-level cache."""
    import uir_pipeline.embed as embed_mod
    fake = _FakeSentenceTransformer()
    embed_mod._MODEL_CACHE[DEFAULT_BGE_MODEL] = fake
    yield fake
    embed_mod._MODEL_CACHE.clear()


# ----------------------------------------------------------------------------
# embed_texts
# ----------------------------------------------------------------------------

def test_embed_texts_empty_returns_empty(stub_sentence_transformer):
    out = embed_texts([])
    assert out.vectors == []
    assert out.dim == 0


def test_embed_texts_returns_384_dim_vectors(stub_sentence_transformer):
    out = embed_texts(["Hello world", "Goodbye world"])
    assert isinstance(out, EmbeddingOutput)
    assert out.dim == 384
    assert len(out.vectors) == 2
    for v in out.vectors:
        assert len(v) == 384


def test_embed_texts_applies_bge_query_prefix(stub_sentence_transformer):
    """BGE-small-en-v1.5 expects the query prefix on encode inputs."""
    stub_sentence_transformer.encode.__self__.last_texts = []  # reset
    embed_texts(["apple"])
    # The fake's encode() received the BGE-required prefix.
    received = stub_sentence_transformer.last_texts
    assert received and received[0].startswith(
        "Represent this sentence for searching relevant passages:"
    )


def test_embed_texts_deterministic_for_same_input(stub_sentence_transformer):
    a = embed_texts(["apple"]).vectors[0]
    b = embed_texts(["apple"]).vectors[0]
    assert a == b


# ----------------------------------------------------------------------------
# mean_pool_vectors
# ----------------------------------------------------------------------------

def test_mean_pool_vectors_empty_returns_empty_list():
    assert mean_pool_vectors([]) == []


def test_mean_pool_vectors_single_row_returns_that_row():
    v = mean_pool_vectors([[0.0, 1.0, 2.0]])
    assert v == [0.0, 1.0, 2.0]


def test_mean_pool_vectors_pair_returns_element_means():
    v = mean_pool_vectors([[0.0, 2.0], [2.0, 4.0]])
    assert v == [1.0, 3.0]


# ----------------------------------------------------------------------------
# Weaviate stubs
# ----------------------------------------------------------------------------

class _FakeBatch:
    def __init__(self):
        self.objects: list[dict[str, Any]] = []
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def add_object(self, *, properties=None, uuid=None, vector=None):
        self.objects.append({"properties": properties, "uuid": uuid, "vector": vector})


class _BatchCtx:
    """Helper nested class reproducing weaviate's ``coll.batch.dynamic()`` ctx-mgr chain.

    ``failed_objects`` is part of the real contract: the batch API never
    raises on a per-object rejection, it accumulates them here. Omitting it
    from this stub is how ``upsert_chunks`` shipped a version that reported
    server-rejected objects as written.
    """
    def __init__(self, parent):
        self.parent = parent
        self._b = _FakeBatch()
    def __enter__(self):
        self.parent.batches.append(self._b)
        return self._b
    def __exit__(self, *a):
        return False
    def dynamic(self):
        return self
    @property
    def failed_objects(self):
        return self.parent.failed_objects


class _FakeDataAPI:
    """Stand-in for ``coll.data`` -- captures insert/replace kwargs.

    The real ``insert`` is insert-only and raises on a repeat uuid, so this
    stub enforces that too: ``upsert_parent_doc`` must reach for ``replace``
    when the object already exists.
    """
    def __init__(self, parent):
        self.parent = parent
    def exists(self, uuid):
        return uuid in self.parent.by_uuid
    def insert(self, *, uuid=None, properties=None, vector=None):
        if uuid in self.parent.by_uuid:
            raise AssertionError(f"insert() called on existing uuid {uuid!r}")
        rec = {"uuid": uuid, "properties": properties, "vector": vector}
        self.parent.inserts.append(rec)
        self.parent.by_uuid[uuid] = rec
    def replace(self, *, uuid=None, properties=None, vector=None):
        rec = {"uuid": uuid, "properties": properties, "vector": vector}
        self.parent.replaces.append(rec)
        self.parent.by_uuid[uuid] = rec


class _FakeCollection:
    def __init__(self):
        self.batches: list[_FakeBatch] = []
        self.inserts: list[dict[str, Any]] = []
        self.replaces: list[dict[str, Any]] = []
        self.by_uuid: dict[str, dict[str, Any]] = {}
        self.failed_objects: list[Any] = []
    @property
    def batch(self):
        return _BatchCtx(self)
    @property
    def data(self):
        return _FakeDataAPI(self)


class _FakeCollectionsAPI:
    def __init__(self):
        self._has: set[str] = set()
        self._defs: dict[str, _FakeCollection] = {}
        self.created_properties: dict[str, list[Any]] = {}
    def exists(self, name):
        return name in self._has
    def create(self, name, **kwargs):
        # The v4 client reads ``prop.textAnalyzer`` off every entry, so raw
        # dicts blow up with AttributeError. Reject them here so this stub
        # cannot green-light a schema the real server would never accept.
        props = list(kwargs.get("properties") or ())
        for p in props:
            if isinstance(p, dict):
                raise AttributeError("'dict' object has no attribute 'textAnalyzer'")
        self._has.add(name)
        self._defs[name] = _FakeCollection()
        self.created_properties[name] = props
    def get(self, name):
        return self._defs.setdefault(name, _FakeCollection())


class _FakeWeaviateClient:
    def __init__(self):
        self.collections = _FakeCollectionsAPI()


# ----------------------------------------------------------------------------
# ensure_collections
# ----------------------------------------------------------------------------

def test_ensure_collections_creates_both_collections():
    c = _FakeWeaviateClient()
    ensure_collections(c)
    assert COLLECTION_CHUNKS in c.collections._has
    assert COLLECTION_PARENT_DOCS in c.collections._has


def test_ensure_collections_idempotent_does_not_recreate():
    c = _FakeWeaviateClient()
    ensure_collections(c)
    chunk_oid_before = id(c.collections._defs[COLLECTION_CHUNKS])
    ensure_collections(c)
    chunk_oid_after = id(c.collections._defs[COLLECTION_CHUNKS])
    assert chunk_oid_before == chunk_oid_after


# ----------------------------------------------------------------------------
# upsert_chunks
# ----------------------------------------------------------------------------

def test_upsert_chunks_writes_per_record():
    c = _FakeWeaviateClient()
    records = [
        {"uir_id": "chunk_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "text": "Hello.",
         "page": 1, "chunk_index": 0, "vector": [0.1] * 384},
        {"uir_id": "chunk_bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
         "text": "World.", "page": 1, "chunk_index": 1, "vector": [0.2] * 384},
    ]
    n = upsert_chunks(c, doc_id="doc_xx", chunk_records=records)
    assert n == 2
    coll = c.collections.get(COLLECTION_CHUNKS)
    [batch] = coll.batches
    assert len(batch.objects) == 2


def test_upsert_chunks_strips_prefix_for_weaviate_uuid():
    c = _FakeWeaviateClient()
    uid = "chunk_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    records = [
        {"uir_id": uid, "text": "Hi.", "page": 1, "chunk_index": 0, "vector": [0.0] * 4},
    ]
    upsert_chunks(c, doc_id="doc_x", chunk_records=records)
    coll = c.collections.get(COLLECTION_CHUNKS)
    [batch] = coll.batches
    obj = batch.objects[0]
    assert obj["uuid"] == strip_uir_prefix(uid)
    # Full prefixed id stored on the BM25 property so retrieval can recover it.
    assert obj["properties"]["uir_id"] == uid


def test_upsert_chunks_truncates_text_preview_to_256():
    c = _FakeWeaviateClient()
    records = [
        {"uir_id": "chunk_aaaa", "text": "x" * 1024,
         "page": 1, "chunk_index": 0, "vector": [0.0] * 4},
    ]
    upsert_chunks(c, doc_id="doc_x", chunk_records=records)
    coll = c.collections.get(COLLECTION_CHUNKS)
    [batch] = coll.batches
    preview = batch.objects[0]["properties"]["text_preview"]
    assert len(preview) == 256


# ----------------------------------------------------------------------------
# upsert_parent_doc
# ----------------------------------------------------------------------------

def test_upsert_parent_doc_writes_aggregate_with_stripped_uuid():
    c = _FakeWeaviateClient()
    doc_id = derive_doc_id("file:///tmp/x.pdf")
    mean_v = [0.1, 0.2, 0.3]
    upsert_parent_doc(c, doc_id, mean_v,
                      extra={"page_count": 5, "chunk_count": 10})
    coll = c.collections.get(COLLECTION_PARENT_DOCS)
    assert len(coll.inserts) == 1
    ins = coll.inserts[0]
    assert ins["uuid"] == strip_uir_prefix(doc_id)
    assert ins["properties"]["uir_id"] == doc_id
    assert ins["properties"]["page_count"] == 5
    assert ins["properties"]["chunk_count"] == 10
    assert ins["vector"] == mean_v


def test_upsert_parent_doc_replaces_on_re_ingest():
    """Regression: ``data.insert`` is insert-only and raised on re-ingest.

    Doc ids are derived deterministically from the source URI, so ingesting
    the same file twice hits the same uuid -- the common case, not an edge.
    """
    c = _FakeWeaviateClient()
    doc_id = derive_doc_id("file:///tmp/x.pdf")
    upsert_parent_doc(c, doc_id, [0.1, 0.2], extra={"page_count": 1, "chunk_count": 2})
    upsert_parent_doc(c, doc_id, [0.9, 0.8], extra={"page_count": 3, "chunk_count": 4})
    coll = c.collections.get(COLLECTION_PARENT_DOCS)
    assert len(coll.inserts) == 1, "second ingest must not insert again"
    assert len(coll.replaces) == 1, "second ingest must replace"
    assert coll.by_uuid[strip_uir_prefix(doc_id)]["properties"]["page_count"] == 3


# ----------------------------------------------------------------------------
# upsert_chunks failure reporting
# ----------------------------------------------------------------------------

class _FailedObj:
    def __init__(self, message):
        self.message = message


def test_upsert_chunks_raises_when_server_rejects_objects():
    """Regression: rejected objects were counted as written.

    The batch API reports per-object rejections on ``failed_objects``
    rather than raising, so returning ``len(chunk_records)`` claimed chunks
    were stored that are absent and unretrievable.
    """
    c = _FakeWeaviateClient()
    # Create first: ``_ensure_collection`` would otherwise replace the
    # collection object and discard the staged failure.
    ensure_collections(c)
    coll = c.collections.get(COLLECTION_CHUNKS)
    coll.failed_objects = [_FailedObj("vector lengths don't match")]
    records = [
        {"uir_id": "chunk_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "text": "Hi.",
         "page": 1, "chunk_index": 0, "vector": [0.0] * 4},
    ]
    with pytest.raises(RuntimeError, match=r"rejected 1/1"):
        upsert_chunks(c, doc_id="doc_x", chunk_records=records)


def test_upsert_chunks_returns_count_when_nothing_failed():
    c = _FakeWeaviateClient()
    records = [
        {"uir_id": "chunk_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "text": "Hi.",
         "page": 1, "chunk_index": 0, "vector": [0.0] * 4},
    ]
    assert upsert_chunks(c, doc_id="doc_x", chunk_records=records) == 1


# ----------------------------------------------------------------------------
# collection schemas
# ----------------------------------------------------------------------------

def test_collections_do_not_share_a_schema():
    """Regression: both collections were created with the chunk schema, so
    the parent doc's page_count/chunk_count were never declared."""
    c = _FakeWeaviateClient()
    ensure_collections(c)
    chunk_props = {p.name for p in c.collections.created_properties[COLLECTION_CHUNKS]}
    parent_props = {p.name for p in c.collections.created_properties[COLLECTION_PARENT_DOCS]}
    assert "text_preview" in chunk_props
    assert "text_preview" not in parent_props
    assert {"page_count", "chunk_count"} <= parent_props
    assert {"page_count", "chunk_count"}.isdisjoint(chunk_props)


def test_ensure_collections_passes_property_objects_not_dicts():
    """Regression: raw dicts raised AttributeError inside the v4 client."""
    c = _FakeWeaviateClient()
    ensure_collections(c)  # the stub raises AttributeError on dict properties
    for name in (COLLECTION_CHUNKS, COLLECTION_PARENT_DOCS):
        for p in c.collections.created_properties[name]:
            assert not isinstance(p, dict)
            assert hasattr(p, "name")


# ----------------------------------------------------------------------------
# derive_doc_id
# ----------------------------------------------------------------------------

def test_derive_doc_id_stable_per_uri():
    a = derive_doc_id("file:///tmp/x.pdf")
    b = derive_doc_id("file:///tmp/x.pdf")
    assert a == b
    assert a.startswith("doc_")


def test_derive_doc_id_differs_for_diff_uris():
    a = derive_doc_id("file:///tmp/x.pdf")
    b = derive_doc_id("file:///tmp/y.pdf")
    assert a != b
