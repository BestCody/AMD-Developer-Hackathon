"""tests/test_embed_weaviate_live.py -- live coverage for the Weaviate upsert path.

Before this module existed, ``embed.ensure_collections`` /
``embed.upsert_chunks`` / ``embed.upsert_parent_doc`` had **no test that ever
reached a real server** -- and all three were broken:

    -- ``_ensure_collection`` passed raw dicts as ``properties``; the v4
       client reads ``prop.textAnalyzer`` off each entry, so the very first
       call raised ``AttributeError``.
    -- both collections were created with the *chunk* schema, so the parent
       doc's ``page_count`` / ``chunk_count`` were undeclared.
    -- ``upsert_parent_doc`` used insert-only ``data.insert``, which raises
       on a repeat uuid -- i.e. on every re-ingest of the same document.

``pipeline.run`` swallowed the resulting exception as a ``logger.warning``,
so a full CLI run reported success while storing nothing.

Run with a server up::

    docker compose up -d
    python -m pytest tests/test_embed_weaviate_live.py -v

Skipped (not failed) when Weaviate is offline, exactly like
``tests/test_weaviate_store.py``.
"""
from __future__ import annotations

import uuid as uuid_pkg

import pytest

from uir_pipeline import embed
from uir_pipeline.utils import deterministic_node_id, strip_uir_prefix
from uir_pipeline.weaviate_store import reachable

DIM = 384


def _weaviate_reachable() -> bool:
    try:
        return bool(reachable())
    except Exception:
        return False


_LIVE: bool = _weaviate_reachable()
_SKIP = pytest.mark.skipif(
    not _LIVE, reason="Weaviate not reachable (start with `docker compose up -d`)"
)

pytestmark = _SKIP


@pytest.fixture()
def client():
    from uir_pipeline.weaviate_store import get_client

    c = get_client()
    try:
        yield c
    finally:
        c.close()


@pytest.fixture()
def collections(client, monkeypatch):
    """Point the module constants at run-stamped collections.

    The helpers read ``COLLECTION_CHUNKS`` / ``COLLECTION_PARENT_DOCS`` as
    module globals at call time, so patching them redirects every write.
    Without this the tests would write into -- and delete -- a developer's
    real ``UIRChunks_v1``.
    """
    stamp = uuid_pkg.uuid4().hex[:8]
    chunks = f"TestChunks_{stamp}"
    parents = f"TestParentDoc_{stamp}"
    monkeypatch.setattr(embed, "COLLECTION_CHUNKS", chunks)
    monkeypatch.setattr(embed, "COLLECTION_PARENT_DOCS", parents)
    try:
        yield chunks, parents
    finally:
        for name in (chunks, parents):
            try:
                if client.collections.exists(name):
                    client.collections.delete(name)
            except Exception:
                pass


def _records(n: int, *, dim: int = DIM) -> list[dict]:
    return [
        {
            "uir_id": deterministic_node_id("chunk", f"seed-{i}"),
            "text": f"chunk text {i}",
            "page": i,
            "chunk_index": i,
            "vector": [0.1 * (i + 1)] * dim,
        }
        for i in range(n)
    ]


def _count(client, name: str) -> int:
    return client.collections.get(name).aggregate.over_all(total_count=True).total_count


class TestEnsureCollections:
    def test_creates_both_collections(self, client, collections):
        chunks, parents = collections
        embed.ensure_collections(client)
        assert client.collections.exists(chunks)
        assert client.collections.exists(parents)

    def test_is_idempotent(self, client, collections):
        embed.ensure_collections(client)
        embed.ensure_collections(client)  # must not raise

    def test_parent_doc_schema_declares_its_own_properties(self, client, collections):
        """Regression: both collections used to share the chunk schema."""
        _chunks, parents = collections
        embed.ensure_collections(client)
        props = {p.name for p in client.collections.get(parents).config.get().properties}
        assert {"uir_id", "page_count", "chunk_count"} <= props
        assert "text_preview" not in props, "parent doc must not carry the chunk schema"

    def test_chunk_schema_declares_its_own_properties(self, client, collections):
        chunks, _parents = collections
        embed.ensure_collections(client)
        props = {p.name for p in client.collections.get(chunks).config.get().properties}
        assert {"uir_id", "doc_id", "page", "chunk_index", "text_preview"} <= props


class TestUpsertChunks:
    def test_writes_records_and_reports_true_count(self, client, collections):
        chunks, _ = collections
        doc_id = deterministic_node_id("doc", "file:///a.pdf")
        written = embed.upsert_chunks(client, doc_id, _records(3))
        assert written == 3
        assert _count(client, chunks) == 3, "reported count must match stored objects"

    def test_is_idempotent_by_uuid(self, client, collections):
        """Re-upserting identical ids overwrites rather than duplicating."""
        chunks, _ = collections
        doc_id = deterministic_node_id("doc", "file:///a.pdf")
        recs = _records(3)
        embed.upsert_chunks(client, doc_id, recs)
        embed.upsert_chunks(client, doc_id, recs)
        assert _count(client, chunks) == 3

    def test_stores_properties_and_vector(self, client, collections):
        chunks, _ = collections
        doc_id = deterministic_node_id("doc", "file:///a.pdf")
        recs = _records(1)
        embed.upsert_chunks(client, doc_id, recs)
        obj = next(iter(client.collections.get(chunks).iterator(include_vector=True)))
        assert obj.properties["uir_id"] == recs[0]["uir_id"]
        assert obj.properties["doc_id"] == doc_id
        assert obj.properties["text_preview"] == "chunk text 0"
        vec = obj.vector["default"] if isinstance(obj.vector, dict) else obj.vector
        assert len(vec) == DIM

    def test_truncates_text_preview_to_256_chars(self, client, collections):
        chunks, _ = collections
        rec = _records(1)[0]
        rec["text"] = "x" * 500
        embed.upsert_chunks(client, deterministic_node_id("doc", "file:///a.pdf"), [rec])
        obj = next(iter(client.collections.get(chunks).iterator()))
        assert len(obj.properties["text_preview"]) == 256

    def test_raises_when_server_rejects_an_object(self, client, collections):
        """A rejected object must not be counted as written.

        The batch API accumulates per-object rejections on
        ``failed_objects`` instead of raising, so the old code returned
        ``len(chunk_records)`` for a batch the server had thrown away.
        A vector-dimension mismatch is the cheapest way to force one.
        """
        doc_id = deterministic_node_id("doc", "file:///a.pdf")
        embed.upsert_chunks(client, doc_id, _records(1))  # fixes dim at 384
        bad = _records(2, dim=DIM // 2)
        with pytest.raises(RuntimeError, match="rejected"):
            embed.upsert_chunks(client, doc_id, bad)


class TestUpsertParentDoc:
    def test_writes_the_aggregate(self, client, collections):
        _, parents = collections
        doc_id = deterministic_node_id("doc", "file:///a.pdf")
        embed.upsert_parent_doc(
            client, doc_id, [0.5] * DIM, extra={"page_count": 3, "chunk_count": 7}
        )
        obj = next(iter(client.collections.get(parents).iterator()))
        assert obj.properties["uir_id"] == doc_id
        assert obj.properties["page_count"] == 3
        assert obj.properties["chunk_count"] == 7

    def test_re_ingest_does_not_raise_and_updates_in_place(self, client, collections):
        """Regression: ``data.insert`` raised on the second ingest of a doc."""
        _, parents = collections
        doc_id = deterministic_node_id("doc", "file:///a.pdf")
        embed.upsert_parent_doc(
            client, doc_id, [0.5] * DIM, extra={"page_count": 3, "chunk_count": 7}
        )
        embed.upsert_parent_doc(
            client, doc_id, [0.9] * DIM, extra={"page_count": 4, "chunk_count": 9}
        )
        assert _count(client, parents) == 1, "re-ingest must overwrite, not duplicate"
        obj = next(iter(client.collections.get(parents).iterator()))
        assert obj.properties["page_count"] == 4
        assert obj.properties["chunk_count"] == 9

    def test_uuid_is_the_stripped_uir_id(self, client, collections):
        _, parents = collections
        doc_id = deterministic_node_id("doc", "file:///a.pdf")
        embed.upsert_parent_doc(client, doc_id, [0.5] * DIM, extra={})
        coll = client.collections.get(parents)
        assert coll.data.exists(strip_uir_prefix(doc_id))

    def test_defaults_counts_to_zero_when_extra_omitted(self, client, collections):
        _, parents = collections
        doc_id = deterministic_node_id("doc", "file:///a.pdf")
        embed.upsert_parent_doc(client, doc_id, [0.5] * DIM)
        obj = next(iter(client.collections.get(parents).iterator()))
        assert obj.properties["page_count"] == 0
        assert obj.properties["chunk_count"] == 0
