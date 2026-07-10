"""embed -- BGE embeddings + Weaviate upsert (Phase K).

PLAN.md \u00a79 Phase K exit:
    -- 384-d chunk embeddings via sentence-transformers (BAAI/bge-small-en-v1.5)
    -- Weaviate ``UIRChunks_v1`` collection created/verified on first run
    -- per-chunk upsert with UIR metadata blob
    -- document-level aggregate (mean pool) into ``UIRParentDoc_v1``

ID-mapping rule (PLAN.md \u00a79 Phase K + \u00a715 decision log):
    -- ``UIR.ID`` shape is ``<prefix>_<uuid5>``; Weaviate's primary node id
       requires a bare UUID. We strip the prefix to form the Weaviate id
       and store the full prefixed UIR id as a BM25-indexed ``uir_id``.
    -- Identical convention for the parent doc collection.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from uir_pipeline.utils import (
    DEFAULT_BGE_MODEL,
    deterministic_node_id,
    strip_uir_prefix,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Weaviate collection names per PLAN.md \u00a79 Phase K + \u00a715 decision log.
COLLECTION_CHUNKS: str = "UIRChunks_v1"
COLLECTION_PARENT_DOCS: str = "UIRParentDoc_v1"


# ----------------------------------------------------------------------------
# Lazy singleton for the sentence-transformers model
# ----------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _get_model(model_id: str = DEFAULT_BGE_MODEL):
    """Lazy-load the sentence-transformers model and cache per ``model_id``."""
    cached = _MODEL_CACHE.get(model_id)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_id)
        if cached is not None:
            return cached
        from sentence_transformers import SentenceTransformer  # lazy
        logger.debug("loading sentence-transformers model %s (cold cache)", model_id)
        m = SentenceTransformer(model_id)
        _MODEL_CACHE[model_id] = m
        return m


# ----------------------------------------------------------------------------
# Public result types
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingOutput:
    """The embeddings result for a single chunk list."""
    vectors: list[list[float]]  # N x D, D = model_dim
    dim: int


# ----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    *,
    model_id: str = DEFAULT_BGE_MODEL,
    batch_size: int = 32,
    normalize: bool = True,
) -> EmbeddingOutput:
    """Compute sentence embeddings for ``texts`` (lazy-loads BGE on first call).

    ``normalize=True`` returns L2-normalized vectors, which is the
    recommended shape for cosine-similarity retrieval at Weaviate time.
    """
    if not texts:
        return EmbeddingOutput(vectors=[], dim=0)

    model = _get_model(model_id)
    # BGE requires a query prefix; we apply it to every text uniformly
    # (the chunker doesn't separate query-vs-corpus at MVP).
    prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    raw = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    # ``raw`` is a numpy array of shape (N, D).
    vectors = raw.tolist()
    return EmbeddingOutput(vectors=vectors, dim=len(vectors[0]) if vectors else 0)


# ----------------------------------------------------------------------------
# Weaviate upsert helpers
# ----------------------------------------------------------------------------

def _collection_properties(name: str) -> list[Any]:
    """Return the ``Property`` schema for ``name``.

    The two collections store different shapes and must not share a schema:
    chunks are per-page fragments, the parent doc is a per-document
    aggregate. Writing a property that isn't declared here relies on
    Weaviate's auto-schema, which we don't want to depend on.

    ``Property`` objects (not raw dicts) are required: the v4 client reads
    ``prop.textAnalyzer`` off each entry during ``collections.create``.
    """
    from weaviate.classes.config import DataType, Property

    if name == COLLECTION_PARENT_DOCS:
        return [
            Property(name="uir_id", data_type=DataType.TEXT),
            Property(name="page_count", data_type=DataType.INT),
            Property(name="chunk_count", data_type=DataType.INT),
        ]
    return [
        Property(name="uir_id", data_type=DataType.TEXT),
        Property(name="doc_id", data_type=DataType.TEXT),
        Property(name="page", data_type=DataType.INT),
        Property(name="chunk_index", data_type=DataType.INT),
        Property(name="text_preview", data_type=DataType.TEXT),
    ]


def _ensure_collection(client: Any, name: str) -> None:
    """Idempotent: create ``name`` if it doesn't exist."""
    if client.collections.exists(name):
        return
    # We BYO vectors (sentence-transformers), so the server must not try to
    # vectorize. ``Vectorizer.none()`` rather than the newer
    # ``Vectors.self_provided()`` because requirements.txt pins only
    # ``weaviate-client>=4.5`` and the latter lands in a later 4.x.
    from weaviate.classes.config import Configure

    client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=_collection_properties(name),
    )


def ensure_collections(client: Any) -> None:
    """Idempotently create the UIRChunks_v1 + UIRParentDoc_v1 collections."""
    _ensure_collection(client, COLLECTION_CHUNKS)
    _ensure_collection(client, COLLECTION_PARENT_DOCS)


def upsert_chunks(
    client: Any,
    doc_id: str,
    chunk_records: list[dict[str, Any]],
) -> int:
    """Upsert ``chunk_records`` to ``UIRChunks_v1``.

    Each record must carry: ``uir_id``, ``text``, ``page``, ``chunk_index``,
    and ``vector`` (list[float]). ``text_preview`` is truncated to 256 chars
    so the property doesn't dump entire chunks.

    Returns the number of records written. Weaviate's batch insert is
    upsert-by-uuid (UUID is the Weaviate primary id; ``uir_id`` is a
    BM25-indexed property used by retrieval-time queries).

    Raises ``RuntimeError`` if the server rejected any object. The batch
    API never raises on a per-object rejection -- it accumulates them on
    ``batch.failed_objects`` -- so counting the ``add_object`` calls would
    report chunks as stored that are in fact absent and unretrievable.
    """
    _ensure_collection(client, COLLECTION_CHUNKS)
    coll = client.collections.get(COLLECTION_CHUNKS)
    submitted = 0
    with coll.batch.dynamic() as batch:
        for rec in chunk_records:
            uuid_stripped = strip_uir_prefix(rec["uir_id"])
            preview = (rec["text"] or "")[:256]
            batch.add_object(
                properties={
                    "uir_id": rec["uir_id"],
                    "doc_id": doc_id,
                    "page": int(rec["page"]),
                    "chunk_index": int(rec["chunk_index"]),
                    "text_preview": preview,
                },
                uuid=uuid_stripped,
                vector=rec["vector"],
            )
            submitted += 1

    failed = list(coll.batch.failed_objects or ())
    if failed:
        sample = "; ".join(str(f.message) for f in failed[:3])
        raise RuntimeError(
            f"weaviate rejected {len(failed)}/{submitted} chunk objects "
            f"for doc {doc_id}: {sample}"
        )
    return submitted


def upsert_parent_doc(
    client: Any,
    doc_id_uir: str,
    mean_vector: list[float],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Upsert the document-level aggregate to ``UIRParentDoc_v1``.

    ``data.insert`` is insert-only -- it raises ``ObjectAlreadyExists`` on a
    repeat uuid. Re-ingesting a document is routine (the doc id is derived
    deterministically from the source URI), so we replace when the object is
    already present. That makes this genuinely idempotent, matching
    ``upsert_chunks``.
    """
    _ensure_collection(client, COLLECTION_PARENT_DOCS)
    coll = client.collections.get(COLLECTION_PARENT_DOCS)
    extra = extra or {}
    uuid = strip_uir_prefix(doc_id_uir)
    properties = {
        "uir_id": doc_id_uir,
        "page_count": int(extra.get("page_count", 0)),
        "chunk_count": int(extra.get("chunk_count", 0)),
    }
    if coll.data.exists(uuid):
        coll.data.replace(uuid=uuid, properties=properties, vector=mean_vector)
    else:
        coll.data.insert(uuid=uuid, properties=properties, vector=mean_vector)


def mean_pool_vectors(vectors: list[list[float]]) -> list[float]:
    """Mean-pool ``vectors`` element-wise; returns a single vector or empty list."""
    if not vectors:
        return []
    n = len(vectors[0])
    sums = [0.0] * n
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [round(s / len(vectors), 6) for s in sums]


def derive_doc_id(uri: str) -> str:
    """Compute the UIR doc-level id from a stable seed (the source URI)."""
    return deterministic_node_id("doc", uri)


# Re-export for callers that prefer `embed.derive_doc_id`.
__all__ = [
    "COLLECTION_CHUNKS",
    "COLLECTION_PARENT_DOCS",
    "EmbeddingOutput",
    "derive_doc_id",
    "embed_texts",
    "ensure_collections",
    "mean_pool_vectors",
    "upsert_chunks",
    "upsert_parent_doc",
]
