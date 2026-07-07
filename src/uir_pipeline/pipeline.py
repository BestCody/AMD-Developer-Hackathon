"""pipeline -- programmatic orchestrator (Phase L).

PLAN.md \u00a79 Phase L exit:
    -- chain ingest -> ocr -> layout -> tables -> chunk -> enrich -> embed -> assemble
    -- provenance block populated with model name, version, and ISO timestamp
    -- emits a single ``UIRV1`` JSON per document
    -- serial processing is fine for MVP

The orchestrator's fast path uses **pdfplumber text extraction** as a
stand-in for the per-page OCR step. This keeps the MVP smoke test fast
(<30s on a single PDF) without forcing a 100MB+ EasyOCR model download
on the test machine. Real-OCR is a one-line swap behind the
``_get_page_text`` indirection.

Weaviate upsert is optional via ``skip_weaviate=True``. When enabled, the
orchestrator (a) ensures both ``UIRChunks_v1`` and ``UIRParentDoc_v1``
collections exist, (b) writes one row per chunk with the prefixed UIR id
stored as a BM25 property, and (c) writes the document-level mean-pool
aggregate to the parent collection.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Public result
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineResult:
    """Per-document pipeline outcome."""
    uir_id: str
    out_path: Path
    chunk_count: int
    entity_count: int
    elapsed_seconds: float


# ----------------------------------------------------------------------------
# Page-text extraction (fast path: pdfplumber; real OCR is a one-line swap)
# ----------------------------------------------------------------------------

def _get_page_text(pdf_path: Path, page_numbers: list[int] | None = None) -> list[tuple[int, str]]:
    """Return a list of ``(page_number, text)`` from ``pdf_path``.

    ``page_numbers`` is 1-based; ``None`` means "all pages". The text is
    pdfplumber's per-page extract_text() output (string). Returns empty
    strings for image-only pages (which is fine for MVP -- the orchestrator
    treats empty pages as no-op).
    """
    import pdfplumber  # lazy
    out: list[tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_numbers is None:
            page_numbers = list(range(1, len(pdf.pages) + 1))
        for pn in page_numbers:
            if not (1 <= pn <= len(pdf.pages)):
                continue
            page = pdf.pages[pn - 1]
            out.append((pn, page.extract_text() or ""))
    return out


def _synthesize_words_for_text(text: str, page: int) -> tuple:
    """Build a per-word token list from ``text`` for one page.

    Each "word" is a simple ``DetectedWord`` with a synthetic 0-1000 bbox
    (since pdfplumber text extraction doesn't carry geometry). The page
    bbox is the full canvas; the LayoutClassifier will sub-cluster by
    y-proximity regardless.
    """
    from uir_pipeline.ocr import DetectedWord
    words: list[DetectedWord] = []
    if not text or not text.strip():
        return ()
    for tok in text.split():
        # Bbox is the full canvas; per-word coords are uniform.
        words.append(DetectedWord(
            text=tok,
            confidence=1.0,
            bbox=(0, 0, 1000, 1000),
            page=page,
        ))
    return tuple(words)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def run(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    skip_weaviate: bool = False,
    dry_run: bool = False,
    with_embeddings: bool = True,
    page_numbers: list[int] | None = None,
    on_progress: Any | None = None,
) -> PipelineResult:
    """Drive the full pipeline on one PDF and return a :class:`PipelineResult`.

    Parameters:
        input_path: PDF file path.
        output_dir: Where to write ``{uir_id}.uir.json``.
        skip_weaviate: True -> don't upsert to Weaviate (default: false).
        dry_run: True -> don't write JSON or Weaviate (default: false).
        with_embeddings: True -> compute BGE embeddings (default: true).
            False -> skip the embed step (faster, useful for tests).
        page_numbers: 1-based list of pages to process (``None`` = all).
        on_progress: optional callback ``fn(stage: str, percent: int)``.

    Returns a :class:`PipelineResult` with the uir id, output path, and
    counts of chunks + entities. Raises on ingest failure; logs and
    continues on per-stage failures (the resulting UIR may be partial).
    """
    t0 = time.monotonic()
    p = Path(input_path)
    output_dir = Path(output_dir)

    from uir_pipeline.chunk import chunk_text
    from uir_pipeline.embed import (
        COLLECTION_CHUNKS,
        COLLECTION_PARENT_DOCS,
        derive_doc_id,
        embed_texts,
        ensure_collections,
        mean_pool_vectors,
        upsert_chunks,
        upsert_parent_doc,
    )
    from uir_pipeline.enrich import enrich_chunks
    from uir_pipeline.ingest import DocumentInput, ingest
    from uir_pipeline.layout import LayoutClassifier
    from uir_pipeline.logging_config import (
        attach_doc_log,
        configure,
        detach_doc_log,
    )
    from uir_pipeline.ocr import OCRPage
    from uir_pipeline.tables import extract_tables
    from uir_pipeline.uir_schema import (
        ChunkNode,
        Entity,
        ExtractionProvenance,
        Metadata,
        NormalizationProvenance,
        Provenance,
        Relationship,
        Semantics,
        Source,
        Structure,
        StructureNode,
        UIRV1,
    )
    from uir_pipeline.utils import (
        bbox_from_pixel,
        deterministic_node_id,
        strip_uir_prefix,
    )

    configure()

    # Stage 1: ingest
    def _progress(stage: str, pct: int) -> None:
        logger.info("pipeline.stage %s (%d%%)", stage, pct)
        if on_progress is not None:
            try:
                on_progress(stage, pct)
            except Exception:
                pass
    _progress("ingest", 5)
    doc: DocumentInput = ingest(p)
    doc_id = derive_doc_id(doc.uri)
    log_dir = output_dir.parent / "logs" if output_dir.name != "logs" else output_dir
    log_handler = attach_doc_log(strip_uir_prefix(doc_id), log_dir)
    try:
        logger.info("ingested %s: %d pages, sha256=%s", doc.uri, doc.page_count, doc.sha256[:12])

        # Stage 2: extract per-page text (pdfplumber fast path)
        _progress("extract_text", 20)
        page_text_pairs = _get_page_text(p, page_numbers=page_numbers)

        # Stage 3: synthesize per-page OCRPage (one DetectedWord per text token)
        _progress("synthesize_ocr", 30)
        ocr_pages: list[OCRPage] = []
        for pn, text in page_text_pairs:
            words = _synthesize_words_for_text(text, pn)
            ocr_pages.append(OCRPage(page_number=pn, words=words))

        # Stage 4: heuristic layout classification
        _progress("layout", 45)
        layout = LayoutClassifier()
        all_regions = []
        for op in ocr_pages:
            all_regions.extend(layout.classify(op, page_height_px=792))

        # Stage 5: pdfplumber tables
        _progress("tables", 55)
        try:
            table_drafts = extract_tables(p, page_numbers=page_numbers)
        except Exception as exc:
            logger.warning("tables extraction failed: %s", exc)
            table_drafts = []

        # Stage 6: chunking -- union of layout regions + table markdown
        _progress("chunk", 70)
        all_chunks: list[Any] = []
        for region in all_regions:
            all_chunks.extend(chunk_text(
                region.text,
                page=region.page,
                bbox=region.bbox,
            ))
        for table in table_drafts:
            all_chunks.extend(chunk_text(
                table.markdown,
                page=table.page_number,
                bbox=table.bbox,
            ))
        if not all_chunks and page_text_pairs:
            # No regions / no tables -- chunk the whole document text.
            full_text = " ".join(text for _, text in page_text_pairs if text)
            all_chunks = chunk_text(full_text, page=1)

        # Stage 7: enrich (NER + co-occurrence)
        _progress("enrich", 80)
        enrichment = enrich_chunks([c.text for c in all_chunks])

        # Stage 8: embed (BGE-small 384-d)
        _progress("embed", 90)
        if with_embeddings and all_chunks:
            try:
                vectors = embed_texts([c.text for c in all_chunks])
            except Exception as exc:
                logger.warning("embed failed (%s) -- writing chunks without vectors", exc)
                vectors = None
        else:
            vectors = None

        # Stage 9: assemble UIRV1
        _progress("assemble", 95)
        source, metadata = doc.to_uir_source_metadata()
        # Override the page_count to match what ingest saw.
        metadata = metadata.model_copy(update={"page_count": doc.page_count})

        # Build chunk nodes
        chunk_nodes: list[ChunkNode] = []
        for i, ck in enumerate(all_chunks):
            ck_id = deterministic_node_id("chunk", doc_id, i, ck.text[:64])
            modal_features = dict(ck.modal_features) if ck.modal_features else {}
            if vectors is not None and i < len(vectors.vectors):
                modal_features["vector"] = {
                    "dim": vectors.dim,
                    "model": "BAAI/bge-small-en-v1.5",
                    "chunk_index": i,
                }
            chunk_nodes.append(ChunkNode(
                id=ck_id,
                type="chunk",
                text=ck.text,
                token_count=ck.token_count,
                page=ck.page,
                bounding_box=ck.bbox,
                confidence=ck.confidence,
                modal_features=modal_features,
            ))

        # Build entity records (UIR v1 doesn't carry per-entity id; the
        # orchestrator keeps the index-based list).
        entities: list[Entity] = [
            Entity(text=e.text, type=e.type, confidence=e.confidence)
            for e in enrichment.entities
        ]
        relationships: list[Relationship] = [
            Relationship(**{"from": r.from_text},
                          to=r.to_text, type=r.type, confidence=r.confidence)
            for r in enrichment.relationships
        ]

        now = datetime.now(timezone.utc)
        provenance = Provenance(
            extraction=ExtractionProvenance(
                model="LayoutLMv3-heuristic",
                version="1.0",
                timestamp=now,
            ),
            normalization=NormalizationProvenance(
                version="1.0",
                timestamp=now,
            ),
        )

        root = StructureNode(
            id=doc_id,
            type="document",
            title=metadata.title,
            page=1,
            children=chunk_nodes,
        )
        uir = UIRV1(
            uiR_version="1.0",
            id=doc_id,
            modal_type="document",
            source=source,
            metadata=metadata,
            structure=Structure(type="hierarchical", root=root),
            semantics=Semantics(
                entities=entities,
                relationships=relationships,
                topics=enrichment.topics,
            ),
            provenance=provenance,
        )

        # Stage 10: write JSON
        out_dir = output_dir
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{doc_id}.uir.json"
            out_path.write_text(uir.model_dump_json(indent=2))
        else:
            out_path = out_dir / f"{doc_id}.uir.json"  # virtual

        # Stage 11: optional Weaviate upsert
        if not skip_weaviate and not dry_run and vectors is not None and all_chunks:
            try:
                from uir_pipeline.weaviate_store import get_client
                client = get_client()
                ensure_collections(client)
                upsert_chunks(client, doc_id, [
                    {
                        "uir_id": cn.id,
                        "text": cn.text,
                        "page": cn.page,
                        "chunk_index": i,
                        "vector": vectors.vectors[i],
                    }
                    for i, cn in enumerate(chunk_nodes)
                ])
                upsert_parent_doc(
                    client, doc_id, mean_pool_vectors(vectors.vectors),
                    extra={"page_count": doc.page_count, "chunk_count": len(chunk_nodes)},
                )
                logger.info("weaviate upsert: %d chunks + 1 doc", len(chunk_nodes))
            except Exception as exc:
                logger.warning("weaviate upsert failed: %s", exc)

        _progress("done", 100)
        elapsed = time.monotonic() - t0
        return PipelineResult(
            uir_id=doc_id,
            out_path=out_path,
            chunk_count=len(chunk_nodes),
            entity_count=len(entities),
            elapsed_seconds=round(elapsed, 3),
        )
    finally:
        detach_doc_log(log_handler)


# Re-exports for callers that prefer ``pipeline.derive_doc_id``-style imports.
__all__ = [
    "PipelineResult",
    "run",
]
