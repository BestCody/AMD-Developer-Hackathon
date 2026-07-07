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
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Boilerplate filter
# ----------------------------------------------------------------------------

# Patterns that surface repeatedly on ArXiv-style PDFs because pdfplumber
# concatenates narrow-kerned corporate tokens without explicit spaces.
# Apply post-NER; drop matching entities and any relationships whose
# endpoints reference a dropped entity. Tested against: "Attention Is All
# You Need" (1706.03762) where this filter dropped 21 entities and 60
# relationships from the 411/736 baseline (Tier 3 fix #2 follow-up).
_BOILERPLATE_RE: tuple[re.Pattern[str], ...] = (
    # The ArXiv permission block ("Google hereby grants permission to..."),
    # including the token-stripped variant pdfplumber emits. The trailing
    # "to" is suffixed via `(?:to)?` not a `\b` so the variant
    # ``"Googleherebygrantspermissionto"`` (pdfplumber concatenates without
    # spaces) still matches: ``\b`` between two word-class chars (``n`` /
    # ``t``) would not fire, so a naïve ``\b…\b`` pattern misses the case.
    re.compile(r"\bgoogle\s+hereby\s+grants?\s+permissions?\b", re.IGNORECASE),
    re.compile(r"\bgoogleherebygrantspermission(?:to)?\b", re.IGNORECASE),
    # Google Research / Brain affiliations repeated in headers/footers.
    re.compile(r"\bgoogle\s*brain\b", re.IGNORECASE),
    re.compile(r"\bgoogle\s*research\b", re.IGNORECASE),
    # Mountain View / corporate addr fragments.
    re.compile(r"\bmountain\s+view\b", re.IGNORECASE),
    # Standard copyright token clusters.
    re.compile(r"\bcopyright\s+\(c\)\s*\d{4}\b", re.IGNORECASE),
)


def _is_boilerplate(text: str) -> bool:
    """Return True iff ``text`` matches any :data:`_BOILERPLATE_RE` pattern.

    Used by :func:`run` post-enrichment to drop noisy generic entities that
    pdfplumber surfaces from ArXiv corporate footers/permission headers.
    """
    return any(p.search(text) for p in _BOILERPLATE_RE)


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


def _get_page_words_with_real_coords(
    pdf_path: Path,
    page_numbers: list[int] | None = None,
) -> list[tuple[int, tuple, str]]:
    """Return ``[(page_number, words_tuple, page_text), ...]`` for ``pdf_path``.

    Tier 1.5 #1: replaces the previous synthetic-bbox path. Each
    :class:`DetectedWord` carries its real pdfplumber bounding box
    ``(x0, top, x1, bottom)`` in PDF-point coordinates so the
    :class:`LayoutClassifier` sees real geometry and emits real
    ``heading`` / ``footer`` / ``paragraph`` labels instead of labeling
    every page as ``header``.

    Heavy pdfplumber open is done ONCE (single-context) so this is also
    faster than the old get_text + synthesize pattern (``pdfplumber.open``
    was invoked twice -- once in :func:`_get_page_text`, once here
    implicitly).

    The returned tuple list preserves 1-based ordering matching the
    orchestrator's page_numbers contract.
    """
    import pdfplumber  # lazy
    from uir_pipeline.ocr import DetectedWord
    out: list[tuple[int, tuple, str]] = []
    if not pdf_path.is_file():
        return out
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_numbers is None:
            page_numbers = list(range(1, len(pdf.pages) + 1))
        for pn in page_numbers:
            if not (1 <= pn <= len(pdf.pages)):
                continue
            page = pdf.pages[pn - 1]
            text = page.extract_text() or ""
            words_list: list[DetectedWord] = []
            # Filter rotated text via pdfplumber's ``upright`` attribute.
            # Arxiv's sideways ``viXra`` watermark and vertical figure-axis
            # labels both emit non-upright glyphs; without this filter they
            # concatenate into the chunk stream as reversed garbage
            # (e.g. ``3202 guA 2 ]LC.sc[ 7v26730.6071:viXra``) and as
            # single-character axis ticks (Fix Plan item #1). Older
            # pdfplumber releases omit ``upright`` entirely so we default
            # the check to ``True`` to preserve pre-fix behaviour for
            # those versions.
            for w in page.extract_words(extra_attrs=["upright"]):
                if not w.get("upright", True):
                    continue
                words_list.append(DetectedWord(
                    text=str(w.get("text", "")).strip(),
                    confidence=1.0,
                    bbox=(int(w["x0"]), int(w["top"]), int(w["x1"]), int(w["bottom"])),
                    page=pn,
                ))
            out.append((pn, tuple(words_list), text))
    return out


def _synthesize_words_for_text(text: str, page: int) -> tuple:
    """Fallback: synthesize one ``DetectedWord`` per whitespace token.

    Used when pdfplumber's ``page.extract_words()`` returns empty for a
    page (rare -- only on full-page image scans). Real word geometries
    are still preferred via :func:`_get_page_words_with_real_coords`;
    this helper only exists so a downstream caller can still construct a
    non-empty :class:`OCRPage` on degenerate input.
    """
    from uir_pipeline.ocr import DetectedWord
    words: list[DetectedWord] = []
    if not text or not text.strip():
        return ()
    for tok in text.split():
        # Synthetic full-canvas bbox -- coarse fall-back only.
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
    from uir_pipeline.enrich import EnrichmentResult, enrich_chunks
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
    def _progress(stage: str, pct: int, **meta: Any) -> None:
        """Emit a stage-progress event.

        ``meta`` is forwarded to ``on_progress(stage, pct, **meta)`` so
        downstream consumers (the web runner, the LAN server, integration
        tests) can surface per-stage diagnostics like
        ``caption_records_empty=3`` without inheriting the orchestrator's
        internal state. Logged alongside the ``(pct)`` field.
        """
        logger.info("pipeline.stage %s (%d%%) meta=%s", stage, pct, meta)
        if on_progress is not None:
            try:
                on_progress(stage, pct, **meta)
            except Exception:
                pass
    _progress("ingest", 5)
    doc: DocumentInput = ingest(p)
    doc_id = derive_doc_id(doc.uri)
    log_dir = output_dir.parent / "logs" if output_dir.name != "logs" else output_dir
    log_handler = attach_doc_log(strip_uir_prefix(doc_id), log_dir)
    try:
        logger.info("ingested %s: %d pages, sha256=%s", doc.uri, doc.page_count, doc.sha256[:12])

        # Stage 2/3 combined: open pdfplumber once and emit BOTH per-page
        # text AND per-word geometries (Tier 1.5 #1). The previous build
        # synthesised a full-canvas bbox per token, which masked the
        # LayoutClassifier's ``_HEADER_Y_PX = 80`` heuristic and made
        # every page produce a ``header`` region -- blocking downstream
        # ``intent.region_kind`` clustering and ``section_path`` detection.
        _progress("extract_text", 20)
        page_data = _get_page_words_with_real_coords(p, page_numbers=page_numbers)
        page_text_pairs: list[tuple[int, str]] = [(pn, text) for pn, _w, text in page_data]
        ocr_pages: list[OCRPage] = [OCRPage(page_number=pn, words=words) for pn, words, _t in page_data]
        _progress("synthesize_ocr", 30)

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

        # Stage 5.5 (Tier 3, per PLAN_TIER3.md): image captioning.
        # Detects figure regions via pdfplumber.Page.images, renders each
        # crop with PyMuPDF, runs Florence-2-base for a structured caption,
        # and emits ChunkNode-compatible shims that share the BGE-embedding
        # pipeline with text chunks. Fail-soft: any exception here is logged
        # and the document emits without figure captions (no UIR schema break).
        _progress("figure_caption", 60)
        figure_chunk_shims: list[Any] = []
        # Counter exposed via on_progress so the UI / integration tests
        # can surface "X figures, Y captioned, Z empty" -- prevents silent
        # loss when Florence-2 fail-softs on bad crops (Tier 3 fix #4).
        caption_records_total = 0
        caption_records_with_text = 0
        try:
            from uir_pipeline.caption import caption_figures_in_pdf
            from uir_pipeline.utils import count_tokens as _bpe_count_tokens
            figure_records = caption_figures_in_pdf(p, page_numbers=page_numbers)
            for fig in (figure_records or []):
                caption_records_total += 1
                cap = (fig.get("caption") or "").strip()
                if not cap:
                    continue
                caption_records_with_text += 1
                figure_chunk_shims.append(SimpleNamespace(
                    text=cap,
                    # BPE token count, not word-split: Florence-2's <MORE_DETAILED_CAPTION>
                    # output is subword-tokenized downstream (BGE embedder), so
                    # keeping the figure ChunkNode on the same scale avoids a
                    # UI badge mismatch and keeps BGE chunk-overlap stitching coherent.
                    token_count=_bpe_count_tokens(cap),
                    page=int(fig["page"]),
                    bbox=tuple(fig["bbox_canvas"]),
                    # 0.8 heuristic floor: Florence-2 doesn't expose per-image
                    # logprob at its API surface, and CaptionerBeam scores aren't
                    # directly translatable to a confidence scalar. PLAN_TIER3
                    # risk 8 flagged confidence propagation as deferred. Re-tune
                    # once we collect a labelled figure-caption dataset.
                    confidence=0.8,
                    modal_features={
                        # Tier 1.5 #2: canonicalize the label against
                        # LayoutLabel. ``caption`` is a first-class label
                        # (the chunk carries the caption TEXT -- ``figure``
                        # is reserved for raw figure regions without text).
                        "intent": {"region_kind": "caption"},
                        "figure": {
                            "image_b64": fig.get("image_b64"),
                            "caption_prompt": fig.get("caption_prompt"),
                            "caption_model": fig.get("caption_model"),
                        },
                    },
                ))
        except Exception as exc:
            logger.warning("figure caption stage failed (fail-soft): %s", exc)
            _progress(
                "figure_caption", 60,
                caption_records_total=0, caption_records_with_text=0,
                caption_records_empty=0, error=str(exc),
            )
        else:
            # Emit end-of-stage progress with the actual counts so the
            # caller (web UI / integration test) can diagnose silent loss.
            _progress(
                "figure_caption", 60,
                caption_records_total=caption_records_total,
                caption_records_with_text=caption_records_with_text,
                caption_records_empty=caption_records_total - caption_records_with_text,
            )

        # Stage 6: chunking -- union of layout regions + table markdown + figure captions.
        # Tier 1 intent metadata: walk regions top-down, track ``section_path``
        # state by detecting numbered headings (e.g. ``"3.2 Multi-Head Attention"``),
        # and attach ``region_kind`` (= LayoutLabel.value) to each emitted chunk.
        # The heading regex is anchored at the start of the line and conservative
        # -- only structural headings (numeric prefix; unnumbered items like
        # ``Abstract`` / ``References`` / ``Acknowledgments`` take their literal
        # text as the path) trigger an update. Cross-chunk linking
        # (``preceding_chunk_id`` / ``following_chunk_id``) is wired in
        # Stage 9 below once deterministic chunk IDs are assigned.
        _progress("chunk", 70)
        all_chunks: list[Any] = []
        # Match structural numbering: ``3``, ``3.2``, ``3.2.1`` followed by
        # ``.`` or whitespace, then the section title. Conservative --
        # we never update ``_section_path`` on a non-match, so prose that
        # happens to start with a year ("2024 saw...") won't trigger.
        _section_heading_re = re.compile(
            r"^\s*(\d+(?:\.\d+)*)[\.\s]+(\S.{2,})$"
        )
        _section_path = ""  # starts empty; first heading lights it up
        n_regions = len(all_regions)
        for i, region in enumerate(all_regions):
            label_str = (
                region.label.value if hasattr(region.label, "value")
                else str(region.label)
            )
            if label_str == "heading":
                m = _section_heading_re.match(region.text.strip())
                if m:
                    _section_path = m.group(1)
                else:
                    # Unnumbered heading (Abstract, References, etc.) -- use
                    # the literal text as the path so it's still queryable
                    # via intent-shaped queries.
                    _section_path = region.text.strip().rstrip(".").strip()
            # A region is the LAST of its section if (a) the next region is
            # a new heading OR (b) this is the last region in the document.
            next_label = (
                str(all_regions[i + 1].label.value)
                if i + 1 < n_regions and hasattr(all_regions[i + 1].label, "value")
                else ""
            )
            is_last_of_section = (
                i == n_regions - 1 or next_label == "heading"
            )
            all_chunks.extend(chunk_text(
                region.text,
                page=region.page,
                bbox=region.bbox,
                region_kind=label_str,
                section_path=(_section_path or None),
                is_section_first=(label_str == "heading"),
                is_section_last=bool(is_last_of_section),
            ))
        for table in table_drafts:
            all_chunks.extend(chunk_text(
                table.markdown,
                page=table.page_number,
                bbox=table.bbox,
                region_kind="table",
            ))
        all_chunks.extend(figure_chunk_shims)  # Tier 3 captions get BGE vectors
        # Drop residual 1-3 char noise chunks (Fix Plan item #4). These
        # are axis-tick fragments such as ``"0"`` / ``"0.5"`` / ``"##"``
        # that leak through ``LayoutClassifier`` from figure regions. They
        # destroy sentence-level retrieval signal without adding context.
        # Real arxiv sentences are >= 4 chars in length, so 4 is a safe
        # floor (we keep "BERT"-class acronym chunks, but drop "0.5").
        all_chunks = [ck for ck in all_chunks if len(ck.text.strip()) >= 4]
        if not all_chunks and page_text_pairs:
            # No regions / no tables -- chunk the whole document text.
            full_text = " ".join(text for _, text in page_text_pairs if text)
            all_chunks = chunk_text(full_text, page=1)

        # Stage 7: enrich (NER + co-occurrence)
        _progress("enrich", 80)
        enrichment = enrich_chunks([c.text for c in all_chunks])
        # Filter Arxiv-style boilerplate entities that pdfplumber emits
        # (concatenated narrow-kerned tokens from the permission footer
        # and corporate affiliation blocks). Drop the matching entities
        # and any relationships whose endpoints reference them. The
        # pattern set covers the documented ArXiv noise -- extend here if a
        # new boilerplate source shows up in the entity-quality log.
        before_entity_count = len(enrichment.entities)
        kept_entities = [e for e in enrichment.entities if not _is_boilerplate(e.text)]
        kept_text = {e.text for e in kept_entities}
        kept_relations = [
            r for r in enrichment.relationships
            if r.from_text in kept_text and r.to_text in kept_text
        ]
        dropped_entities = before_entity_count - len(kept_entities)
        dropped_relations = len(enrichment.relationships) - len(kept_relations)
        if dropped_entities or dropped_relations:
            logger.info(
                "boilerplate-filter: dropped %d entities and %d relationships (arXiv footer noise)",
                dropped_entities, dropped_relations,
            )
            _progress(
                "enrich", 80,
                dropped_entities=dropped_entities,
                dropped_relations=dropped_relations,
                entity_count=len(kept_entities),
                relationship_count=len(kept_relations),
            )
        # EnrichmentResult is a frozen dataclass -- rebuild instead of
        # mutating so downstream consumers see the post-filter view in one
        # place. Topics carry through unmodified.
        enrichment = EnrichmentResult(
            entities=kept_entities,
            relationships=kept_relations,
            topics=enrichment.topics,
        )

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
        chunk_ids: list[str] = []
        for i, ck in enumerate(all_chunks):
            ck_id = deterministic_node_id("chunk", doc_id, i, ck.text[:64])
            chunk_ids.append(ck_id)
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
        # Tier 1.C: wire consecutive ``preceding_chunk_id`` and
        # ``following_chunk_id`` per chunk. We use consecutive wiring (not
        # cross-section jumps) so the co-occurrence sliding window in the
        # enrich stage stays coherent. A future "section_first" jump can be
        # layered on top if intent-shaped queries need it.
        for i, cn in enumerate(chunk_nodes):
            mf = cn.modal_features
            if i > 0:
                mf["preceding_chunk_id"] = {"chunk_id": chunk_ids[i - 1]}
            if i < len(chunk_nodes) - 1:
                mf["following_chunk_id"] = {"chunk_id": chunk_ids[i + 1]}

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

        # Lift distinct ``modal_features.section.path`` values into real
        # ``StructureNode(type="section")`` parents (Fix Plan item #3).
        # ``StructureChild`` is a discriminated union so a section node
        # can wrap chunk children with no schema churn. Chunks without a
        # section path stay directly under root. ``current_section``
        # state is local so we always append consecutive same-path chunks
        # to the same parent before opening a new section on path change.
        children: list[Any] = []
        current_section: StructureNode | None = None
        current_path: str = ""
        for cn in chunk_nodes:
            section_path: str = (
                (cn.modal_features.get("section", {}).get("path") or "")
                if cn.modal_features
                else ""
            )
            if section_path:
                if current_section is None or current_path != section_path:
                    sec_id = deterministic_node_id(
                        "section", doc_id, len(children), section_path,
                    )
                    current_section = StructureNode(
                        id=sec_id,
                        type="section",
                        title=section_path,
                        page=cn.page,
                        children=[cn],
                    )
                    children.append(current_section)
                    current_path = section_path
                else:
                    current_section.children.append(cn)
            else:
                current_section = None
                current_path = ""
                children.append(cn)
        root = StructureNode(
            id=doc_id,
            type="document",
            title=metadata.title,
            page=1,
            children=children,
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
