"""docling_extract -- Docling-backed fast-path text + layout extractor.

PLAN §17 §OCR follow-up: the previous fast path used pdfplumber text
extraction, which flattens sections / tables / figures / math into a
linear stream before the chunker sees them. Docling (IBM, MIT) emits
pre-typed DoclingDocument items so chunks come out structurally typed
without a regex / heuristic post-hoc pass.

This module is a defensive wrapper around the heavy `docling` package:
``DocumentConverter`` is lazy-imported on first call so an environment
without docling installed can still ``from docling_extract import``
this module cleanly. The wrapper raises :class:`DoclingUnavailable`
when docling is missing OR the converter fails -- the calling orchestrator
catches that and falls back to the legacy pdfplumber path so existing
tests / PDFs still produce valid UIR.

Output contract:
    ``DoclingResult.regions`` -- list of :class:`LayoutRegion`-shaped
        dicts with ``text``, ``page``, ``bbox`` ``(x1,y1,x2,y2)``,
        ``label`` string drawn from the existing pipeline vocabulary
        (``"heading"``, ``"paragraph"``, ``"table"``, ``"figure"``,
        ``"caption"``, ``"list_item"``).
    ``DoclingResult.tables`` -- list of :class:`TableDraft`-shaped dicts
        carrying ``markdown``, ``page``, ``bbox``. Docling exports
        each table to native Markdown so the chunker's existing
        markdown-aware path renders the table verbatim.
    ``DoclingResult.page_texts`` -- ``[(page_number, joined_text),
        ...]`` for any consumer that needs page-level text outside
        the typed regions (e.g. the noise-filter fallback path).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# Vocabulary mapping. Docling's GroupLabel enum names vary by version,
# so we use duck-typed string comparison and unknown-label fallback.
# Keys are matched lower-cased; values are the pipeline's canonical labels.
_LABEL_MAP: dict[str, str] = {
    "title": "heading",
    "section_header": "heading",
    "heading": "heading",
    "subtitle": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list",
    "list": "list",
    "table": "table",
    "figure": "figure",
    "picture": "figure",
    "caption": "caption",
    "formula": "paragraph",  # MVP: render formula-bearing blocks as prose
    "equation": "paragraph",
    "code": "paragraph",
}


@dataclass
class DoclingResult:
    """Result of running :func:`extract_with_docling` on a PDF path.

    The lists are python lists rather than tuples so callers can mutate
    in-place (e.g. add a fallback region) without rebuilding the dataclass.
    Empty lists signal "no structured content here" -- callers should
    fall through to the legacy pdfplumber extraction in that case.
    """
    regions: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    page_texts: list[tuple[int, str]] = field(default_factory=list)
    pictures: list[dict[str, Any]] = field(default_factory=list)


class DoclingUnavailable(RuntimeError):
    """Raised when the docling package / converter is missing or fails.

    The orchestrator catches this :class:`DoclingUnavailable` and falls
    back to the pdfplumber fast path. We use a custom exception type so
    the orchestrator can distinguish ``docling`` failures from real
    PDF parse errors and not mistakenly retry the heavy Docling call.
    """


class DoclingPartialConversion(DoclingUnavailable):
    """Docling converted only part of the document.

    Docling reports ``ConversionStatus.PARTIAL_SUCCESS`` when some pages
    fail (commonly ``std::bad_alloc`` under memory pressure) but the rest
    convert. It does **not** raise -- it hands back a ``DoclingDocument``
    containing whatever survived.

    Silently accepting that is the worst possible outcome for this
    pipeline: the job reports ``done``, the UIR looks well-formed, and an
    agent later answers questions from 30% of a contract with no
    indication that the other 70% was dropped. A loud failure is
    recoverable; a quiet one is not.

    Set ``DOCLING_ALLOW_PARTIAL=1`` to downgrade this to a warning and
    keep the partial document -- only do that if the caller genuinely
    tolerates missing pages.
    """


def _import_docling_or_raise() -> Any:
    """Lazy-import :class:`docling.document_converter.DocumentConverter`.

    The docling package transitively pulls in torch + transformers +
    onnxruntime + a 2 GB HuggingFace weight cache. That cost is paid
    only on first invocation -- importing this module is free.

    Raises :class:`DoclingUnavailable` if the import fails so callers
    can fall through to the legacy pdfplumber path without crashing.
    """
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # noqa: BLE001 -- import-time errors are diverse
        raise DoclingUnavailable(
            f"docling package not importable: {type(exc).__name__}: {exc}"
        ) from exc
    return DocumentConverter


def _build_converter(*, ocr: bool = True) -> Any:
    """Instantiate a ``DocumentConverter`` on the pypdfium2 PDF backend.

    Docling's default backend (``docling-parse`` v4) raises a native
    ``std::bad_alloc`` inside its ``preprocess`` stage on ordinary
    born-digital PDFs. Measured on a 15-page arXiv paper:

        default backend             SIGSEGV (exit 139), server dies
        default backend, OCR off    PARTIAL_SUCCESS, 11/15 pages dropped
        pypdfium2 backend           SUCCESS, 0 errors, 3.3x the text

    It is not a memory ceiling: peak RSS is *higher* on the runs that
    succeed (1.7 GB) than on the ones that die (1.5 GB), against 2.4 GB
    free. ``page_batch_size`` has no effect on it either -- 1 and 4
    produce byte-identical output. The allocation that fails is per-page
    and oversized, so a larger machine only postpones it.

    pypdfium2 is a binding to PDFium, the renderer in Chrome. Docling
    ships it as a first-class backend.

    ``ocr`` toggles docling's OCR stage. It is not implicated in the crash
    -- ``preprocess`` runs before it -- but it roughly doubles conversion
    time and is pure waste on born-digital PDFs. See :func:`_resolve_ocr`
    for how the caller decides.
    """
    DocumentConverter = _import_docling_or_raise()
    try:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
    except Exception as exc:  # noqa: BLE001
        # A docling old enough to lack these symbols still converts; take
        # the default backend rather than refusing to run at all.
        logger.warning(
            "docling pypdfium2 backend unavailable (%s: %s); falling back to "
            "the default backend, which may crash on multi-page PDFs.",
            type(exc).__name__, exc,
        )
        return DocumentConverter()

    options = PdfPipelineOptions()
    options.do_ocr = ocr
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=PyPdfiumDocumentBackend, pipeline_options=options
            )
        }
    )


# A born-digital page carries hundreds of characters. A scanned page read
# without OCR carries ~0 -- pypdfium finds no embedded glyphs at all. The gap
# is wide enough that the exact threshold barely matters; 50 sits far from
# both populations. Averaged over the document, so a few blank or full-page
# figure pages don't trip a re-run of an otherwise fine text PDF.
_SCANNED_CHARS_PER_PAGE = 50


def _resolve_ocr() -> str:
    """Return the OCR strategy: ``"auto"``, ``"on"`` or ``"off"``.

    ``DOCLING_OCR`` unset (or ``auto``) means: convert without OCR, and
    only pay for it if the result looks like a scan.
    """
    raw = (os.environ.get("DOCLING_OCR") or "auto").strip().lower()
    if raw in ("1", "true", "yes", "on", "force"):
        return "on"
    if raw in ("0", "false", "no", "off", "never"):
        return "off"
    if raw != "auto":
        logger.warning("unrecognised DOCLING_OCR=%r; treating as 'auto'", raw)
    return "auto"


def _text_char_count(doc: Any) -> int:
    return sum(
        len(getattr(t, "text", "") or "") for t in (getattr(doc, "texts", None) or [])
    )


def _page_count(doc: Any) -> int:
    pages = getattr(doc, "pages", None)
    try:
        return max(1, len(pages))  # type: ignore[arg-type]
    except TypeError:
        return 1


def _looks_scanned(doc: Any) -> bool:
    """True when the document yielded so little text it must be page images.

    A scanned PDF read without OCR does not fail -- it converts cleanly to a
    document with no text. The job would report ``done`` over an empty UIR.
    This is the same silent-data-loss shape as PARTIAL_SUCCESS, reached from
    the other direction, so it gets the same treatment: detect and act.
    """
    return _text_char_count(doc) / _page_count(doc) < _SCANNED_CHARS_PER_PAGE


def _convert_checked(converter: Any, pdf_path: Path) -> Any:
    """Run one conversion and return its ``DoclingDocument``."""
    try:
        result = converter.convert(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 -- Docling conversion errors are heterogeneous
        raise DoclingUnavailable(
            f"docling converter failed on {pdf_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _assert_conversion_complete(result, pdf_path)

    doc = getattr(result, "document", None) or getattr(result, "output", None)
    if doc is None:
        raise DoclingUnavailable(
            f"docling converter returned no document on {pdf_path} "
            "(unexpected result.shape)"
        )
    return doc


def _bbox_xyxy(bbox: Any) -> tuple[int, int, int, int]:
    """Coerce a Docling bbox (likely ``(l, t, r, b)``) into ``(x1, y1, x2, y2)``.

    Docling's ``BoundingBox`` exposes ``.l``/``.t``/``.r``/``.b`` as
    ``Optional[float]``. We defensively ``getattr`` so older / newer
    Docling builds don't trip us, and clamp to the UIR canvas
    (0-1000) so Pydantic's BoundingBox validator doesn't reject
    out-of-range downstream values.
    """
    try:
        if hasattr(bbox, "l"):
            x1, y1, x2, y2 = bbox.l, bbox.t, bbox.r, bbox.b
        elif isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            return (0, 0, 0, 0)
    except Exception:  # noqa: BLE001 -- defensive against bad bbox shapes
        return (0, 0, 0, 0)
    # Clamp to the UIR canvas so Pydantic's BoundingBox validator passes.
    def _clamp(v: Any) -> int:
        try:
            return max(0, min(1000, int(round(float(v)))))
        except (TypeError, ValueError):
            return 0
    cx1, cy1, cx2, cy2 = _clamp(x1), _clamp(y1), _clamp(x2), _clamp(y2)
    # Docling's BoundingBox defaults to a BOTTOMLEFT coord origin, so ``t``
    # is numerically *greater* than ``b`` and a naive l/t/r/b -> x1/y1/x2/y2
    # mapping yields y1 > y2. UIR's ChunkNode validator requires
    # ``x1 <= x2 and y1 <= y2``. Boxes are axis-aligned, so ordering each
    # axis is correct under either origin convention.
    if cx1 > cx2:
        cx1, cx2 = cx2, cx1
    if cy1 > cy2:
        cy1, cy2 = cy2, cy1
    return (cx1, cy1, cx2, cy2)


def _label_of(item: Any) -> str:
    """Return the canonical label for a Docling item.

    Docling exposes ``item.label`` as either a ``GroupLabel`` enum or a
    plain string depending on the version. ``item.label_name`` is the
    newer descriptive accessor. We duck-type both.
    """
    raw = None
    for attr in ("label_name", "label"):
        v = getattr(item, attr, None)
        if v is not None:
            raw = v
            break
    if raw is None:
        return "paragraph"
    key = str(raw).lower().strip()
    return _LABEL_MAP.get(key, "paragraph")


#: A decimal point that PDF glyph extraction split apart: ``28 . 4``, ``0 . 1``.
#: The whitespace *before* the dot is the signature -- prose never puts a space
#: there, so this cannot rejoin a sentence boundary like ``"...in 2017. 5 of
#: them..."``, which has no leading space. A trailing space is optional because
#: the split shows up as both ``0 . 1`` and ``0 .1``.
_SPLIT_DECIMAL_RE: Final[re.Pattern[str]] = re.compile(r"(?<=\d)\s+\.\s*(?=\d)")


def normalize_extracted_text(text: str) -> str:
    """Repair glyph-spacing artifacts in text lifted out of a PDF.

    pypdfium reports per-glyph positions and Docling joins them on advance
    width, so a decimal point set with extra kerning becomes its own token:
    the attention paper's ``Pdrop = 0.1`` extracts as ``Pdrop = 0 . 1`` and
    its ``28.4`` BLEU as ``28 . 4``.

    That is not cosmetic. The chat prompt tells the model to "quote figures
    exactly as written", so it faithfully quotes ``0 . 1``; retrieval keyed on
    ``"0.1"`` never matches; and any downstream numeric parse fails. Fifteen
    occurrences in one 15-page paper.

    Applied at :func:`_text_of`, the single point every region, table and
    page-text passes through.
    """
    return _SPLIT_DECIMAL_RE.sub(".", text)


def _text_of(item: Any) -> str:
    """Return the text string for a Docling item.

    Different Docling classes store text under different attrs:
    DoclingTextItem exposes ``.text`` directly; higher-level containers
    may not. We duck-type, normalize whitespace, and return empty
    string when no text is present so the downstream chunker can
    dedup empty regions cheaply.
    """
    for attr in ("text", "orig", "exported_text"):
        v = getattr(item, attr, None)
        if isinstance(v, str) and v.strip():
            return normalize_extracted_text(v.strip())
    # Docling FigureItem / TableItem handle text via export methods.
    for method in ("export_to_markdown", "get_text"):
        fn = getattr(item, method, None)
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, str) and v.strip():
                    return normalize_extracted_text(v.strip())
            except Exception:  # noqa: BLE001 -- export failures are silent here
                continue
    return ""


def _first_prov(item: Any) -> Any:
    """Return an item's first provenance record, or ``None``.

    Docling 2.x models ``item.prov`` as a *list* of ``ProvenanceItem`` (an
    item can appear on more than one page). Older shapes exposed a single
    object. Reading ``.page_no`` / ``.bbox`` straight off the list silently
    yields ``None``, which is how every region ended up on page 1 with a
    zero bbox.
    """
    prov = getattr(item, "prov", None)
    if prov is None:
        return None
    if isinstance(prov, (list, tuple)):
        return prov[0] if prov else None
    return prov


def _page_number(item: Any, fallback: int) -> int:
    """Extract a stable 1-based page number for ``item`` from provenance."""
    try:
        prov = _first_prov(item)
        if prov is None:
            return fallback
        page = getattr(prov, "page", None) or getattr(prov, "page_no", None)
        if isinstance(page, int) and page >= 1:
            return page
    except Exception:  # noqa: BLE001
        pass
    return fallback


def extract_with_docling(
    pdf_path: Path | str,
    *,
    converter: Any | None = None,
) -> DoclingResult:
    """Run the Docling converter on ``pdf_path`` and return typed regions.

    ``converter`` is an optional injected ``DocumentConverter``-shaped
    INSTANCE (i.e. anything with a ``convert(path)`` method). Tests use
    it to skip the 2 GB weight download and exercise the mapping logic
    against a fake class. Injecting a converter bypasses OCR resolution
    entirely -- you get exactly the one conversion you asked for.

    Otherwise OCR is chosen by ``DOCLING_OCR``: ``auto`` (the default)
    converts without OCR and re-converts *with* it only when the result
    looks like a scan. That keeps born-digital PDFs off the slow path --
    OCR roughly doubles conversion time -- without silently returning an
    empty document for a scan.

    Raises :class:`DoclingUnavailable` if docling isn't installed OR
    the conversion step fails (caller should fall through to the
    legacy pdfplumber path on this exception).
    """
    pdf_path = Path(pdf_path).expanduser()
    if converter is not None:
        return _walk_doc(_convert_checked(converter, pdf_path))

    mode = _resolve_ocr()
    if mode == "on":
        return _walk_doc(_convert_checked(_build_converter(ocr=True), pdf_path))

    doc = _convert_checked(_build_converter(ocr=False), pdf_path)
    if mode == "off" or not _looks_scanned(doc):
        return _walk_doc(doc)

    # No embedded glyphs: these pages are images. Pay for OCR now rather
    # than hand back a well-formed, empty UIR.
    logger.info(
        "%s yielded %d chars over %d page(s) without OCR; re-converting with "
        "OCR enabled (set DOCLING_OCR=off to skip).",
        pdf_path, _text_char_count(doc), _page_count(doc),
    )
    return _walk_doc(_convert_checked(_build_converter(ocr=True), pdf_path))


def _assert_conversion_complete(result: Any, pdf_path: Path) -> None:
    """Raise unless Docling converted the whole document.

    ``converter.convert()`` returns normally on a partial conversion; the
    only signal is ``result.status``. Reading it is the difference between
    "this document failed" and a UIR that silently omits most of its pages.

    Duck-typed on purpose: fake converters in tests need not carry a
    ``status``, and a ``status`` we don't recognise is treated as fine
    rather than blocking an otherwise-good conversion.
    """
    status = getattr(result, "status", None)
    if status is None:
        return
    name = str(getattr(status, "name", status)).upper()

    if name in ("SUCCESS", "PENDING", "STARTED"):
        return

    errors = list(getattr(result, "errors", None) or [])
    detail = "; ".join(str(e)[:160] for e in errors[:3]) or "no error detail reported"

    if name == "PARTIAL_SUCCESS":
        if _env_flag("DOCLING_ALLOW_PARTIAL"):
            logger.warning(
                "docling PARTIAL_SUCCESS on %s -- keeping the partial document "
                "because DOCLING_ALLOW_PARTIAL is set. %d page error(s): %s",
                pdf_path, len(errors), detail,
            )
            return
        raise DoclingPartialConversion(
            f"docling converted {pdf_path} only partially "
            f"({len(errors)} page error(s)): {detail}. "
            "Pages were dropped; the resulting UIR would be incomplete. "
            "If the detail says std::bad_alloc, check that the pypdfium2 "
            "backend is actually in use (see _build_converter) -- docling's "
            "default backend fails this way on ordinary PDFs. Lowering "
            "page_batch_size does NOT help; it has no measurable effect. "
            "Set DOCLING_ALLOW_PARTIAL=1 to accept the partial document."
        )

    # FAILURE / SKIPPED / anything else non-success.
    raise DoclingUnavailable(
        f"docling conversion of {pdf_path} reported status={name}: {detail}"
    )


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _walk_doc(doc: Any) -> DoclingResult:
    """Walk a DoclingDocument and emit UIR-shaped regions / tables.

    Docling's structure is version-dependent: some builds expose
    ``doc.pages[i].items``, others expose only ``doc.body`` / a flat
    ``doc.texts``. We duck-type over both layouts so the wrapper works
    across the 2.x line. Items missing all expected attributes are
    silently dropped (the orchestrator's noise-filter will catch their
    downstream consequences anyway).
    """
    regions: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    page_texts: list[tuple[int, str]] = []
    pictures: list[dict[str, Any]] = []

    # First pass: tables (Docling carries them as a top-level collection
    # in addition to embedding them in the page stream; emit once so
    # we don't double-count).
    table_items = getattr(doc, "tables", None) or []
    seen_table_keys: set[tuple[int, str]] = set()
    for ti in table_items:
        try:
            md = _text_of(ti)
            if not md:
                continue
        except Exception:  # noqa: BLE001
            continue
        bbox = _bbox_xyxy(getattr(_first_prov(ti), "bbox", None))
        page = _page_number(ti, fallback=1)
        key = (page, md[:120])
        if key in seen_table_keys:
            continue
        seen_table_keys.add(key)
        tables.append({
            "markdown": md,
            "page": page,
            "bbox": bbox,
        })

    # Second pass: typed regions block-by-block, page-aware.
    #
    # Docling 2.x exposes ``doc.pages`` as a ``dict[int, PageItem]`` and puts
    # no ``items`` stream on the page object -- content lives in the flat
    # ``doc.texts``, each item carrying its own page in ``prov``. Iterating
    # the dict yields *keys* (ints), so the old code called
    # ``getattr(1, "items")`` -> None for every page and never reached the
    # ``doc.texts`` fallback, emitting zero regions for every document.
    pages_attr = getattr(doc, "pages", None) or []
    page_objs = list(pages_attr.values()) if isinstance(pages_attr, dict) else list(pages_attr)

    def _page_no_of(page: Any, fallback: int) -> int:
        try:
            pn = int(getattr(page, "page_no", fallback))
            return pn if pn >= 1 else fallback
        except Exception:  # noqa: BLE001
            return fallback

    # Prefer a page-attached item stream when a build offers one; otherwise
    # bucket the flat text stream by each item's own provenance page.
    if any(getattr(p, "items", None) for p in page_objs):
        groups = [
            (_page_no_of(p, i), list(getattr(p, "items", None) or []))
            for i, p in enumerate(page_objs, start=1)
        ]
    else:
        buckets: dict[int, list[Any]] = {}
        for it in list(getattr(doc, "texts", None) or []):
            buckets.setdefault(_page_number(it, 1), []).append(it)
        groups = sorted(buckets.items()) or [(1, [])]

    for page_no, items in groups:
        text_buffer: list[str] = []
        for it in items:
            try:
                label_raw = _label_of(it)
            except Exception:  # noqa: BLE001
                label_raw = "paragraph"
            try:
                text = _text_of(it)
            except Exception:  # noqa: BLE001
                text = ""
            try:
                bbox = _bbox_xyxy(getattr(_first_prov(it), "bbox", None))
            except Exception:  # noqa: BLE001
                bbox = (0, 0, 0, 0)
            try:
                page_no = _page_number(it, fallback=page_no)
            except Exception:  # noqa: BLE001
                pass
            if label_raw == "table":
                # Tables were already emitted in the first pass; skip
                # emission here to avoid duplicating with `tables`.
                continue
            if not text and label_raw in ("heading", "paragraph", "list_item"):
                # Empty headings/paragraphs are noise.
                continue
            region = {
                "text": text,
                "page": page_no,
                "bbox": bbox,
                "label": label_raw,
            }
            regions.append(region)
            if text:
                text_buffer.append(text)
        page_texts.append((page_no, "\n\n".join(text_buffer)))

    # Third pass: pictures/figures. Docling exposes them via either
    # ``doc.pictures`` (top-level collection) or page-level items. We
    # duck-type -- the bbox is already 0-1000-clamped by ``_bbox_xyxy``.
    seen_pic_keys: set[tuple[int, tuple]] = set()
    picture_items = list(getattr(doc, "pictures", None) or [])
    for pi in picture_items:
        try:
            bb = _bbox_xyxy(getattr(_first_prov(pi), "bbox", None))
        except Exception:  # noqa: BLE001
            bb = (0, 0, 0, 0)
        pg = _page_number(pi, fallback=1)
        # Not `key`: the tables loop above binds that name to a
        # (page, markdown) pair, and reusing it here reads as the same thing.
        pic_key = (pg, bb)
        if pic_key in seen_pic_keys or bb == (0, 0, 0, 0):
            continue
        seen_pic_keys.add(pic_key)
        pictures.append({
            "page": pg,
            "bbox": bb,
            "bbox_pixel": bb,  # already 0-1000 (no pdfplumber coords)
            "kind": "picture",
        })
    return DoclingResult(
        regions=regions, tables=tables, page_texts=page_texts,
        pictures=pictures,
    )


def docling_environment_enabled() -> bool:
    """Return ``True`` iff the ``docling`` package is importable.

    Used by the CLI to surface a one-line status when ``--fast-path docling``
    is requested in an environment where docling failed to install (a
    missing dep, a Python-version mismatch, an unsupported CUDA / ROCm
    toolchain). Returns False for the ``DoclingUnavailable`` path too,
    so the CLI never silently mis-routes.
    """
    try:
        _import_docling_or_raise()
        return True
    except DoclingUnavailable:
        return False


__all__ = [
    "DoclingPartialConversion",
    "DoclingResult",
    "DoclingUnavailable",
    "_LABEL_MAP",
    "docling_environment_enabled",
    "extract_with_docling",
]
