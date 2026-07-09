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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    return (_clamp(x1), _clamp(y1), _clamp(x2), _clamp(y2))


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
            return v.strip()
    # Docling FigureItem / TableItem handle text via export methods.
    for method in ("export_to_markdown", "get_text"):
        fn = getattr(item, method, None)
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, str) and v.strip():
                    return v.strip()
            except Exception:  # noqa: BLE001 -- export failures are silent here
                continue
    return ""


def _page_number(item: Any, fallback: int) -> int:
    """Extract a stable 1-based page number for ``item`` from provenance."""
    try:
        prov = getattr(item, "prov", None)
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
    against a fake class. When ``None``, this function calls
    :func:`_import_docling_or_raise` and instantiates the real
    ``DocumentConverter``.

    Raises :class:`DoclingUnavailable` if docling isn't installed OR
    the conversion step fails (caller should fall through to the
    legacy pdfplumber path on this exception).
    """
    pdf_path = Path(pdf_path).expanduser()
    if converter is None:
        DocumentConverter = _import_docling_or_raise()
        # Real `DocumentConverter().convert(p)` is the upstream contract;
        # we instantiate once per call so the HF model state is reused
        # but a fresh conversion context is opened per PDF.
        converter = DocumentConverter()
    try:
        result = converter.convert(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 -- Docling conversion errors are heterogeneous
        raise DoclingUnavailable(
            f"docling converter failed on {pdf_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    doc = getattr(result, "document", None) or getattr(result, "output", None)
    if doc is None:
        raise DoclingUnavailable(
            f"docling converter returned no document on {pdf_path} "
            "(unexpected result.shape)"
        )

    return _walk_doc(doc)


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
        bbox = _bbox_xyxy(getattr(ti, "prov", None) and getattr(ti.prov, "bbox", None))
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
    pages = getattr(doc, "pages", None) or []
    page_iter = pages if pages else [None]  # body-only fallback
    for idx, page in enumerate(page_iter, start=1):
        text_buffer: list[str] = []
        page_no = idx
        if page is not None and hasattr(page, "page_no"):
            try:
                pn = int(page.page_no)
                if pn >= 1:
                    page_no = pn
            except Exception:  # noqa: BLE001
                pass
        items = list(getattr(page, "items", None) or []) if page is not None else list(getattr(doc, "texts", None) or [])
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
                bbox = _bbox_xyxy(
                    getattr(getattr(it, "prov", None), "bbox", None)
                )
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
            bb = _bbox_xyxy(
                getattr(getattr(pi, "prov", None), "bbox", None)
            )
        except Exception:  # noqa: BLE001
            bb = (0, 0, 0, 0)
        pg = _page_number(pi, fallback=1)
        key = (pg, bb)
        if key in seen_pic_keys or bb == (0, 0, 0, 0):
            continue
        seen_pic_keys.add(key)
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
    "DoclingResult",
    "DoclingUnavailable",
    "_LABEL_MAP",
    "docling_environment_enabled",
    "extract_with_docling",
]
