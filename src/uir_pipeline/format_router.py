"""format_router -- per-file format detection + extraction-route dispatch.

PLAN \u00a717 \u00a7Multi-format follow-up: widens the UIR pipeline from "PDF only"
to 13+ formats. Docling (already in deps) handles structured formats
(DOCX/PPTX/XLSX/HTML/EPUB/LaTeX/IPYNB/image-bearing) without a per-format
adapter; pageless text formats (TXT/MD/code/CSV/RTF) flow through a
:func:`src.uir_pipeline.chunk.paginate_pageless` token-window route.

This module is the single source of truth for "what format is this file
and how do we extract it":

    >>> fmt, route = format_router.route("foo.docx")
    >>> fmt.upper(), route.value
    ('DOCX', 'docling')

    >>> format_router.source_format_label("PDF")
    'PDF'

The magic-byte table is checked first because file extensions are
user-supplied and lie (e.g. a ``.txt`` is often a real PDF). ZIP-based
formats (DOCX/PPTX/XLSX/EPUB) are disambiguated by their internal
``word/`` / ``ppt/`` / ``xl/`` directory structure per the OOXML spec.
"""
from __future__ import annotations

import zipfile
from enum import Enum
from pathlib import Path
from typing import Final


# ----------------------------------------------------------------------------
# Route enum
# ----------------------------------------------------------------------------

class FormatRoute(str, Enum):
    """How :func:`pipeline.run` should extract a given file.

    ``PDF`` is the IBM Docling fast path (single backend -- pdfplumber
    was retired). ``DOCLING`` is a sibling: the same Docling fast path is
    reused on DOCX/PPTX/XLSX/HTML/EPUB/LaTeX/IPYNB because
    :class:`docling.document_converter.DocumentConverter` already accepts
    those formats natively.
    ``TEXT`` is the pageless route: read whole file -> paginate ->
    chunk -> embed without invoking Docling. ``IMAGE``
    feeds a single PIL image through :func:`caption.caption_images`
    for a one-shot Florence-2 caption that becomes a single-page
    :class:`LayoutRegion`. ``SKIP`` is the explicit opt-out (unknown
    extension, encrypted, unsupported).
    """
    PDF = "pdf"
    DOCLING = "docling"
    PPTX_NATIVE = "pptx"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SKIP = "skip"


# ----------------------------------------------------------------------------
# Magic-byte signatures
# ----------------------------------------------------------------------------

# PDF 1.x magic and 2.0 spec (PDF 32000-1:2008 \u00a77.5.2 + 32000-2:2017).
PDF_MAGIC: Final[bytes] = b"%PDF-"

# ZIP local-file-header signature. Used by DOCX/PPTX/XLSX/EPUB/ODT and
# any JAR-style container. We peek inside to disambiguate.
ZIP_MAGIC: Final[bytes] = b"PK\x03\x04"

# RTF always starts with ``{\rtf<N>...}``.
RTF_MAGIC: Final[bytes] = b"{\\rtf"

# Bytes read from head of file for magic-byte detection. 8 bytes is
# enough to discriminate every magic above.
_MAGIC_PREFIX_LEN: Final[int] = 8


# ----------------------------------------------------------------------------
# Extension tables
# ----------------------------------------------------------------------------

# Structured formats routed through Docling (``.convert()`` accepts them
# alongside PDF).
_DOCLING_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".docx", ".doc",       # Microsoft Word (``.doc`` is legacy binary; Docling may not accept -- skip)
    ".pptx", ".ppt",       # PowerPoint
    ".xlsx", ".xls",       # Excel
    ".epub",               # e-book
    ".html", ".htm",       # web
    ".tex",                # LaTeX source
    # NOT ``.ipynb``: DocumentConverter's allow-list carries no notebook
    # format, so it failed with "File format not allowed: x.ipynb". A
    # notebook is JSON, so the TEXT route reads its cells directly.
    # See ``pipeline._notebook_to_text``.
})

# Plain-text / code / structured-data formats that have no "page"
# concept. Read whole file -> :func:`chunk.paginate_pageless`.
_TEXT_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".txt", ".md", ".markdown",
    ".csv", ".tsv",
    ".rtf",
    # Docling's allow-list has no notebook format, so this reads the JSON and
    # concatenates the cells (``pipeline._notebook_to_text``).
    ".ipynb",
    # Common source code / config extensions; treated as plain text.
    ".py", ".pyi", ".pyx",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".go",
    ".rs",
    ".java", ".kt", ".kts", ".scala",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
    ".rb", ".php",
    ".swift", ".m", ".mm",
    ".sh", ".bash", ".zsh", ".fish",
    ".yaml", ".yml", ".toml", ".ini", ".conf",
    ".json", ".jsonl", ".ndjson",
    ".xml", ".svg",
    ".rst",
})

# Image formats -- routed through Florence-2 captioner.
_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff",
    ".bmp", ".gif", ".webp",
})

# Audio formats -- routed through vLLM Whisper + pyannote speaker diarization.
_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".mp3", ".wav", ".m4a",
    ".flac", ".ogg", ".aac", ".wma",
})

# Video formats -- routed through ffmpeg audio extraction + frame sampling + Whisper + Florence-2.
_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".mp4", ".avi", ".mov",
    ".webm", ".mkv", ".flv", ".wmv", ".m4v",
})


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------

def detect_format(path: str | Path) -> str:
    """Return the canonical format label for ``path`` (e.g. ``"PDF"``).

    Magic-bytes are checked first; we fall through to extension only
    when the magic is ambiguous (text / code / image files all share
    the same "no magic" signature). Returns ``""`` when neither path
    can detect a supported format -- callers should treat this as
    :data:`FormatRoute.SKIP`.
    """
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(_MAGIC_PREFIX_LEN)
    except OSError:
        return ""

    if head.startswith(PDF_MAGIC):
        return "PDF"
    if head.startswith(RTF_MAGIC):
        return "RTF"
    if head.startswith(ZIP_MAGIC):
        return _detect_zip_subtype(p)

    # No magic -- fall back to extension.
    ext = p.suffix.lower()
    if not ext:
        return ""
    if ext in _IMAGE_EXTENSIONS:
        return "IMAGE"
    if ext in _AUDIO_EXTENSIONS:
        return "AUDIO"
    if ext in _VIDEO_EXTENSIONS:
        return "VIDEO"
    if ext in _DOCLING_EXTENSIONS:
        return _detect_zip_subtype(p) if ext in {".docx", ".pptx", ".xlsx", ".epub"} \
            else ext.lstrip(".").upper()
    if ext in _TEXT_EXTENSIONS:
        return ext.lstrip(".").upper()
    return ""


_OOXML_DIR_TO_FORMAT: Final[dict[str, str]] = {
    "word/": "DOCX",
    "ppt/": "PPTX",
    "xl/": "XLSX",
}


def _detect_zip_subtype(path: Path) -> str:
    """Disambiguate ZIP-based formats by inspecting :meth:`namelist`.

    DOCX/PPTX/XLSX/EPUB all start with the same ZIP magic; we look one
    directory level in to tell them apart per the OOXML convention
    (``[Content_Types].xml`` at root + ``word/`` / ``ppt/`` / ``xl/``
    in the file list). EPUB carries its own ``META-INF/container.xml``
    marker. Returns ``""`` (caller falls back to extension) when the
    ZIP is malformed or the container doesn't carry OOXML structure.
    """
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            for prefix, label in _OOXML_DIR_TO_FORMAT.items():
                if any(n.startswith(prefix) for n in names):
                    return label
            if any(n == "META-INF/container.xml" for n in names):
                return "EPUB"
    except (zipfile.BadZipFile, OSError, KeyError):
        pass
    # Fallback to extension so the caller still gets a useful label.
    ext = path.suffix.lower().lstrip(".")
    return ext.upper() if ext else ""


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

# Files that share the ``.doc`` (legacy binary Word) or ``.ppt`` / ``.xls``
# (legacy binary Office) magic haven't been validated against Docling
# (only OOXML is documented). Treat them as SKIP until we add a real
# legacy-format adapter.
_LEGACY_BINARY_OFFICE: Final[frozenset[str]] = frozenset({"DOC", "PPT", "XLS"})


def classify_route(format_str: str) -> FormatRoute:
    """Map a :func:`detect_format` result to a :class:`FormatRoute`."""
    if not format_str:
        return FormatRoute.SKIP
    fs = format_str.upper()
    if fs == "PDF":
        return FormatRoute.PDF
    if fs in _LEGACY_BINARY_OFFICE:
        return FormatRoute.SKIP  # not yet supported
    if fs == "PPTX":
        # PLAN §17 §Multi-format follow-up: PPTX is structurally simple
        # (title + body placeholders per slide) -- the docling layout
        # model returns 0 regions for python-pptx-generated fixtures and
        # adds no value over a native walk. Route through a python-pptx
        # text walker instead. See ``_extract_pptx_route`` in
        # ``src/uir_pipeline/pipeline.py`` for the implementation.
        # NOTE: checked BEFORE the docling-extensions set because
        # ``.pptx`` IS in ``_DOCLING_EXTENSIONS_LABELS`` (the underlying
        # ``DocumentConverter`` accepts PPTX natively), so the order
        # matters -- this branch must run first or it is dead code.
        return FormatRoute.PPTX_NATIVE
    if fs in _DOCLING_EXTENSIONS_LABELS or fs in {"DOCX", "XLSX", "EPUB", "HTML", "TEX"}:
        return FormatRoute.DOCLING
    if fs == "IMAGE" or fs in {e.lstrip(".").upper() for e in _IMAGE_EXTENSIONS}:
        return FormatRoute.IMAGE
    if fs == "AUDIO" or fs in {e.lstrip(".").upper() for e in _AUDIO_EXTENSIONS}:
        return FormatRoute.AUDIO
    if fs == "VIDEO" or fs in {e.lstrip(".").upper() for e in _VIDEO_EXTENSIONS}:
        return FormatRoute.VIDEO
    if fs in {e.lstrip(".").upper() for e in _TEXT_EXTENSIONS}:
        return FormatRoute.TEXT
    return FormatRoute.SKIP


# Helper: built from ``_DOCLING_EXTENSIONS`` but flattened to upper-case
# labels without dots (``.html`` -> ``"HTML"``).
_DOCLING_EXTENSIONS_LABELS: Final[frozenset[str]] = frozenset(
    e.lstrip(".").upper() for e in _DOCLING_EXTENSIONS
)


# ----------------------------------------------------------------------------
# Top-level entry
# ----------------------------------------------------------------------------

def route(path: str | Path) -> tuple[str, FormatRoute]:
    """One-shot ``(format, route)`` dispatch for the orchestrator."""
    fmt = detect_format(path)
    return fmt, classify_route(fmt)


def source_format_label(format_str: str) -> str:
    """Return the value to write into :attr:`Source.format`.

    Preserves ``"PDF"`` for legacy v1 JSONs and uppercases everything
    else. Returns ``"UNKNOWN"`` for the SKIP path so a corrupted
    manifest still has a parseable label.
    """
    if not format_str:
        return "UNKNOWN"
    return format_str.upper()


# ----------------------------------------------------------------------------
# CLI rglob support
# ----------------------------------------------------------------------------

# Combined-extension set consumed by the CLI for ``rglob`` discovery.
# Includes everything ``route()`` can classify (incl. legacy binary
# office so the discovery step surfaces them before SKIP kicks in).
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".pdf",
    *_DOCLING_EXTENSIONS,
    *_TEXT_EXTENSIONS,
    *_IMAGE_EXTENSIONS,
    *_AUDIO_EXTENSIONS,
    *_VIDEO_EXTENSIONS,
})

#: Extensions the orchestrator can actually convert.
#:
#: A strict subset of :data:`SUPPORTED_EXTENSIONS`, which includes the legacy
#: binary Office formats (``.doc`` / ``.ppt`` / ``.xls``) because they are
#: *recognised*. They are not convertible: :func:`classify_route` sends them to
#: ``SKIP``. Uploading one used to be accepted by the web form and then fail
#: several seconds later inside ``ingest_any``. Callers that gate input --
#: the upload form, the CLI's directory sweep -- want this set.
CONVERTIBLE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    ext for ext in SUPPORTED_EXTENSIONS
    if classify_route(ext.lstrip(".").upper()) is not FormatRoute.SKIP
)


__all__ = [
    "CONVERTIBLE_EXTENSIONS",
    "FormatRoute",
    "SUPPORTED_EXTENSIONS",
    "classify_route",
    "detect_format",
    "route",
    "source_format_label",
]
