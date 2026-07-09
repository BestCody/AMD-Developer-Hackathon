"""ingest -- validate, hash, and extract metadata from a PDF file (Phase E).

PLAN \\u00a79 Phase E exit:
    -- unit tests for sha256 (known vector)
    -- MIME detection via PDF magic bytes (``%PDF-``)
    -- ``pypdf`` metadata extraction (title, author, created, page_count)
    -- returns immutable ``DocumentInput`` dataclass

The dataclass field names mirror the UIR ``Source`` + ``Metadata`` schema
(Phase B) so the Phase L ``assemble`` step can map directly without a
translation layer.

pypdf 6.x auto-parses PDF dates (``D:YYYYMMDDhhmmssZ``) into
``datetime`` with tzinfo. We trust that -- no manual date parser.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

# PDF 1.x magic (PDF 32000-1:2008 \\u00a77.5.2).
PDF_MAGIC: Final[bytes] = b"%PDF-"
_PDF_MAGIC_RE: Final[re.Pattern[bytes]] = re.compile(rb"%PDF-\d\.\d")

# RFC 8118 -- ``application/pdf`` is the canonical PDF MIME type.
PDF_MIME_TYPE: Final[str] = "application/pdf"

# Sha256 file-read chunk size -- 1 MiB balances syscalls vs allocation.
_SHA256_CHUNK_BYTES: Final[int] = 1024 * 1024

# Bytes read from the head of the file to detect the PDF magic.
_PDF_MAGIC_PREFIX_LEN: Final[int] = len(PDF_MAGIC) + 6

# Default filler title when the PDF has no /Title metadata.
_DEFAULT_TITLE: Final[str] = "(untitled)"
# Default language per PLAN \\u00a79 -- enriched in Phase J (spaCy NER).
_DEFAULT_LANGUAGE: Final[str] = "en"


@dataclass(frozen=True)
class DocumentInput:
    """Result of ingesting a raw document file on disk.

    Field names mirror UIR ``Source`` + ``Metadata`` so Phase L can map
    directly without translation. All fields are populated from the
    file -- none are inferable from each other.

    PLAN §17 §Multi-format widens the contract from PDF-only to any
    supported format. ``format`` and ``route`` are populated by the
    per-format ingress module (kept on the dataclass so legacy PDF-only
    callers see ``format="PDF"`` / ``route="pdf"`` for free).

    Note: ``source_path`` is the path the user provided (possibly a
    string), while ``uri`` is ``pathlib.Path.resolve().as_uri()`` -- the
    absolute ``file://`` form fed into UIR ``Source.uri``.
    """
    source_path: Path
    uri: str
    mime_type: str
    size_bytes: int
    sha256: str
    timestamp: datetime
    title: str | None
    author: str | None
    created: datetime | None
    modified: datetime | None
    page_count: int
    # PLAN §17 §Multi-format. Defaults preserve legacy v1 behaviour for
    # every existing PDF-only test / caller.
    format: str = "PDF"
    route: str | None = None

    def to_uir_source_metadata(self) -> tuple[Any, Any]:
        """Return ``(Source, Metadata)`` Pydantic models ready for UIRV1.

        Import is local to avoid a hard import cycle at module load:
        ``uir_schema`` does not depend on ``ingest``, but a top-level
        import here would force eager schema load wherever ``ingest``
        is referenced (e.g. orchestrator).

        PLAN §17 §Multi-format: ``Source.format`` + ``Source.route``
        mirror this dataclass. ``Metadata.format`` mirrors it too so
        JSON consumers can pivot on a single field without joining.

        Defaults:
            -- ``title`` falls back to ``"(untitled)"`` so the UIR schema's
               non-null ``title: str`` constraint is always satisfied.
            -- ``language`` defaults to ``"en"`` (Phase J will overwrite).
            -- ``domain`` stays ``None`` (Phase J may fill).
        """
        from uir_pipeline.uir_schema import Source, Metadata
        from uir_pipeline.format_router import source_format_label
        fmt_label = source_format_label(self.format)
        return (
            Source(
                uri=self.uri,
                format=fmt_label,
                route=self.route,
                mime_type=self.mime_type,
                size_bytes=self.size_bytes,
                checksum=f"sha256:{self.sha256}",
                timestamp=self.timestamp,
            ),
            Metadata(
                title=self.title or _DEFAULT_TITLE,
                author=self.author,
                created=self.created,
                modified=self.modified,
                page_count=self.page_count,
                language=_DEFAULT_LANGUAGE,
                domain=None,
                format=fmt_label,
            ),
        )


# ----------------------------------------------------------------------------
# Helpers (public for direct unit-test coverage)
# ----------------------------------------------------------------------------

def compute_sha256(path: Path, chunk_bytes: int = _SHA256_CHUNK_BYTES) -> str:
    """Stream ``path`` through sha256 and return the hex digest.

    Reads in ``chunk_bytes``-sized slices so the helper works on
    arbitrarily large PDFs without loading them into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_pdf_magic_bytes(path: Path) -> bool:
    """Return ``True`` iff ``path`` starts with the PDF 1.x magic prefix.

    Reads only the first ``_PDF_MAGIC_PREFIX_LEN`` bytes (``%PDF-<digit>.<digit>``)
    so the helper is cheap on multi-MB inputs.
    """
    with open(path, "rb") as f:
        head = f.read(_PDF_MAGIC_PREFIX_LEN)
    return bool(_PDF_MAGIC_RE.match(head))


def extract_pdf_metadata(path: Path) -> dict[str, Any]:
    """Use ``pypdf`` to extract title / author / dates / page_count.

    pypdf 6.x ``DocumentInformation`` already converts ``/CreationDate`` and
    ``/ModDate`` to tz-aware :class:`datetime` objects, so no manual
    parsing is required. ``TextStringObject`` values (returned for the
    title / author) are stripped of surrounding whitespace before return.
    """
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    info = reader.metadata
    return {
        "title": _text_or_none(info, "title"),
        "author": _text_or_none(info, "author"),
        "created": _datetime_or_none(info, "creation_date"),
        "modified": _datetime_or_none(info, "modification_date"),
        "page_count": len(reader.pages),
    }


def _text_or_none(info: Any, attr_name: str) -> str | None:
    """Return ``str(info.<attr_name>).strip()`` or ``None`` if empty."""
    if info is None:
        return None
    v = getattr(info, attr_name, None)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _datetime_or_none(info: Any, attr_name: str) -> datetime | None:
    """Return tz-aware :class:`datetime` or ``None``.

    pypdf returns tz-aware datetimes when the PDF embeds timezone info;
    if not, we attach UTC so downstream ISO8601 serialization has a
    consistent offset.
    """
    if info is None:
        return None
    v = getattr(info, attr_name, None)
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return None


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def ingest(path: str | Path) -> DocumentInput:
        """Validate, hash, and extract metadata from a PDF file.

        PLAN §17 §Multi-format: this function remains the PDF-specific
        ingress. Non-PDF formats route through dedicated ingress modules
        (``src/uir_pipeline/ingest_rtf.py`` for RTF, per-format Docling
        callers for DOCX/PPTX/XLSX/HTML/EPUB/LaTeX/IPYNB). The orchestrator
        (:func:`src.uir_pipeline.pipeline.run`) is the dispatcher -- it
        reads ``format_router.route(path)`` first and only calls ``ingest``
        when the route is ``FormatRoute.PDF``.

        Raises:
            FileNotFoundError: ``path`` does not exist (or is not a regular file).
            ValueError: file is not a PDF (magic-byte prefix missing).
            RuntimeError: ``pypdf`` fails to parse the file (corrupted / encrypted).
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{p} is not a regular file")
        if not detect_pdf_magic_bytes(p):
            raise ValueError(
                f"{p.name} is not a PDF (magic bytes %PDF-<version> absent "
                f"in first {_PDF_MAGIC_PREFIX_LEN} bytes)"
            )

        sha = compute_sha256(p)
        meta = extract_pdf_metadata(p)
        size = p.stat().st_size
        ts = datetime.now(timezone.utc)

        return DocumentInput(
            source_path=p,
            uri=p.resolve().as_uri(),
            mime_type=PDF_MIME_TYPE,
            size_bytes=size,
            sha256=sha,
            timestamp=ts,
            title=meta["title"],
            author=meta["author"],
            created=meta["created"],
            modified=meta["modified"],
            page_count=meta["page_count"],
            format="PDF",
            route="pdf",
        )


__all__ = [
    "DocumentInput",
    "PDF_MAGIC",
    "PDF_MIME_TYPE",
    "compute_sha256",
    "detect_pdf_magic_bytes",
    "extract_pdf_metadata",
    "ingest",
]
