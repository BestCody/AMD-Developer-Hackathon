"""tests/test_ingest.py -- PDF ingest (Phase E).

Tests cover:
    -- :func:`compute_sha256`: known vector + empty-file + streaming-matches-one-shot
    -- :func:`detect_pdf_magic_bytes`: valid + invalid + truncated prefixes
    -- :func:`extract_pdf_metadata`: page_count + title/author/dates via
       a freshly emitted ``pypdf.PdfWriter`` blob
    -- :func:`ingest`: full DocumentInput + ``to_uir_source_metadata``
       round-trip + rejection of non-PDF + rejection of missing path

PDFs are created at test time using ``pypdf.PdfWriter.add_blank_page`` so
no binary fixtures need to live in the repo. pypdf 6.x parses PDF dates
into tz-aware ``datetime`` natively, so we test against that contract.
"""
from __future__ import annotations

import hashlib
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pypdf import PdfWriter

from uir_pipeline.ingest import (
    DocumentInput,
    PDF_MAGIC,
    PDF_MIME_TYPE,
    compute_sha256,
    detect_pdf_magic_bytes,
    extract_pdf_metadata,
    ingest,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _write_blank_pdf(
    p: Path,
    *,
    n_pages: int = 1,
    info: dict | None = None,
) -> Path:
    """Build a minimal valid PDF (1+ blank pages, optional /Info dict).

    Returns the path the bytes were written to.
    """
    w = PdfWriter()
    if info:
        w.add_metadata(info)
    for _ in range(n_pages):
        w.add_blank_page(width=612, height=792)
    with open(p, "wb") as f:
        w.write(f)
    return p


# ----------------------------------------------------------------------------
# compute_sha256
# ----------------------------------------------------------------------------

def test_compute_sha256_known_vector_hello_world(tmp_path):
    """sha256('hello world') = b94d27b9... (the canonical NIST test vector,
    11 bytes -- no trailing newline)."""
    p = tmp_path / "known.txt"
    p.write_bytes(b"hello world")
    assert compute_sha256(p) == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_compute_sha256_empty_file(tmp_path):
    """sha256('') = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855."""
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert compute_sha256(p) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_compute_sha256_matches_one_shot_hashlib(tmp_path):
    """Streaming hash must equal a single-block hashlib invocation."""
    blob = b"ABCD" * 10_000
    p = tmp_path / "blob.bin"
    p.write_bytes(blob)
    assert compute_sha256(p) == hashlib.sha256(blob).hexdigest()


def test_compute_sha256_handles_chunk_boundary(tmp_path):
    """A blob spanning multiple chunks (1 MiB chunk size) hashes correctly."""
    blob = b"x" * (1024 * 1024 * 3 + 17)  # 3 MiB + 17 -- three full chunks + tail
    p = tmp_path / "multi.bin"
    p.write_bytes(blob)
    assert compute_sha256(p) == hashlib.sha256(blob).hexdigest()


def test_compute_sha256_with_smaller_chunk_size(tmp_path):
    """Smaller ``chunk_bytes`` must still produce the same digest."""
    blob = b"y" * (1024 * 512 + 7)
    p = tmp_path / "smallchunk.bin"
    p.write_bytes(blob)
    digest_one_mib = compute_sha256(p)
    digest_small = compute_sha256(p, chunk_bytes=128)
    assert digest_one_mib == digest_small


# ----------------------------------------------------------------------------
# detect_pdf_magic_bytes
# ----------------------------------------------------------------------------

def test_detect_pdf_magic_bytes_accepts_valid(tmp_path):
    p = tmp_path / "good.pdf"
    p.write_bytes(b"%PDF-1.7\n% rest of fake pdf\n")
    assert detect_pdf_magic_bytes(p) is True


def test_detect_pdf_magic_bytes_accepts_pdf_2_0(tmp_path):
    """Magic regex tolerates any 1.x or 2.x version (per PDF 32000)."""
    p = tmp_path / "pdf2.pdf"
    p.write_bytes(b"%PDF-2.0\n")
    assert detect_pdf_magic_bytes(p) is True


def test_detect_pdf_magic_bytes_rejects_text(tmp_path):
    p = tmp_path / "text.txt"
    p.write_bytes(b"hello world")
    assert detect_pdf_magic_bytes(p) is False


def test_detect_pdf_magic_bytes_rejects_truncated(tmp_path):
    """%PDF without the version digit-pair must NOT match."""
    p = tmp_path / "trunc.pdf"
    p.write_bytes(b"%PDF")  # missing version suffix
    assert detect_pdf_magic_bytes(p) is False


def test_detect_pdf_magic_bytes_rejects_wrong_version_format(tmp_path):
    """%PDF-1 (missing minor digit) -> no match."""
    p = tmp_path / "badvf.pdf"
    p.write_bytes(b"%PDF-1\ndata")
    assert detect_pdf_magic_bytes(p) is False


def test_detect_pdf_magic_bytes_rejects_empty(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert detect_pdf_magic_bytes(p) is False


def test_pdf_magic_constant_matches_spec():
    """The ``PDF_MAGIC`` public constant is the spec-mandated prefix."""
    assert PDF_MAGIC == b"%PDF-"


# ----------------------------------------------------------------------------
# extract_pdf_metadata
# ----------------------------------------------------------------------------

def test_extract_pdf_metadata_page_count_single(tmp_path):
    p = _write_blank_pdf(tmp_path / "one.pdf", n_pages=1)
    info = extract_pdf_metadata(p)
    assert info["page_count"] == 1


def test_extract_pdf_metadata_page_count_many(tmp_path):
    p = _write_blank_pdf(tmp_path / "many.pdf", n_pages=5)
    info = extract_pdf_metadata(p)
    assert info["page_count"] == 5


def test_extract_pdf_metadata_with_info_dict(tmp_path):
    """pypdf 6.x auto-parses /CreationDate + /ModDate -> tz-aware datetime."""
    p = _write_blank_pdf(
        tmp_path / "with_info.pdf",
        n_pages=2,
        info={
            "/Title": "Q2 Earnings",
            "/Author": "Buffy <buffy@example.com>",
            "/CreationDate": "D:20260401090000Z",
            "/ModDate": "D:20260403090000+00'00'",
        },
    )
    info = extract_pdf_metadata(p)

    assert info["title"] == "Q2 Earnings"
    assert info["author"] == "Buffy <buffy@example.com>"
    # pypdf 6.x keeps ``+00'00'`` suffix as tz=UTC.
    assert info["created"] == datetime(
        2026, 4, 1, 9, 0, tzinfo=timezone.utc
    )
    assert info["modified"] == datetime(
        2026, 4, 3, 9, 0, tzinfo=timezone.utc
    )
    assert info["page_count"] == 2


def test_extract_pdf_metadata_no_info_dict(tmp_path):
    """A PDF with no /Info returns ``None`` for title/author/dates."""
    p = _write_blank_pdf(tmp_path / "anon.pdf", n_pages=1)
    info = extract_pdf_metadata(p)
    assert info["title"] is None
    assert info["author"] is None
    assert info["created"] is None
    assert info["modified"] is None
    assert info["page_count"] == 1


def test_extract_pdf_metadata_strips_whitespace(tmp_path):
    """If /Title is "   Spaced   ", we strip before returning."""
    p = _write_blank_pdf(
        tmp_path / "spaced.pdf",
        info={"/Title": "   Spaced   ", "/Author": "  A  "},
    )
    info = extract_pdf_metadata(p)
    assert info["title"] == "Spaced"
    assert info["author"] == "A"


# ----------------------------------------------------------------------------
# ingest()
# ----------------------------------------------------------------------------

def test_ingest_returns_dataclass(tmp_path):
    p = _write_blank_pdf(tmp_path / "doc.pdf", n_pages=3)
    di = ingest(p)
    assert isinstance(di, DocumentInput)


def test_ingest_fills_size_and_hash_and_path(tmp_path):
    p = _write_blank_pdf(tmp_path / "doc.pdf", n_pages=2)
    di = ingest(p)
    assert di.source_path == p
    assert di.size_bytes == p.stat().st_size
    assert di.sha256 == compute_sha256(p)
    assert di.page_count == 2


def test_ingest_mime_type_is_application_pdf(tmp_path):
    p = _write_blank_pdf(tmp_path / "doc.pdf")
    di = ingest(p)
    assert di.mime_type == PDF_MIME_TYPE == "application/pdf"


def test_ingest_uri_starts_with_file_scheme(tmp_path):
    p = _write_blank_pdf(tmp_path / "doc.pdf")
    di = ingest(p)
    assert di.uri.startswith("file://")
    assert Path(di.uri.removeprefix("file://")).exists() or \
        di.uri.endswith("/doc.pdf")


def test_ingest_timestamp_within_recent_window(tmp_path):
    """``timestamp`` is set at ingest call -- should be ~now."""
    p = _write_blank_pdf(tmp_path / "doc.pdf")
    before = datetime.now(timezone.utc)
    di = ingest(p)
    after = datetime.now(timezone.utc)
    assert before <= di.timestamp <= after


def test_ingest_metadata_propagates(tmp_path):
    p = _write_blank_pdf(
        tmp_path / "meta.pdf",
        n_pages=4,
        info={
            "/Title": "Quarterly Report",
            "/Author": "AMA",
            "/CreationDate": "D:20260701090000Z",
        },
    )
    di = ingest(p)
    assert di.title == "Quarterly Report"
    assert di.author == "AMA"
    assert di.created == datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
    assert di.modified is None  # not set in /Info
    assert di.page_count == 4


def test_ingest_rejects_non_pdf_with_value_error(tmp_path):
    p = tmp_path / "notpdf.txt"
    p.write_bytes(b"this is just text, not a pdf")
    with pytest.raises(ValueError) as excinfo:
        ingest(p)
    assert "PDF" in str(excinfo.value) or "pdf" in str(excinfo.value).lower()


def test_ingest_rejects_missing_path(tmp_path):
    p = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        ingest(p)


def test_ingest_rejects_directory_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path)  # tmp_path is a directory, not a file


def test_ingest_accepts_string_path(tmp_path):
    p = _write_blank_pdf(tmp_path / "doc.pdf")
    di = ingest(str(p))  # str, not Path
    assert di.source_path == p


# ----------------------------------------------------------------------------
# DocumentInput.to_uir_source_metadata
# ----------------------------------------------------------------------------

def test_to_uir_source_metadata_with_full_info(tmp_path):
    p = _write_blank_pdf(
        tmp_path / "doc.pdf",
        n_pages=2,
        info={
            "/Title": "T",
            "/Author": "A",
            "/CreationDate": "D:20260101000000Z",
        },
    )
    di = ingest(p)
    source, metadata = di.to_uir_source_metadata()

    assert source.uri == di.uri
    assert source.format == "PDF"
    assert source.mime_type == PDF_MIME_TYPE
    assert source.size_bytes == di.size_bytes
    assert source.checksum == f"sha256:{di.sha256}"

    assert metadata.title == "T"
    assert metadata.author == "A"
    assert metadata.page_count == 2
    assert metadata.language == "en"  # default; Phase J enriches


def test_to_uir_source_metadata_uses_untitled_when_missing(tmp_path):
    """Title falls back to ``"(untitled)"`` per PLAN §9 Phase E."""
    p = _write_blank_pdf(tmp_path / "anon.pdf")
    di = ingest(p)
    _, metadata = di.to_uir_source_metadata()
    assert metadata.title == "(untitled)"
    assert metadata.author is None
    assert metadata.created is None
    assert metadata.modified is None


def test_to_uir_source_metadata_validates_against_schema(tmp_path):
    """The (Source, Metadata) pair round-trips cleanly through UIRV1."""
    from uir_pipeline.uir_schema import (
        ExtractionProvenance, NormalizationProvenance, Provenance,
        Semantics, Structure, StructureNode, UIRV1,
    )
    p = _write_blank_pdf(
        tmp_path / "roundtrip.pdf",
        n_pages=1,
        info={"/Title": "RTdoc", "/Author": "Tester"},
    )
    di = ingest(p)
    source, metadata = di.to_uir_source_metadata()

    # Build a minimal UIR tree with stub Structure + Semantics + Provenance
    # to verify the Source/Metadata pair round-trips through UIRV1.
    # The id must match NODE_ID_PATTERN (8-4-4-4-12 hex hyphenated UUID);
    # a zero-padded 36-char string is rejected because there are no hyphens.
    ts = datetime(2026, 7, 7, tzinfo=timezone.utc)
    doc_id = f"doc_{_uuid.uuid4()}"
    uir = UIRV1(
        uiR_version="1.0",
        id=doc_id,
        modal_type="document",
        source=source,
        metadata=metadata,
        structure=Structure(
            root=StructureNode(id=doc_id, type="document", children=[]),
        ),
        semantics=Semantics(),
        provenance=Provenance(
            extraction=ExtractionProvenance(
                model="LayoutLMv3", version="1.2.0", timestamp=ts,
            ),
            normalization=NormalizationProvenance(
                version="1.0", timestamp=ts,
            ),
        ),
    )
    # Validation already ran during construction; assert no extras leaked.
    assert uir.source.size_bytes == di.size_bytes
    assert uir.metadata.page_count == 1
    assert uir.metadata.title == "RTdoc"
