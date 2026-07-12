"""Tests for src.uir_pipeline.format_router (PLAN §17 §Multi-format)."""
from __future__ import annotations

from pathlib import Path


from uir_pipeline.format_router import (
    FormatRoute,
    SUPPORTED_EXTENSIONS,
    classify_route,
    detect_format,
    route,
    source_format_label,
)


# ----------------------------------------------------------------------------
# detect_format: magic-byte paths
# ----------------------------------------------------------------------------

def test_detect_format_pdf(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    assert detect_format(p) == "PDF"


def test_detect_format_pdf_2_0(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-2.0\n")
    assert detect_format(p) == "PDF"


def test_detect_format_docx(tmp_path: Path):
    p = tmp_path / "doc.docx"
    # Write a minimal OOXML zip with [Content_Types].xml + word/ dir.
    import zipfile
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("[Content_Types].xml", "<?xml version='1.0'?>")
        z.writestr("word/document.xml", "<doc/>")
    assert detect_format(p) == "DOCX"


def test_detect_format_pptx(tmp_path: Path):
    p = tmp_path / "deck.pptx"
    import zipfile
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("[Content_Types].xml", "<x/>")
        z.writestr("ppt/slides/slide1.xml", "<s/>")
    assert detect_format(p) == "PPTX"


def test_detect_format_xlsx(tmp_path: Path):
    p = tmp_path / "sheet.xlsx"
    import zipfile
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("[Content_Types].xml", "<x/>")
        z.writestr("xl/worksheets/sheet1.xml", "<w/>")
    assert detect_format(p) == "XLSX"


def test_detect_format_epub(tmp_path: Path):
    p = tmp_path / "book.epub"
    import zipfile
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("META-INF/container.xml", "<c/>")
        z.writestr("OEBPS/content.xhtml", "<h/>")
    assert detect_format(p) == "EPUB"


def test_detect_format_rtf(tmp_path: Path):
    p = tmp_path / "doc.rtf"
    p.write_bytes(b"{\\rtf1\\ansi Hello world}")
    assert detect_format(p) == "RTF"


# ----------------------------------------------------------------------------
# detect_format: extension fallback paths
# ----------------------------------------------------------------------------

def test_detect_format_md_extension(tmp_path: Path):
    p = tmp_path / "notes.md"
    p.write_text("# Heading\n\nBody", encoding="utf-8")
    assert detect_format(p) == "MD"


def test_detect_format_txt_extension(tmp_path: Path):
    p = tmp_path / "notes.txt"
    p.write_text("Body line one.\n\nBody line two.", encoding="utf-8")
    assert detect_format(p) == "TXT"


def test_detect_format_csv_extension(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2", encoding="utf-8")
    assert detect_format(p) == "CSV"


def test_detect_format_code_extension(tmp_path: Path):
    p = tmp_path / "module.py"
    p.write_text("def hello():\n    return 1", encoding="utf-8")
    assert detect_format(p) == "PY"


def test_detect_format_image_extension(tmp_path: Path):
    p = tmp_path / "img.PNG"  # case-insensitive suffix
    p.write_bytes(b"\x89PNG\r\n\x1a\n")  # real PNG magic wins over suffix alone
    assert detect_format(p) == "IMAGE" or detect_format(p) == "PNG"
    # PNG magic is also valid for the IMAGE branch.


def test_detect_format_unknown_returns_empty(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\xff\xab\xcd mystery")
    assert detect_format(p) == ""


def test_detect_format_missing_file(tmp_path: Path):
    """Missing file returns ``""`` (caller routes to SKIP)."""
    p = tmp_path / "does-not-exist.pdf"
    assert detect_format(p) == ""


# ----------------------------------------------------------------------------
# classify_route
# ----------------------------------------------------------------------------

def test_classify_route_pdf():
    assert classify_route("PDF") == FormatRoute.PDF


def test_classify_route_docx_to_docling():
    assert classify_route("DOCX") == FormatRoute.DOCLING


def test_classify_route_pptx_to_pptx_native():
    assert classify_route("PPTX") == FormatRoute.PPTX_NATIVE


def test_classify_route_html_to_docling():
    assert classify_route("HTML") == FormatRoute.DOCLING


def test_classify_route_md_to_text():
    assert classify_route("MD") == FormatRoute.TEXT


def test_classify_route_txt_to_text():
    assert classify_route("TXT") == FormatRoute.TEXT


def test_classify_route_py_to_text():
    assert classify_route("PY") == FormatRoute.TEXT


def test_classify_route_rtf_to_text():
    assert classify_route("RTF") == FormatRoute.TEXT


def test_classify_route_png_to_image():
    assert classify_route("PNG") == FormatRoute.IMAGE


def test_classify_route_empty_to_skip():
    assert classify_route("") == FormatRoute.SKIP


def test_classify_route_unknown_to_skip():
    assert classify_route("BLAHBLAH") == FormatRoute.SKIP


# ----------------------------------------------------------------------------
# route convenience
# ----------------------------------------------------------------------------

def test_route_pdf(tmp_path: Path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    fmt, r = route(p)
    assert fmt == "PDF"
    assert r == FormatRoute.PDF


def test_route_docx(tmp_path: Path):
    import zipfile
    p = tmp_path / "x.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("[Content_Types].xml", "<x/>")
        z.writestr("word/document.xml", "<d/>")
    fmt, r = route(p)
    assert fmt == "DOCX"
    assert r == FormatRoute.DOCLING


# ----------------------------------------------------------------------------
# source_format_label
# ----------------------------------------------------------------------------

def test_source_format_label_pdf_passes_through():
    assert source_format_label("PDF") == "PDF"


def test_source_format_label_uppercases():
    assert source_format_label("docx") == "DOCX"


def test_source_format_label_empty_returns_unknown():
    assert source_format_label("") == "UNKNOWN"


# ----------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS
# ----------------------------------------------------------------------------

def test_supported_extensions_includes_many():
    """Sanity-check that the CLI rglob set covers the obvious formats."""
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".pptx" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS
    assert ".html" in SUPPORTED_EXTENSIONS
    assert ".epub" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS
    assert ".rtf" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".ipynb" in SUPPORTED_EXTENSIONS
    assert ".tex" in SUPPORTED_EXTENSIONS
    assert ".py" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# CONVERTIBLE_EXTENSIONS
# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS names every extension the router *recognises*, including
# the legacy binary Office formats it classifies SKIP. Gating uploads on it
# accepted a .doc and then failed the job seconds later inside ingest_any.

def test_convertible_is_a_subset_of_supported():
    from uir_pipeline.format_router import CONVERTIBLE_EXTENSIONS, SUPPORTED_EXTENSIONS

    assert CONVERTIBLE_EXTENSIONS < SUPPORTED_EXTENSIONS


def test_convertible_excludes_exactly_the_legacy_binary_office_formats():
    from uir_pipeline.format_router import CONVERTIBLE_EXTENSIONS, SUPPORTED_EXTENSIONS

    assert sorted(SUPPORTED_EXTENSIONS - CONVERTIBLE_EXTENSIONS) == [".doc", ".ppt", ".xls"]


def test_no_convertible_extension_classifies_to_skip():
    from uir_pipeline.format_router import (
        CONVERTIBLE_EXTENSIONS,
        FormatRoute,
        classify_route,
    )

    for ext in CONVERTIBLE_EXTENSIONS:
        route = classify_route(ext.lstrip(".").upper())
        assert route is not FormatRoute.SKIP, f"{ext} -> SKIP but marked convertible"


def test_the_headline_formats_are_convertible():
    from uir_pipeline.format_router import CONVERTIBLE_EXTENSIONS

    for ext in (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv",
                ".rtf", ".py", ".ipynb", ".html", ".tex", ".png"):
        assert ext in CONVERTIBLE_EXTENSIONS, ext


# ----------------------------------------------------------------------------
# Audio format routing
# ----------------------------------------------------------------------------

def test_detect_format_audio_extensions(tmp_path: Path):
    for ext in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"):
        p = tmp_path / f"test{ext}"
        p.write_text("not real audio data")
        assert detect_format(p) == "AUDIO", f"failed for {ext}"


def test_classify_route_audio_to_audio():
    assert classify_route("AUDIO") == FormatRoute.AUDIO
    assert classify_route("MP3") == FormatRoute.AUDIO
    assert classify_route("WAV") == FormatRoute.AUDIO
    assert classify_route("M4A") == FormatRoute.AUDIO
    assert classify_route("FLAC") == FormatRoute.AUDIO
    assert classify_route("OGG") == FormatRoute.AUDIO
    assert classify_route("AAC") == FormatRoute.AUDIO
    assert classify_route("WMA") == FormatRoute.AUDIO


def test_route_audio_mp3(tmp_path: Path):
    p = tmp_path / "song.mp3"
    p.write_text("not real audio")
    fmt, r = route(p)
    assert fmt == "AUDIO"
    assert r == FormatRoute.AUDIO


def test_supported_extensions_includes_audio():
    from uir_pipeline.format_router import SUPPORTED_EXTENSIONS

    for ext in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"):
        assert ext in SUPPORTED_EXTENSIONS, f"missing {ext}"


def test_convertible_extensions_includes_audio():
    from uir_pipeline.format_router import CONVERTIBLE_EXTENSIONS

    for ext in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"):
        assert ext in CONVERTIBLE_EXTENSIONS, f"missing {ext}"
