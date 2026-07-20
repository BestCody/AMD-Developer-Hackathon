"""Generate the two synthetic stress-test PDFs described in issue.tmp.

These are intentionally *broken* documents that reproduce the two extraction
bugs, so they double as regression fixtures. They are generated with reportlab
(+ PIL for the rasterized scan) and written under tests/fixtures/sample_pdfs
(those PDFs are gitignored by design, so they are regenerated locally rather
than committed). The guarded end-to-end tests in tests/test_docling_extract.py
re-run the pipeline against them after a fix.

    tests/fixtures/sample_pdfs/01_messy_multicolumn_report.pdf
        Two-column body text with a large, rotated, semi-transparent
        "DRAFT - DO NOT DISTRIBUTE" stamp overlapping the middle of
        section 3. Reproduces Bug 2 (decorative / watermark text
        leaking into the content stream).

    tests/fixtures/sample_pdfs/02_scan_simulated_invoice.pdf
        An invoice whose *only* content is a rasterized image of a
        table (no embedded text layer). Reproduces Bug 1: pypdfium
        finds ~0 glyphs, so the pipeline routes it through OCR, and
        the OCR'd table's UNIT/EXT columns arrive mis-aligned.

NOTE: these PDFs are NOT committed -- ``tests/fixtures/sample_pdfs/*`` is
gitignored by design (see ``.gitignore``, "Real PDFs ... pulled separately
per fixture"). This script regenerates them locally so the guarded
end-to-end tests in ``tests/test_docling_extract.py`` have something to run
against; without the PDFs those tests skip gracefully.

Run:  python3 scripts/generate_issue_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "fixtures" / "sample_pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BODY = (
    "The quick brown fox jumps over the lazy dog while the system ingests "
    "documents at scale. Layout models must recover reading order even when "
    "columns interleave and figures interrupt the prose. Retrieval quality "
    "depends on every paragraph landing in the chunk stream exactly once."
)


def _section(out: "canvas.Canvas", title: str, y: float) -> None:
    out.setFont("Helvetica-Bold", 14)
    out.drawString(0.9 * inch, y, title)
    out.setFont("Helvetica", 10)


def _two_column_body(
    out: "canvas.Canvas", text: str, top: float, bottom: float,
) -> None:
    """Flow ``text`` across two stacked text columns (left then right)."""
    col_w = 3.1 * inch
    gap = 0.2 * inch
    left_x = 0.9 * inch
    right_x = left_x + col_w + gap
    out.setFont("Helvetica", 10)
    for x in (left_x, right_x):
        y = top
        for word in text.split():
            if y < bottom:
                break
            out.drawString(x, y, word)
            y -= 0.18 * inch


def build_multicolumn_report() -> Path:
    """Two-column report with a rotated DRAFT watermark over section 3."""
    path = OUT_DIR / "01_messy_multicolumn_report.pdf"
    out = canvas.Canvas(str(path), pagesize=letter)

    # Section 1 + 2 on the upper half.
    _section(out, "1. Introduction", 10.2 * inch)
    _two_column_body(
        out, BODY * 3, top=9.9 * inch, bottom=6.6 * inch,
    )
    _section(out, "2. Related Work", 6.3 * inch)
    _two_column_body(
        out, BODY * 3, top=6.0 * inch, bottom=3.4 * inch,
    )

    # Section 3 -- the watermark overlaps the body text in the middle.
    _section(out, "3. Method", 3.1 * inch)
    _two_column_body(
        out, BODY * 4, top=2.8 * inch, bottom=1.0 * inch,
    )

    # Large rotated semi-transparent stamp in the MIDDLE of section 3.
    out.saveState()
    out.setFillAlpha(0.35)
    out.setStrokeAlpha(0.35)
    out.setFillColorRGB(0.6, 0.0, 0.0)
    out.setFont("Helvetica-Bold", 54)
    # Translate to page centre, rotate 30 degrees, then draw centred.
    out.translate(4.25 * inch, 2.6 * inch)
    out.rotate(30)
    out.drawCentredString(0, 0, "DRAFT - DO NOT DISTRIBUTE")
    out.restoreState()

    out.showPage()
    out.save()
    return path


def build_scan_invoice() -> Path:
    """Invoice rendered as a raster image only (image-only / simulated scan).

    The page carries no text layer, so a non-OCR conversion yields ~0 glyphs
    and the pipeline re-routes through OCR -- exercising Bug 1's path.
    """
    from PIL import Image, ImageDraw, ImageFont

    path = OUT_DIR / "02_scan_simulated_invoice.pdf"
    W, H = 1700, 2200
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 34
        )
        small = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 28
        )
    except Exception:
        font = ImageFont.load_default()
        small = font

    draw.text((80, 60), "ACME CORP - INVOICE", fill="black", font=font)
    draw.line((80, 120, W - 80, 120), fill="black", width=3)

    headers = ["SKU", "Description", "QTY", "UNIT $", "EXT $"]
    rows = [
        ["A1", "Widget", "2", "10.00", "20.00"],
        ["A2", "Gadget", "3", "15.00", "45.00"],
        ["B9", "Cable", "5", "2.50", "12.50"],
        ["C4", "Bracket", "1", "8.00", "8.00"],
    ]
    x0, y = 120, 200
    col_x = [x0 + i * 300 for i in range(5)]
    for cx, h in zip(col_x, headers):
        draw.text((cx, y), h, fill="black", font=small)
    y += 50
    draw.line((80, y - 20, W - 80, y - 20), fill="black", width=2)
    for r in rows:
        for cx, cell in zip(col_x, r):
            draw.text((cx, y), cell, fill="black", font=small)
        y += 50

    # Render to PDF as a single full-page image (no text layer).
    out = canvas.Canvas(str(path), pagesize=letter)
    img_path = OUT_DIR / "_invoice_tmp.png"
    img.save(img_path)
    out.drawImage(
        str(img_path), 0, 0, width=letter[0], height=letter[1],
        preserveAspectRatio=False,
    )
    out.showPage()
    out.save()
    img_path.unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    p1 = build_multicolumn_report()
    p2 = build_scan_invoice()
    print("wrote", p1)
    print("wrote", p2)
