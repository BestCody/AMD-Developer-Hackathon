"""scripts/generate_fixtures.py -- generate 3 fixture PDFs for Phase N.

Profiles (per PLAN.md \u00a79 Phase N):
    -- flat_text.pdf      single column, plain prose
    -- dense_table.pdf    4x5 financial table spanning 3+ rows x 3+ cols
    -- multi_column.pdf   2-column layout

We use reportlab to generate deterministic PDFs that exercise the
pipeline. Each fixture is small (< 5 pages) so the integration smoke
test runs quickly on a single PDF.

Usage:
    python scripts/generate_fixtures.py             # write all 3 fixtures
    python scripts/generate_fixtures.py flat_text  # write one
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running as a script.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _out_path(name: str) -> Path:
    """Resolve output path under tests/fixtures/sample_pdfs/."""
    p = _ROOT / "tests" / "fixtures" / "sample_pdfs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _make_doc(path: Path) -> "reportlab.platypus.SimpleDocTemplate":
    """Build a SimpleDocTemplate with letter page size + 0.75" margins."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate
    return SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )


# ----------------------------------------------------------------------------
# flat_text.pdf
# ----------------------------------------------------------------------------

def gen_flat_text() -> Path:
    """A single-column prose document; 2 pages of paragraphs."""
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, Spacer

    out = _out_path("flat_text.pdf")
    doc = _make_doc(out)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Quarterly Operations Report", styles["Title"]))
    story.append(Spacer(1, 12))
    paragraphs = [
        "This report covers the operational highlights and key performance "
        "indicators for the past quarter. The team has executed on the major "
        "milestones and remains on track for the next planning cycle.",
        "Engineering throughput improved by twelve percent over the prior "
        "quarter, driven by a small number of high-leverage process "
        "improvements across the build and release pipelines. Customer-facing "
        "latency dropped by seven percent on the slowest endpoints.",
        "Customer success escalations declined by twenty percent. The team "
        "attributed this to a more rigorous postmortem cadence and earlier "
        "detection of regression patterns across the product surface.",
        "Looking forward, the team plans to invest in observability and "
        "automated rollback tooling to further reduce customer impact from "
        "incidents. Hiring remains on plan and the on-call rotation is "
        "healthy with no uncovered shifts this quarter.",
    ]
    for p in paragraphs:
        story.append(Paragraph(p, styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    return out


# ----------------------------------------------------------------------------
# dense_table.pdf
# ----------------------------------------------------------------------------

def gen_dense_table() -> Path:
    """A 4x5 financial table embedded in light prose."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle,
    )

    out = _out_path("dense_table.pdf")
    doc = _make_doc(out)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Financial Summary, Q1-Q4", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "The following table summarizes revenue and expense categories across "
        "the four quarters of the fiscal year. All figures are in thousands.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 12))

    data = [
        ["Category", "Q1", "Q2", "Q3", "Q4"],
        ["Revenue",  "1,200", "1,340", "1,510", "1,640"],
        ["COGS",     "320",   "350",   "410",   "430"],
        ["Marketing","180",   "210",   "260",   "280"],
        ["R&D",      "410",   "440",   "480",   "510"],
    ]
    t = Table(data, colWidths=[1.4 * 72, 1.0 * 72, 1.0 * 72, 1.0 * 72, 1.0 * 72])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Q4 revenue grew 8% over Q3, driven by expansion in the existing "
        "customer base. COGS remained roughly flat as a percentage of revenue.",
        styles["BodyText"],
    ))
    doc.build(story)
    return out


# ----------------------------------------------------------------------------
# multi_column.pdf
# ----------------------------------------------------------------------------

def gen_multi_column() -> Path:
    """A 2-column layout using a BaseDocTemplate with two side-by-side Frames."""
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    )

    out = _out_path("multi_column.pdf")
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    title = styles["Title"]

    class TwoColumnDoc(BaseDocTemplate):
        def __init__(self, filename, **kwargs):
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            super().__init__(filename, pagesize=letter, **kwargs)
            page_w, page_h = letter
            gutter = 0.25 * inch
            col_w = (page_w - 1.5 * inch - gutter) / 2
            margin = 0.75 * inch
            self.addPageTemplates([
                PageTemplate(
                    id="two_col",
                    frames=[
                        Frame(margin, margin, col_w, page_h - 2 * margin,
                              id="left", showBoundary=0),
                        Frame(margin + col_w + gutter, margin, col_w, page_h - 2 * margin,
                              id="right", showBoundary=0),
                    ],
                ),
            ])

    doc = TwoColumnDoc(str(out))
    story = []
    story.append(Paragraph("Two-Column Document", title))
    story.append(Spacer(1, 12))
    long_para_1 = (
        "Left column begins here. This column carries the introductory "
        "context and primary analysis. The narrative flows top-down with "
        "clear paragraph breaks so the heuristic layout classifier can "
        "easily detect distinct regions. Lorem ipsum dolor sit amet, "
        "consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
        "labore et dolore magna aliqua. Ut enim ad minim veniam, quis "
        "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
        "consequat. Duis aute irure dolor in reprehenderit in voluptate velit "
        "esse cillum dolore eu fugiat nulla pariatur."
    )
    long_para_2 = (
        "Right column begins here and continues the discussion. The text is "
        "deliberately long enough to flow across multiple line wraps so the "
        "chunker can demonstrate the BGE-tokenizer-aware overlap stitching. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui "
        "officia deserunt mollit anim id est laborum. Sed ut perspiciatis "
        "unde omnis iste natus error sit voluptatem accusantium doloremque "
        "laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore "
        "veritatis et quasi architecto beatae vitae dicta sunt explicabo."
    )
    story.append(Paragraph(long_para_1, normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(long_para_2, normal))
    doc.build(story)
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    targets = {
        "flat_text": gen_flat_text,
        "dense_table": gen_dense_table,
        "multi_column": gen_multi_column,
    }
    if len(sys.argv) > 1:
        # Subset selection: ``python scripts/generate_fixtures.py flat_text``
        for name in sys.argv[1:]:
            if name not in targets:
                print(f"unknown fixture: {name}", file=sys.stderr)
                return 1
    for name, fn in targets.items():
        path = fn()
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
