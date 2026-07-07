"""tables -- pdfplumber-backed table extraction (Phase H).

PLAN.md \u00a79 Phase H exit:
    -- detects tables on a fixture PDF page
    -- converts each detected table to markdown preserving the header row
    -- emits a structured ``TableDraft`` per table (bbox + markdown + counts)
    -- gracefully falls back to "no tables detected" when pdfplumber returns []

We deliberately use ``pdfplumber`` (per PLAN.md \u00a76) over ``camelot`` -- it
installs cleanly on macOS / Linux ROCm and emits bbox in PDF point
coordinates (post-Phase-I's normalization).

Note: we no longer pre-check file existence; pdfplumber raises on missing
files so the test fixture can inject stubs without being pre-empted by
``Path.is_file()``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Public types (frozen dataclass per PLAN.md \u00a75)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TableDraft:
    """A draft table discovered on a single PDF page.

    Bbox is on the 0-1000 virtual canvas (per PLAN.md \u00a78) -- the
    orchestrator threads it through ``StructureNode.bounding_box`` and
    ``ChunkNode.bounding_box``. ``markdown`` is the canonical
    GitHub-flavored markdown serialization (header row preserved).
    """

    page_number: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) on 0-1000 canvas
    markdown: str
    row_count: int
    col_count: int
    confidence: float  # 0-1; pdfplumber doesn't emit, so we use a heuristic.


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _table_confidence(rows: list[list[str]], cols: int) -> float:
    """Heuristic 0-1 score for ``table plausibility``.

    pdfplumber does not emit a confidence field, so we synthesize one
    from row regularity and column fill. The orchestrator surfaces this
    on the UIR table-node ``confidence`` field so downstream consumers
    can apply their own threshold.
    """
    if not rows or cols < 2:
        return 0.0
    filled = sum(1 for r in rows if len([c for c in r if c.strip()]) >= 2)
    fill_ratio = filled / len(rows)
    return round(min(1.0, 0.4 + 0.6 * fill_ratio), 3)


def _render_markdown(rows: list[list[str]]) -> str:
    """Convert a rectangular list-of-lists to GFM markdown.

    ``rows`` is a 2-D list of strings with at least one row. Returns
    an empty string if the table is empty. Cells are escaped only for
    ``|`` to keep markdown parsable.
    """
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    ncols = len(header)

    def cell(s: str) -> str:
        return (s or "").replace("|", r"\|").strip()

    out_lines = ["| " + " | ".join(cell(c) for c in header) + " |"]
    out_lines.append("|" + "|".join(["---"] * ncols) + "|")
    for row in body:
        if len(row) < ncols:
            row = list(row) + [""] * (ncols - len(row))
        elif len(row) > ncols:
            row = list(row)[:ncols]
        out_lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(out_lines)


def _normalize_table_bbox(
    pdf_points_bbox: tuple[float, float, float, float],
    page_w_pts: float,
    page_h_pts: float,
) -> tuple[int, int, int, int]:
    """Convert a pdfplumber bbox (x0, y_top, x1, y_bot) to the 0-1000 canvas.

    pdfplumber's y axis is top-origin (same as our 0-1000 canvas), so we
    do NOT flip y-coords. We just scale + canonical-order via min/max so
    either top-up or top-down input is tolerated.
    """
    from uir_pipeline.utils import BBOX_CANVAS
    x0, top, x1, bot = pdf_points_bbox

    def clamp_scale(c: float, dim: float) -> int:
        clamped = max(0.0, min(c, dim))
        return round((clamped * BBOX_CANVAS) / dim)

    nx0 = clamp_scale(x0, page_w_pts)
    nx1 = clamp_scale(x1, page_w_pts)
    nyt = clamp_scale(top, page_h_pts)
    nyb = clamp_scale(bot, page_h_pts)
    return (min(nx0, nx1), min(nyt, nyb), max(nx0, nx1), max(nyt, nyb))


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def extract_tables(
    pdf_path: str | Path,
    page_numbers: list[int] | None = None,
) -> list[TableDraft]:
    """Detect tables on ``pdf_path`` and return a list of :class:`TableDraft`.

    ``page_numbers`` is 1-based; ``None`` means "all pages with at least
    one candidate table." Each table's bbox is normalized to the 0-1000
    virtual canvas using the page's actual pdfplumber-derived dimensions.

    A missing/corrupt ``pdf_path`` surfaces as ``FileNotFoundError`` or
    pdfplumber's native exception -- we do not pre-check so that
    in-process test stubs that intercept ``pdfplumber.open`` get to see
    the call.
    """
    from pdfplumber import open as pdfplumber_open  # lazy

    p = Path(pdf_path)

    drafts: list[TableDraft] = []
    with pdfplumber_open(str(p)) as pdf:
        page_iter = (
            (n, pdf.pages[n - 1]) for n in page_numbers
            if 0 < n <= len(pdf.pages)
        ) if page_numbers else enumerate(pdf.pages, start=1)

        for pn, page in page_iter:
            try:
                tables = page.find_tables() or []
            except Exception as exc:
                logger.debug("page %d find_tables failed: %s", pn, exc)
                continue
            for table_obj in tables:
                rows = table_obj.extract() or []
                rows = [[c if c is not None else "" for c in row] for row in rows]
                if not rows:
                    continue
                ncols = len(rows[0])
                markdown = _render_markdown(rows)
                nx0, ny0, nx1, ny1 = _normalize_table_bbox(
                    table_obj.bbox, page.width, page.height,
                )
                drafts.append(TableDraft(
                    page_number=pn,
                    bbox=(nx0, ny0, nx1, ny1),
                    markdown=markdown,
                    row_count=len(rows),
                    col_count=ncols,
                    confidence=_table_confidence(rows, ncols),
                ))
    return drafts


__all__ = [
    "TableDraft",
    "extract_tables",
]
