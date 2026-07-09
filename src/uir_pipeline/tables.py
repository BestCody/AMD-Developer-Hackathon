"""tables -- TableDraft dataclass for the Docling fast path.

PLAN section H exit (legacy pdfplumber path retired):
    -- exposes :class:`TableDraft` so :func:`pipeline._docling_to_table_draft`
       can synthesize counts from a Docling markdown table.
    -- pgplumber-based ``extract_tables`` was removed when the pdfplumber
       fast path was retired. Table detection is now Docling's job via
       ``DoclingResult.tables`` (see :mod:`uir_pipeline.docling_extract`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableDraft:
    """A draft table discovered on a single PDF page.

    Bbox is on the 0-1000 virtual canvas (PLAN section 8) -- the
    orchestrator threads it through ``StructureNode.bounding_box`` and
    ``ChunkNode.bounding_box``. ``markdown`` is the canonical
    GitHub-flavored markdown serialization (header row preserved).
    """

    page_number: int
    bbox: tuple[int, int, int, int]
    markdown: str
    row_count: int
    col_count: int
    confidence: float


__all__ = ["TableDraft"]
