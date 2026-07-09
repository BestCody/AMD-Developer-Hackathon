"""uir_pipeline.uir_schema -- Pydantic v2 models for UIR v1.0.

This module is the SOURCE OF TRUTH for the UIR JSON contract.

It mirrors the `UIR Schema (Strict)` block in INSTRUCTIONS.md exactly.
Any drift from INSTRUCTIONS.md is a failing test in tests/test_uir_schema.py.

Design rationale: PLAN.md \u00a78 + \u00a79 Phase B.

Quick tour of the model tree::

    UIRV1
    \u251c\u2500 source       Source
    \u251c\u2500 metadata     Metadata
    \u251c\u2500 structure    Structure
    \u2502   \u2514\u2500 root    StructureNode
    \u2502       \u2514\u2500 children: list[StructureChild]
    \u2502           \u251c\u2500 StructureNode  (recursive, discriminated by `type`)
    \u2502           \u2514\u2500 ChunkNode
    \u251c\u2500 semantics    Semantics
    \u2502   \u251c\u2500 entities      list[Entity]
    \u2502   \u251c\u2500 relationships list[Relationship]
    \u2502   \u2514\u2500 topics        list[str]
    \u2514\u2500 provenance  Provenance
        \u251c\u2500 extraction    ExtractionProvenance (has model)
        \u2514\u2500 normalization NormalizationProvenance (no model)

Run as a module to emit JSON Schema to disk::

    python -m uir_pipeline.uir_schema docs/uir.schema.json
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Canonical UUID v4/v5 hex-char pattern (8-4-4-4-12 groups).
_UUID_HEX_SUFFIX = (
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Node IDs are deterministic strings of the form `<prefix>_<uuid>`; the prefix
# indicates the node's role in the hierarchy (PLAN.md \u00a78).
NODE_ID_PATTERN: str = (
    r"^(doc|section|table|figure|list|chunk|entity)_" + _UUID_HEX_SUFFIX + r"$"
)

# Bounding-box coordinates are normalized to a 0-1000 virtual canvas (PLAN.md
# \u00a78); downstream consumers can convert per their own page geometry.
_CANVAS_MIN = 0
_CANVAS_MAX = 1000

# `StructureNode.type` -> expected `<prefix>_` on the id (cross-field
# enforcement below). ``ChunkNode`` is omitted because its ``type`` is
# ``Literal["chunk"]``; the prefix check there hardcodes ``"chunk"``.
_STRUCTURE_TYPE_TO_PREFIX = {
    "document": "doc",
    "section": "section",
    "table": "table",
    "figure": "figure",
    "list": "list",
}


# ----------------------------------------------------------------------------
# Custom validators
# ----------------------------------------------------------------------------

def _validate_bbox(v: Any) -> tuple[int, int, int, int]:
    """Validate bounding-box spec format: [x1, y1, x2, y2] rectangle.

    - 4 integers (no floats, no booleans, no strings).
    - Canonical ordering: x1 <= x2 and y1 <= y2.
    - All coordinates within the 0-1000 virtual canvas (PLAN.md \u00a78).
    """
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise ValueError(
            f"bounding_box must be a 4-element list [x1, y1, x2, y2]; got {v!r}",
        )
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise ValueError(f"bounding_box items must be ints; got {v!r}")
    x1, y1, x2, y2 = v
    if x1 > x2 or y1 > y2:
        raise ValueError(
            f"bounding_box must satisfy x1 <= x2 and y1 <= y2; got {(x1, y1, x2, y2)}",
        )
    if (
        x1 < _CANVAS_MIN or y1 < _CANVAS_MIN
        or x2 > _CANVAS_MAX or y2 > _CANVAS_MAX
    ):
        raise ValueError(
            f"bounding_box coordinates must lie within [0, 1000] virtual canvas; "
            f"got {(x1, y1, x2, y2)}",
        )
    return (x1, y1, x2, y2)


# Public alias for downstream consumers and tests.
BoundingBox = Annotated[
    tuple[int, int, int, int],
    AfterValidator(_validate_bbox),
    Field(
        description=(
            "[x1, y1, x2, y2] rectangle normalized to a 0-1000 virtual canvas. "
            "Canonical ordering (x1 <= x2 and y1 <= y2)."
        ),
    ),
]


def _enforce_id_prefix(entity_id: str, expected_prefix: str, entity_type: str) -> str:
    """Pure helper: confirm ``entity_id`` has the prefix expected for ``entity_type``.

    Raises ``ValueError`` with a diagnostic message if not. ``entity_id`` must
    already match ``NODE_ID_PATTERN`` (the field-level validator handles that).
    """
    actual = entity_id.split("_", 1)[0]
    if actual != expected_prefix:
        raise ValueError(
            f"id prefix {actual!r} does not match type {entity_type!r} "
            f"(expected prefix {expected_prefix!r}); got id={entity_id!r}"
        )
    return entity_id


# ----------------------------------------------------------------------------
# Source / metadata
# ----------------------------------------------------------------------------

class Source(BaseModel):
    """Document source descriptor.

    ``format`` is widened from ``Literal["PDF"]`` to ``str`` (PLAN §17
    §Multi-format) so DOCX/PPTX/XLSX/HTML/EPUB/LaTeX/IPYNB/RTF/TXT/MD/CSV/
    image/code can flow through the same UIR contract without per-format
    schema churn. Per-format validation is delegated to ``format_router``
    which classifies the file at ingest time. ``route`` records the
    extraction route chosen for this document so downstream consumers
    can reconstruct provenance. Both new fields default to ``None`` /
    legacy ``"PDF"`` so pre-§17 v1 UIRs remain valid.
    """
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(..., description="Source URI (e.g. s3://, file://, https://).")
    format: str = Field(
        default="PDF",
        description=(
            "Format of the source document. Phase 1 was ``PDF`` only; "
            "Phase 17+ allows DOCX/PPTX/XLSX/HTML/EPUB/LaTeX/IPYNB/RTF/"
            "TXT/MD/CSV/IMAGE/code. Validation deferred to "
            "``format_router.classify_route`` at ingest."
        ),
    )
    route: str | None = Field(
        default=None,
        description=(
            "Extraction route dispatched by ``format_router``. One of "
            "``pdf``, ``docling``, ``text``, ``image``, ``skip``. "
            "Optional for backwards compatibility with pre-§17 UIRs."
        ),
    )
    mime_type: str = Field(..., description="IANA MIME type (e.g. application/pdf).")
    size_bytes: int = Field(..., ge=0, description="File size in bytes.")
    checksum: str = Field(
        ...,
        description="Algorithm-prefixed checksum (e.g. 'sha256:abc...').",
    )
    timestamp: datetime = Field(
        ..., description="ISO8601 ingest/processing timestamp.",
    )


class Metadata(BaseModel):
    """Document metadata (PDF + multi-format Phase 17+).

    ``page_count`` remains required (gte 0); for pageless formats
    (TXT/MD/code/CSV with no native page concept) the caller synthesises
    ``page_count=1`` via :func:`src.uir_pipeline.chunk.paginate_pageless`.
    ``format`` mirrors :attr:`Source.format` so JSON consumers can
    pivot on a single field without joining.
    """
    model_config = ConfigDict(extra="forbid")

    title: str
    author: str | None = None
    created: datetime | None = None
    modified: datetime | None = None
    page_count: int = Field(..., ge=0)
    language: str = Field(
        default="en", description="ISO 639-1 code (e.g. 'en').",
    )
    domain: str | None = Field(
        default=None,
        description="Inferred document domain (e.g. 'financial'), or null.",
    )
    format: str | None = Field(
        default=None,
        description=(
            "Mirror of ``Source.format`` for convenience. None when the "
            "UIR was emitted by pre-§17 code paths (legacy PDFs only)."
        ),
    )


# ----------------------------------------------------------------------------
# Structure (recursive hierarchy)
# ----------------------------------------------------------------------------

class StructureNode(BaseModel):
    """A non-leaf node in the document hierarchy.

    Cross-field validation: ``id`` must begin with the conventional prefix for
    this node's ``type`` (e.g. ``type=document`` requires ``id="doc_<uuid>"``).
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=NODE_ID_PATTERN)
    type: Literal["document", "section", "table", "figure", "list"]
    title: str | None = None
    page: int | None = Field(default=None, ge=0)
    bounding_box: BoundingBox | None = None
    children: list["StructureChild"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _id_prefix_matches_type(self) -> "StructureNode":
        expected = _STRUCTURE_TYPE_TO_PREFIX[self.type]
        _enforce_id_prefix(self.id, expected, self.type)
        return self


class ChunkNode(BaseModel):
    """A leaf text chunk in the document hierarchy."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=NODE_ID_PATTERN)
    type: Literal["chunk"]
    text: str
    token_count: int = Field(..., ge=0)
    page: int = Field(..., ge=0)
    bounding_box: BoundingBox | None = None
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Per-chunk extraction confidence [0, 1].",
    )
    modal_features: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Free-form per-modality feature dict. Phase 2 adds audio/video "
            "entries here without schema churn."
        ),
    )

    @model_validator(mode="after")
    def _id_prefix_matches_type(self) -> "ChunkNode":
        # `self.type` is bounded by Literal["chunk"] at parse time; only
        # the id prefix needs cross-field enforcement.
        _enforce_id_prefix(self.id, "chunk", "chunk")
        return self


# Children of StructureNode can be either a nested StructureNode or a leaf
# ChunkNode. Discriminated union via the `type` literal field.
StructureChild = Annotated[
    Union[StructureNode, ChunkNode],
    Field(discriminator="type"),
]


# Required after the forward-ref definition so children references resolve.
StructureNode.model_rebuild()


class Structure(BaseModel):
    """The hierarchical structural tree."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["hierarchical"] = "hierarchical"
    root: StructureNode


# ----------------------------------------------------------------------------
# Semantics
# ----------------------------------------------------------------------------

class Entity(BaseModel):
    """A named entity extracted from the document.

    NOTE: entities are referenced from relationships by id (see
    ``Relationship.from_id``/``to_id``). Entity objects themselves do not
    carry an ``id`` here per INSTRUCTIONS.md; downstream code maintains the
    id mapping outside the schema. Cross-referential integrity is therefore
    not enforced by the schema (and is a v1-deferred validation per
    PLAN.md \u00a79 Phase E).
    """
    model_config = ConfigDict(extra="forbid")

    text: str
    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class Relationship(BaseModel):
    """A relationship between two entities (referenced by id or by name).

    The JSON key `from` is aliased to Python attribute `from_` because
    `from` is a Python reserved word.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(..., alias="from", description="Source entity id or name.")
    to: str = Field(..., description="Target entity id or name.")
    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class Semantics(BaseModel):
    """Document-level semantic enrichment (entities, relationships, topics)."""
    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------

class ExtractionProvenance(BaseModel):
    """Provenance for the extraction step (identifies the upstream model)."""
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Upstream model name (e.g. 'LayoutLMv3').")
    version: str = Field(..., description="Upstream model version (e.g. '1.2.0').")
    timestamp: datetime


class NormalizationProvenance(BaseModel):
    """Provenance for the UIR normalization step. No upstream model id."""
    model_config = ConfigDict(extra="forbid")

    version: str = Field(
        ..., description="UIR normalization schema version (e.g. '1.0').",
    )
    timestamp: datetime


class Provenance(BaseModel):
    """Pipeline-run provenance (extraction + normalization)."""
    model_config = ConfigDict(extra="forbid")

    extraction: ExtractionProvenance
    normalization: NormalizationProvenance


# ----------------------------------------------------------------------------
# Top-level UIR v1.0
# ----------------------------------------------------------------------------

class UIRV1(BaseModel):
    """Universal Intermediate Representation v1.0 (Phase 1: PDF documents).

    Top-level discriminator: ``modal_type``. Phase 1 carries only
    ``modal_type=document``; Phase 2+ adds audio/video/image variants.
    """
    model_config = ConfigDict(extra="forbid")

    uiR_version: Literal["1.0"]
    id: str = Field(
        ...,
        description=(
            "Top-level UIR id. Conventionally a UUID5 derived from the source "
            "URI per PLAN.md \u00a78. The schema does not enforce a particular "
            "shape here to allow upstream UUID library upgrades."
        ),
    )
    modal_type: Literal["document", "image"] = "document"
    source: Source
    metadata: Metadata
    structure: Structure
    semantics: Semantics
    provenance: Provenance


# ----------------------------------------------------------------------------
# JSON Schema export helper
# ----------------------------------------------------------------------------

def schema_json_dict() -> dict[str, Any]:
    """Return UIRV1's JSON Schema as a Python dict.

    Downstream documentation tooling (Sphinx, OpenAPI, etc.) can consume this.
    Use ``scripts/export_uir_json_schema.py`` for the on-disk write.
    """
    return UIRV1.model_json_schema()


__all__ = [
    "BoundingBox",
    "ChunkNode",
    "Entity",
    "ExtractionProvenance",
    "Metadata",
    "NODE_ID_PATTERN",
    "NormalizationProvenance",
    "Provenance",
    "Relationship",
    "Semantics",
    "Source",
    "Structure",
    "StructureChild",
    "StructureNode",
    "UIRV1",
    "_validate_bbox",
    "schema_json_dict",
]


def main() -> None:
    """CLI entrypoint: write the UIR JSON Schema to ``docs/uir.schema.json``.
    
    Usage: ``python -m uir_pipeline.uir_schema`` or with an explicit path:
    ``python -m uir_pipeline.uir_schema some/other/path.json``.

    Resolves the default output path relative to this file so the script
    works regardless of CWD.
    """
    default_out = Path(__file__).resolve().parent.parent / "docs" / "uir.schema.json"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema_json_dict(), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
