"""tests/test_uir_schema.py -- UIR v1.0 Pydantic schema validation.

These tests are the contract for the UIR JSON contract. They must pass.

Per INSTRUCTIONS.md (`UIR Schema (Strict)`):
  - The example JSON shape round-trips through UIRV1.model_validate_json.
  - Free-form fields (modal_features) accept additional modality entries.
  - Anything beyond spec is rejected (extra='forbid').

Per PLAN.md \u00a79 Phase B exit:
  - schema loads
  - pydantic.UIRV1.model_validate_json(spec_example) succeeds with the spec example
  - pytest tests/test_uir_schema.py is green
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from uir_pipeline.uir_schema import (
    BoundingBox,
    ChunkNode,
    Entity,
    Metadata,
    Provenance,
    Relationship,
    Semantics,
    Source,
    Structure,
    StructureNode,
    UIRV1,
    _validate_bbox,
    schema_json_dict,
)


# Test UUID namespace (fixed for deterministic fixtures).
_TEST_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _uuid5(*parts: str) -> str:
    """uuid5 with a fixed namespace; deterministic for the same inputs."""
    return str(uuid.uuid5(_TEST_NS, "::".join(parts)))


@pytest.fixture
def spec_example_dict() -> dict[str, Any]:
    """The INSTRUCTIONS.md spec example with placeholder UUIDs replaced by
    real UUID5s that satisfy the regex validator."""
    return {
        "uiR_version": "1.0",
        "id": _uuid5("top-doc"),
        "modal_type": "document",
        "source": {
            "uri": "s3://bucket/file.pdf",
            "format": "PDF",
            "mime_type": "application/pdf",
            "size_bytes": 2450000,
            "checksum": "sha256:" + "0" * 64,
            "timestamp": "2026-07-07T00:00:00Z",
        },
        "metadata": {
            "title": "Q2 Earnings Report",
            "author": "Jane Doe",
            "created": "2026-04-01T09:00:00Z",
            "modified": "2026-07-01T10:00:00Z",
            "page_count": 10,
            "language": "en",
            "domain": "financial",
        },
        "structure": {
            "type": "hierarchical",
            "root": {
                "id": "doc_" + _uuid5("doc"),
                "type": "document",
                "title": "Q2 Earnings",
                "children": [
                    {
                        "id": "section_" + _uuid5("doc", "section"),
                        "type": "section",
                        "title": "Executive Summary",
                        "page": 1,
                        "bounding_box": [10, 100, 500, 200],
                        "children": [
                            {
                                "id": "chunk_" + _uuid5("doc", "section", "chunk"),
                                "type": "chunk",
                                "text": "Revenue grew 12% YoY.",
                                "token_count": 8,
                                "page": 1,
                                "bounding_box": [10, 100, 500, 200],
                                "confidence": 0.95,
                                "modal_features": {
                                    "text": {"quality": 0.98},
                                    "layout": {"type": "paragraph", "reading_order": 1},
                                },
                            },
                        ],
                    },
                ],
            },
        },
        "semantics": {
            "entities": [
                {"text": "revenue", "type": "financial_metric", "confidence": 0.92},
                {"text": "margin", "type": "financial_metric", "confidence": 0.88},
            ],
            "relationships": [
                {
                    "from": "entity_" + _uuid5("ent", "revenue"),
                    "to": "entity_" + _uuid5("ent", "margin"),
                    "type": "correlated_with",
                    "confidence": 0.78,
                },
            ],
            "topics": ["earnings", "growth"],
        },
        "provenance": {
            "extraction": {
                "model": "LayoutLMv3",
                "version": "1.2.0",
                "timestamp": "2026-07-07T00:00:00Z",
            },
            "normalization": {
                "version": "1.0",
                "timestamp": "2026-07-07T00:00:00Z",
            },
        },
    }


# ----------------------------------------------------------------------------
# Round-trip & conformance tests
# ----------------------------------------------------------------------------

def test_schema_loads():
    # Smoke-importable.
    from uir_pipeline import uir_schema  # noqa: F401


def test_spec_example_validates(spec_example_dict):
    json_str = json.dumps(spec_example_dict)
    uir = UIRV1.model_validate_json(json_str)
    assert uir.uiR_version == "1.0"
    assert uir.modal_type == "document"
    assert uir.id == spec_example_dict["id"]


def test_spec_example_round_trips_through_dump(spec_example_dict):
    uir = UIRV1.model_validate_json(json.dumps(spec_example_dict))
    dumped = uir.model_dump(mode="json", exclude_none=False)
    reloaded = UIRV1.model_validate(dumped)
    assert reloaded.model_dump(mode="json") == dumped


# ----------------------------------------------------------------------------
# Negative tests: spec violations must fail validation
# ----------------------------------------------------------------------------

def test_unknown_top_level_field_rejected(spec_example_dict):
    spec_example_dict["extra_top_level_field"] = "fail"
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_wrong_uir_version_rejected(spec_example_dict):
    spec_example_dict["uiR_version"] = "0.9"
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_wrong_modal_type_rejected(spec_example_dict):
    spec_example_dict["modal_type"] = "image"
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_bad_doc_node_id_prefix_rejected(spec_example_dict):
    """document-type nodes must have id starting with `doc_`."""
    bad = "section_" + spec_example_dict["structure"]["root"]["id"].split("_", 1)[1]
    spec_example_dict["structure"]["root"]["id"] = bad
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_bad_chunk_id_prefix_rejected(spec_example_dict):
    chunk_id = spec_example_dict["structure"]["root"]["children"][0]["children"][0]["id"]
    spec_example_dict["structure"]["root"]["children"][0]["children"][0]["id"] = (
        "section_" + chunk_id.split("_", 1)[1]
    )
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_malformed_uuid_suffix_rejected(spec_example_dict):
    spec_example_dict["structure"]["root"]["id"] = "doc_not-a-uuid"
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


# ----------------------------------------------------------------------------
# Bounding-box tests
# ----------------------------------------------------------------------------

def test_bbox_must_have_four_elements():
    with pytest.raises(ValueError):
        _validate_bbox([1, 2, 3])
    with pytest.raises(ValueError):
        _validate_bbox([1, 2, 3, 4, 5])


def test_bbox_items_must_be_ints():
    with pytest.raises(ValueError):
        _validate_bbox([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        _validate_bbox(["1", "2", "3", "4"])
    with pytest.raises(ValueError):
        _validate_bbox([True, False, True, False])  # bool is rejected


def test_bbox_ordering_must_be_canonical():
    with pytest.raises(ValueError):
        _validate_bbox([100, 0, 50, 200])  # x1 > x2
    with pytest.raises(ValueError):
        _validate_bbox([0, 200, 100, 50])  # y1 > y2


def test_bbox_accepts_canonical_form():
    assert _validate_bbox([0, 0, 100, 200]) == (0, 0, 100, 200)
    assert _validate_bbox([10, 100, 500, 200]) == (10, 100, 500, 200)


def test_bbox_rejects_out_of_canvas():
    """Coordinates outside the 0-1000 virtual canvas (PLAN.md \u00a78) must reject."""
    with pytest.raises(ValueError):
        _validate_bbox([100, 100, 1100, 200])  # x2 > 1000
    with pytest.raises(ValueError):
        _validate_bbox([100, 100, 500, 1200])  # y2 > 1000
    with pytest.raises(ValueError):
        _validate_bbox([-1, 100, 100, 200])   # x1 < 0
    with pytest.raises(ValueError):
        _validate_bbox([100, -50, 500, 200])  # y1 < 0


def test_bbox_accepts_canvas_boundary():
    """The 0 and 1000 boundaries are inclusive; degenerate boxes are canonical."""
    assert _validate_bbox([0, 0, 0, 0]) == (0, 0, 0, 0)            # zero-size origin
    assert _validate_bbox([1000, 0, 1000, 1000]) == (1000, 0, 1000, 1000)  # max corner
    assert _validate_bbox([100, 100, 500, 100]) == (100, 100, 500, 100)  # y1 == y2
    assert _validate_bbox([500, 200, 500, 600]) == (500, 200, 500, 600)  # x1 == x2


def test_bbox_invalid_in_chunk_spec(spec_example_dict):
    spec_example_dict["structure"]["root"]["children"][0]["children"][0][
        "bounding_box"
    ] = [200, 0, 100, 100]  # x1 > x2
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_bbox_out_of_canvas_in_chunk_spec(spec_example_dict):
    """BBox out-of-canvas must reject at UIRV1 parse time, not just at unit test."""
    spec_example_dict["structure"]["root"]["children"][0]["children"][0][
        "bounding_box"
    ] = [100, 100, 1100, 200]  # x2 > 1000
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


# ----------------------------------------------------------------------------
# Confidence tests
# ----------------------------------------------------------------------------

def test_chunk_confidence_accepts_endpoints():
    ChunkNode.model_validate({
        "id": "chunk_" + _uuid5("c", "0"),
        "type": "chunk",
        "text": "",
        "token_count": 0,
        "page": 0,
        "confidence": 0.0,
    })
    ChunkNode.model_validate({
        "id": "chunk_" + _uuid5("c", "1"),
        "type": "chunk",
        "text": "",
        "token_count": 0,
        "page": 0,
        "confidence": 1.0,
    })


def test_chunk_confidence_rejects_below_zero(spec_example_dict):
    spec_example_dict["structure"]["root"]["children"][0]["children"][0][
        "confidence"
    ] = -0.01
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_chunk_confidence_rejects_above_one(spec_example_dict):
    spec_example_dict["structure"]["root"]["children"][0]["children"][0][
        "confidence"
    ] = 1.5
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_entity_confidence_is_bounded(spec_example_dict):
    spec_example_dict["semantics"]["entities"][0]["confidence"] = 2.0
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


# ----------------------------------------------------------------------------
# modal_features free-form expansion
# ----------------------------------------------------------------------------

def test_modal_features_accepts_arbitrary_keys(spec_example_dict):
    """Phase 2 expansion: text + layout + audio + video all coexist."""
    spec_example_dict["structure"]["root"]["children"][0]["children"][0][
        "modal_features"
    ] = {
        "text": {"quality": 0.98},
        "layout": {"type": "paragraph", "reading_order": 1},
        "audio": {"pitch": 0.5, "duration_s": 12.0},
        "video": {"fps": 30},
    }
    uir = UIRV1.model_validate_json(json.dumps(spec_example_dict))
    cf = uir.structure.root.children[0].children[0]
    assert cf.modal_features["audio"]["pitch"] == 0.5
    assert cf.modal_features["video"]["fps"] == 30


# ----------------------------------------------------------------------------
# Relationship "from" alias
# ----------------------------------------------------------------------------

def test_relationship_from_alias_in_spec(spec_example_dict):
    uir = UIRV1.model_validate_json(json.dumps(spec_example_dict))
    rel = uir.semantics.relationships[0]
    # JSON `from` key maps to Python's `from_` attribute.
    assert rel.from_ == spec_example_dict["semantics"]["relationships"][0]["from"]


def test_relationship_round_trip_preserves_from_key():
    rel = Relationship.model_validate({
        "from": "entity_x",
        "to": "entity_y",
        "type": "rel",
        "confidence": 0.5,
    })
    dumped = rel.model_dump(mode="json", by_alias=True)
    assert dumped["from"] == "entity_x"


def test_relationship_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Relationship.model_validate({
            "from": "a", "to": "b", "type": "r", "confidence": 0.5,
            "extra_unknown": "fail",
        })


# ----------------------------------------------------------------------------
# Datetime / ISO8601
# ----------------------------------------------------------------------------

def test_iso8601_zulu_accepted(spec_example_dict):
    uir = UIRV1.model_validate_json(json.dumps(spec_example_dict))
    ts = uir.source.timestamp
    assert ts.tzinfo is not None
    assert (ts.year, ts.month, ts.day) == (2026, 7, 7)


def test_invalid_datetime_rejected(spec_example_dict):
    spec_example_dict["source"]["timestamp"] = "not-a-date"
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


# ----------------------------------------------------------------------------
# Provenance strictness
# ----------------------------------------------------------------------------

def test_provenance_extraction_requires_model(spec_example_dict):
    spec_example_dict["provenance"]["extraction"].pop("model")
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


def test_provenance_normalization_must_not_have_model(spec_example_dict):
    """NormalizationProvenance rejects extra fields via extra='forbid'."""
    spec_example_dict["provenance"]["normalization"]["model"] = "trait"
    with pytest.raises(ValidationError):
        UIRV1.model_validate_json(json.dumps(spec_example_dict))


# ----------------------------------------------------------------------------
# JSON Schema export
# ----------------------------------------------------------------------------

def test_schema_json_dict_returns_a_schema():
    schema = schema_json_dict()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "uiR_version" in schema["properties"]
    # The structural tree is fully described.
    assert "structure" in schema["properties"]
    assert "semantics" in schema["properties"]
    assert "provenance" in schema["properties"]
