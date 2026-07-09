"""tests/test_fireworks_vision.py -- unit tests for Fireworks AI vision layer.

Tests cover:
    - Image loading and PNG conversion (real PNG, JPEG-to-PNG, unsupported)
    - UIR/UMR builders (synthetic data, no API call)
    - Env var validation helper
    - Pipeline result shape
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from uir_pipeline.fireworks_vision import (
    _ensure_png,
    _get_api_key,
    _get_vision_model,
    load_image_as_b64_png,
    load_image_as_png,
)
from uir_pipeline.image_pipeline import (
    ImagePipelineResult,
    _build_uir,
    _build_umr,
    run_image_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def png_bytes() -> bytes:
    """Return minimal valid PNG bytes (1x1 red pixel)."""
    import base64
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ"
        "/PchF5QAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    d = tmp_path / "images"
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Image conversion tests
# ---------------------------------------------------------------------------


class TestImageConversion:
    def test_png_passthrough(self, png_bytes: bytes) -> None:
        result = _ensure_png(png_bytes, original_ext="png")
        assert result == png_bytes
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_jpeg_to_png(self, tmp_image_dir: Path, png_bytes: bytes) -> None:
        """A .jpg file with any content is read and re-encoded as PNG."""
        jpg_path = tmp_image_dir / "test.jpg"
        jpg_path.write_bytes(png_bytes)  # Use valid PNG bytes as .jpg
        result = load_image_as_png(jpg_path)
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_b64_png_output(self, tmp_image_dir: Path, png_bytes: bytes) -> None:
        path = tmp_image_dir / "test.png"
        path.write_bytes(png_bytes)
        data_uri = load_image_as_b64_png(path)
        assert data_uri.startswith("data:image/png;base64,")

    def test_unsupported_extension(self, tmp_image_dir: Path) -> None:
        path = tmp_image_dir / "test.heic"
        path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="unsupported.*extension"):
            load_image_as_png(path)

    def test_corrupt_image_raises(self, tmp_image_dir: Path) -> None:
        path = tmp_image_dir / "test.png"
        path.write_bytes(b"not a real image at all")
        with pytest.raises(ValueError):
            load_image_as_png(path)


# ---------------------------------------------------------------------------
# Env-var helper tests
# ---------------------------------------------------------------------------


class TestEnvHelpers:
    def test_get_api_key_missing(self) -> None:
        old = os.environ.pop("FIREWORKS_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
                _get_api_key()
        finally:
            if old is not None:
                os.environ["FIREWORKS_API_KEY"] = old

    def test_get_api_key_present(self) -> None:
        old = os.environ.get("FIREWORKS_API_KEY")
        os.environ["FIREWORKS_API_KEY"] = "fw_key_123"
        try:
            assert _get_api_key() == "fw_key_123"
        finally:
            if old is not None:
                os.environ["FIREWORKS_API_KEY"] = old
            else:
                del os.environ["FIREWORKS_API_KEY"]

    def test_default_vision_model(self) -> None:
        old = os.environ.pop("FIREWORKS_VISION_MODEL", None)
        try:
            model = _get_vision_model()
            assert "minimax" in model.lower()
            assert model.startswith("accounts/fireworks/models/")
        finally:
            if old is not None:
                os.environ["FIREWORKS_VISION_MODEL"] = old


# ---------------------------------------------------------------------------
# UIR builder tests
# ---------------------------------------------------------------------------


class TestUIRBuilder:
    @pytest.fixture
    def uir_result(self, tmp_image_dir: Path) -> dict:
        path = tmp_image_dir / "photo.jpg"
        path.write_bytes(b"fake")
        return _build_uir(
            doc_id="doc_test123",
            image_path=path,
            description="A beautiful sunset over mountains.",
            model="test-model/v1",
            intent=None,
            prompt="Describe this image in detail.",
            usage={"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
        )

    def test_modal_type(self, uir_result: dict) -> None:
        assert uir_result["modal_type"] == "image"

    def test_version(self, uir_result: dict) -> None:
        assert uir_result["uiR_version"] == "1.0"

    def test_source_format(self, uir_result: dict) -> None:
        assert uir_result["source"]["format"] == "JPG"

    def test_source_filename(self, uir_result: dict) -> None:
        assert uir_result["source"]["filename"] == "photo.jpg"

    def test_metadata(self, uir_result: dict) -> None:
        assert uir_result["metadata"]["page_count"] == 1
        assert uir_result["metadata"]["chunk_count"] == 1

    def test_structure_has_figure(self, uir_result: dict) -> None:
        children = uir_result["structure"]["root"]["children"]
        assert len(children) == 1
        assert children[0]["type"] == "figure"

    def test_chunk_contains_description(self, uir_result: dict) -> None:
        chunk = uir_result["structure"]["root"]["children"][0]["children"][0]
        assert "sunset" in chunk["text"]
        assert chunk["region_kind"] == "figure"

    def test_chunk_modal_features(self, uir_result: dict) -> None:
        chunk = uir_result["structure"]["root"]["children"][0]["children"][0]
        img = chunk["modal_features"]["image"]
        assert img["model"] == "test-model/v1"
        assert img["intent"] is None
        assert img["usage"]["total_tokens"] == 165

    def test_provenance(self, uir_result: dict) -> None:
        ext = uir_result["provenance"]["extraction"]
        assert ext["model"] == "test-model/v1"

    def test_intent_in_modal(self, tmp_image_dir: Path) -> None:
        path = tmp_image_dir / "test.png"
        path.write_bytes(b"fake")
        uir = _build_uir(
            doc_id="doc_test456",
            image_path=path,
            description="The chart shows quarterly revenue growth.",
            model="test-model/v2",
            intent="What does this chart show?",
            prompt="Answer the user's question about this image.",
            usage=None,
        )
        chunk = uir["structure"]["root"]["children"][0]["children"][0]
        assert chunk["modal_features"]["image"]["intent"] == "What does this chart show?"
        assert chunk["modal_features"]["image"]["usage"] == {}


# ---------------------------------------------------------------------------
# UMR builder tests
# ---------------------------------------------------------------------------


class TestUMRBuilder:
    @pytest.fixture
    def uir_dict(self, tmp_image_dir: Path) -> dict:
        path = tmp_image_dir / "chart.png"
        path.write_bytes(b"fake")
        return _build_uir(
            doc_id="doc_umr_test",
            image_path=path,
            description="Quarterly revenue: Q1 $10M, Q2 $15M, Q3 $22M, Q4 $28M.",
            model="vision-model/1.0",
            intent="Show me the quarterly revenue trend.",
            prompt="Answer the user's question.",
            usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        )

    def test_umr_contains_title(self, uir_dict: dict) -> None:
        umr = _build_umr(uir_dict)
        assert "chart.png" in umr

    def test_umr_contains_model(self, uir_dict: dict) -> None:
        umr = _build_umr(uir_dict)
        assert "vision-model/1.0" in umr

    def test_umr_contains_intent(self, uir_dict: dict) -> None:
        umr = _build_umr(uir_dict)
        assert "quarterly revenue trend" in umr

    def test_umr_contains_description(self, uir_dict: dict) -> None:
        umr = _build_umr(uir_dict)
        assert "Q1 $10M" in umr

    def test_umr_contains_token_usage(self, uir_dict: dict) -> None:
        umr = _build_umr(uir_dict)
        assert "280" in umr

    def test_umr_no_intent(self, tmp_image_dir: Path) -> None:
        path = tmp_image_dir / "photo.png"
        path.write_bytes(b"fake")
        uir = _build_uir(
            doc_id="doc_no_intent",
            image_path=path,
            description="A scenic landscape with mountains.",
            model="test-model/1.0",
            intent=None,
            prompt="Describe this image.",
            usage=None,
        )
        umr = _build_umr(uir)
        assert "photo.png" in umr
        assert "scenic landscape" in umr

    def test_umr_no_figure_children(self) -> None:
        """UIR with no figure children renders stub content."""
        uir = {
            "source": {"filename": "test.png"},
            "metadata": {},
            "structure": {"root": {"children": []}},
            "provenance": {"extraction": {}},
        }
        umr = _build_umr(uir)
        assert "test.png" in umr


# ---------------------------------------------------------------------------
# Pipeline result tests
# ---------------------------------------------------------------------------


class TestImagePipelineResult:
    def test_dataclass_defaults(self) -> None:
        r = ImagePipelineResult(
            uir_id="test_id",
            out_path=Path("/out/test.uir.json"),
            umr_path=Path("/out/test.umr.md"),
            description_length=100,
            model_used="test-model",
            elapsed_seconds=2.5,
        )
        assert r.error is None
        assert r.uir_id == "test_id"

    def test_dataclass_with_error(self) -> None:
        r = ImagePipelineResult(
            uir_id="test_id",
            out_path=Path("/out/test.uir.json"),
            umr_path=Path("/out/test.umr.md"),
            description_length=0,
            model_used="?",
            elapsed_seconds=0.5,
            error="API call failed",
        )
        assert r.error == "API call failed"


# ---------------------------------------------------------------------------
# Pipeline dry-run test
# ---------------------------------------------------------------------------


class TestRunImagePipeline:
    def test_unsupported_format(
        self, tmp_image_dir: Path, tmp_path: Path
    ) -> None:
        """Unsupported format returns error result (does not crash)."""
        path = tmp_image_dir / "test.heic"
        path.write_bytes(b"fake heic data")
        out_dir = tmp_path / "output"

        result = run_image_pipeline(path, out_dir, dry_run=True)
        assert result.error is not None
        assert "unsupported" in result.error.lower()

    def test_on_progress_callback(
        self, tmp_image_dir: Path, tmp_path: Path, png_bytes: bytes, monkeypatch
    ) -> None:
        """on_progress callback is invoked during pipeline stages."""
        import uir_pipeline.image_pipeline as ip

        def mock_describe(*args, **kwargs):
            return {
                "success": True,
                "description": "test description.",
                "model": "mock-model/1.0",
                "prompt": "test",
                "intent": None,
                "usage": {},
            }
        monkeypatch.setattr(ip._fv, "describe_image", mock_describe)
        monkeypatch.setattr(ip._fv, "_get_api_key", lambda: "mock-key")

        path = tmp_image_dir / "test.png"
        path.write_bytes(png_bytes)
        out_dir = tmp_path / "output"
        stages: list[str] = []

        def progress(stage: str, pct: int) -> None:
            stages.append(stage)

        run_image_pipeline(
            path, out_dir, model="test-model/1.0",
            dry_run=True, on_progress=progress,
        )
        assert "convert_png" in stages
        assert "fireworks_vision" in stages
        assert "assemble_uir" in stages
        assert "assemble_umr" in stages
        assert "done" in stages

    def test_dry_run_full_flow(
        self, tmp_image_dir: Path, tmp_path: Path, png_bytes: bytes, monkeypatch
    ) -> None:
        """Full dry-run with mocked API returns clean result."""
        import uir_pipeline.image_pipeline as ip

        def mock_describe(*args, **kwargs):
            return {
                "success": True,
                "description": "A red pixel on a white background.",
                "model": "mock-model/1.0",
                "prompt": "Describe this image.",
                "intent": None,
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            }

        monkeypatch.setattr(ip._fv, "describe_image", mock_describe)
        monkeypatch.setattr(ip._fv, "_get_api_key", lambda: "mock-key")

        path = tmp_image_dir / "test.png"
        path.write_bytes(png_bytes)
        out_dir = tmp_path / "output"

        result = run_image_pipeline(path, out_dir, dry_run=False, model="mock-model/1.0")
        assert result.error is None
        assert result.uir_id
        assert result.out_path.exists()
        assert result.umr_path.exists()
        assert result.description_length > 0
        assert "mock-model" in result.model_used

        import json
        uir = json.loads(result.out_path.read_text())
        assert uir["modal_type"] == "image"
        assert "red pixel" in uir["structure"]["root"]["children"][0]["children"][0]["text"]

        umr = result.umr_path.read_text()
        assert "red pixel" in umr
        assert "test.png" in umr
