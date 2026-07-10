"""The IMAGE route must fail loudly, not return a success-shaped result.

`run_image_pipeline` reports failure in `ImagePipelineResult.error` rather than
raising. `pipeline.run` folded that into `chunk_count=0` and returned a normal
`PipelineResult`, so a failed image analysis looked like this::

    $ python pipeline.py chart.png --output-data out/
    done chart.png: chunks=0 -> out/doc_4c81bb91.uir.json
    $ echo $?
    0
    $ ls out/
    (empty)

The named file was never written. In the console the job reported `done` and
`/api/result/<job>` 404'd. Same silent-failure shape as Docling's
PARTIAL_SUCCESS: the pipeline knew, and threw the knowledge away.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from uir_pipeline.pipeline import ImageAnalysisError, run


@pytest.fixture
def png(tmp_path: Path) -> Path:
    pytest.importorskip("PIL")
    from PIL import Image

    p = tmp_path / "chart.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(p)
    return p


def _image_result(**kw):
    base = dict(
        uir_id="doc_x",
        out_path=Path("out") / "doc_x.uir.json",
        umr_path=Path("out") / "doc_x.umr.md",
        description_length=0,
        model_used="m",
        elapsed_seconds=0.1,
        error=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_image_analysis_failure_raises_instead_of_reporting_done(png, tmp_path, monkeypatch):
    import uir_pipeline.image_pipeline as ip

    monkeypatch.setattr(
        ip, "run_image_pipeline",
        lambda *_a, **_k: _image_result(error="HTTP 401: Unauthorized"),
    )
    with pytest.raises(ImageAnalysisError, match="401"):
        run(png, output_dir=tmp_path / "out", skip_weaviate=True, with_embeddings=False)


def test_image_analysis_error_is_a_runtime_error(png, tmp_path, monkeypatch):
    """The web runner catches broad exceptions; CLI turns it into exit 1."""
    assert issubclass(ImageAnalysisError, RuntimeError)


def test_successful_image_returns_one_chunk(png, tmp_path, monkeypatch):
    import uir_pipeline.image_pipeline as ip

    out = tmp_path / "out"
    monkeypatch.setattr(
        ip, "run_image_pipeline",
        lambda *_a, **_k: _image_result(out_path=out / "doc_x.uir.json"),
    )
    result = run(png, output_dir=out, skip_weaviate=True, with_embeddings=False)
    assert result.chunk_count == 1
    assert result.uir_id == "doc_x"


# ---------------------------------------------------------------------------
# describe_image's fail-soft contract
# ---------------------------------------------------------------------------
# It documents `success`/`error` keys, but a missing API key raised ValueError
# out of `_get_api_key` while an HTTP 401 returned the dict -- so
# `run_image_pipeline`'s single `if not result["success"]` branch handled one
# failure and not the other.

def test_missing_api_key_returns_an_error_dict_not_an_exception(png, monkeypatch):
    from uir_pipeline.fireworks_vision import describe_image

    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    out = describe_image(png)
    assert out["success"] is False
    assert "FIREWORKS_API_KEY" in out["error"]
    assert out["description"] == ""


def test_missing_api_key_error_dict_has_every_documented_key(png, monkeypatch):
    from uir_pipeline.fireworks_vision import describe_image

    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    out = describe_image(png)
    for key in ("success", "description", "model", "prompt", "intent", "error", "usage"):
        assert key in out, f"missing {key}"


def test_blank_api_key_is_treated_as_missing(png, monkeypatch):
    from uir_pipeline.fireworks_vision import describe_image

    monkeypatch.setenv("FIREWORKS_API_KEY", "   ")
    out = describe_image(png)
    assert out["success"] is False


def test_missing_key_never_reaches_the_network(png, monkeypatch):
    """A config error must not cost an HTTP round trip."""
    import requests

    from uir_pipeline.fireworks_vision import describe_image

    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

    def _boom(*_a, **_k):
        raise AssertionError("describe_image posted without an API key")

    monkeypatch.setattr(requests, "post", _boom)
    assert describe_image(png)["success"] is False
