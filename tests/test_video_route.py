"""The VIDEO route must fail loudly, not return a success-shaped result.

`run_video_pipeline` reports failure in `VideoPipelineResult.error` rather than
raising. `pipeline.run` translates that into an `VideoAnalysisError` so the
caller sees a clear failure instead of a ``done`` job with a missing file.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from uir_pipeline.pipeline import VideoAnalysisError, run


@pytest.fixture
def mp4(tmp_path: Path) -> Path:
    p = tmp_path / "test.mp4"
    p.write_text("fake video content")
    return p


def _video_result(**kw: Any):
    base = dict(
        uir_id="doc_x",
        out_path=Path("out") / "doc_x.uir.json",
        umr_path=Path("out") / "doc_x.umr.md",
        transcription_length=0,
        chunk_count=0,
        entity_count=0,
        frame_count=0,
        frame_descriptions=0,
        model_used="m",
        elapsed_seconds=0.1,
        duration_seconds=None,
        error=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_video_analysis_failure_raises_instead_of_reporting_done(mp4, tmp_path, monkeypatch):
    import uir_pipeline.video_pipeline as vp

    monkeypatch.setattr(
        vp, "run_video_pipeline",
        lambda *a, **k: _video_result(error="ffmpeg missing"),
    )
    with pytest.raises(VideoAnalysisError, match="ffmpeg missing"):
        run(mp4, output_dir=tmp_path / "out", skip_weaviate=True, with_embeddings=False)


def test_video_analysis_error_is_a_runtime_error():
    assert issubclass(VideoAnalysisError, RuntimeError)


def test_successful_video_returns_chunks(mp4, tmp_path, monkeypatch):
    import uir_pipeline.video_pipeline as vp

    out = tmp_path / "out"
    monkeypatch.setattr(
        vp, "run_video_pipeline",
        lambda *a, **k: _video_result(
            out_path=out / "doc_x.uir.json",
            chunk_count=2,
            transcription_length=11,
            frame_count=3,
            frame_descriptions=3,
        ),
    )
    result = run(mp4, output_dir=out, skip_weaviate=True, with_embeddings=False)
    assert result.chunk_count == 2
    assert result.uir_id == "doc_x"
