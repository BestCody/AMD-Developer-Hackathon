"""The AUDIO route must fail loudly, not return a success-shaped result.

`run_audio_pipeline` reports failure in `AudioPipelineResult.error` rather than
raising. `pipeline.run` translates that into an `AudioAnalysisError` so the
caller sees a clear failure instead of a ``done`` job with a missing file.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import wave

from uir_pipeline.pipeline import AudioAnalysisError, run


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "test.wav"
    with wave.open(str(p), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\x00" * 44100 * 2)  # 1 second of silence
    return p


def _audio_result(**kw: Any):
    base = dict(
        uir_id="doc_x",
        out_path=Path("out") / "doc_x.uir.json",
        umr_path=Path("out") / "doc_x.umr.md",
        transcription_length=0,
        chunk_count=0,
        entity_count=0,
        model_used="m",
        elapsed_seconds=0.1,
        language_detected=None,
        duration_seconds=None,
        speaker_count=None,
        error=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_audio_analysis_failure_raises_instead_of_reporting_done(wav, tmp_path, monkeypatch):
    import uir_pipeline.audio_pipeline as ap

    monkeypatch.setattr(
        ap, "transcribe_audio",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vLLM crashed")),
    )
    with pytest.raises(AudioAnalysisError, match="vLLM crashed"):
        run(wav, output_dir=tmp_path / "out", skip_weaviate=True, with_embeddings=False)


def test_audio_analysis_error_is_a_runtime_error():
    assert issubclass(AudioAnalysisError, RuntimeError)


def test_successful_audio_returns_chunks(wav, tmp_path, monkeypatch):
    import uir_pipeline.audio_pipeline as ap

    out = tmp_path / "out"
    monkeypatch.setattr(
        ap, "transcribe_audio",
        lambda *a, **k: {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            "language": "en",
            "all_text": "hello world",
            "duration_seconds": 1.0,
        },
    )
    monkeypatch.setattr(
        ap, "diarize_audio",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        ap, "run_audio_pipeline",
        lambda *a, **k: _audio_result(
            out_path=out / "doc_x.uir.json",
            chunk_count=2,
            transcription_length=11,
        ),
    )
    result = run(wav, output_dir=out, skip_weaviate=True, with_embeddings=False)
    assert result.chunk_count == 2
    assert result.uir_id == "doc_x"
