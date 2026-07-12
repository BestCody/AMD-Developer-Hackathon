"""Tests for src.uir_pipeline.video_pipeline.

These tests mock ffmpeg, Whisper, and Florence-2 dependencies so they run fast
without GPU or model downloads. End-to-end video coverage (with real models)
lives in ``tests/integration/`` (slow, gated).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from uir_pipeline.video_pipeline import (
    VideoPipelineResult,
    _build_umr,
    _build_uir,
    _choose_interval,
    _fuse_modalities,
    _get_video_metadata,
    _visual_only_chunks,
    run_video_pipeline,
)
from uir_pipeline.chunk import ChunkDraft


# ---------------------------------------------------------------------------
# VideoPipelineResult
# ---------------------------------------------------------------------------

class TestVideoPipelineResult:
    def test_dataclass_fields(self):
        r = VideoPipelineResult(
            uir_id="u",
            out_path=Path("out.json"),
            umr_path=Path("out.md"),
            transcription_length=42,
            chunk_count=3,
            entity_count=0,
            frame_count=10,
            frame_descriptions=8,
            model_used="openai/whisper-small",
            elapsed_seconds=1.2,
            duration_seconds=120.0,
            error=None,
        )
        assert r.transcription_length == 42
        assert r.chunk_count == 3
        assert r.frame_count == 10
        assert r.frame_descriptions == 8


# ---------------------------------------------------------------------------
# _choose_interval
# ---------------------------------------------------------------------------

class TestChooseInterval:
    def test_short_video(self):
        assert _choose_interval(30.0) == 5.0

    def test_medium_video(self):
        assert _choose_interval(120.0) == 10.0

    def test_long_video(self):
        assert _choose_interval(600.0) == 30.0

    def test_very_long_video(self):
        # Cap at 20 frames total: 600s / 20 = 30s
        assert _choose_interval(1200.0) == 60.0


# ---------------------------------------------------------------------------
# _fuse_modalities
# ---------------------------------------------------------------------------

class TestFuseModalities:
    def test_audio_and_visual(self):
        audio_segments = [
            {"start": 0.0, "end": 10.0, "text": "hello world", "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 20.0, "text": "goodbye world", "speaker": "SPEAKER_01"},
        ]
        visual_frames = [
            {"timestamp": 2.0, "description": "A person waves"},
            {"timestamp": 12.0, "description": "A person leaves"},
        ]
        chunks = _fuse_modalities(audio_segments, visual_frames)
        # Two short segments are bundled into one chunk by default target_tokens.
        assert len(chunks) == 1
        assert "[Visual 0:02] A person waves" in chunks[0].text
        assert "hello world" in chunks[0].text
        assert "[Visual 0:12] A person leaves" in chunks[0].text
        assert "goodbye world" in chunks[0].text

        # Check modal_features
        mf0 = chunks[0].modal_features
        assert "video_segment" in mf0
        assert mf0["video_segment"]["start"] == 0.0
        assert mf0["video_segment"]["end"] == 20.0
        assert mf0["video_segment"]["speaker"] == "SPEAKER_00"
        assert len(mf0["video_segment"]["visual_frames"]) == 2

    def test_no_audio(self):
        visual_frames = [
            {"timestamp": 0.0, "description": "Frame one"},
            {"timestamp": 5.0, "description": "Frame two"},
        ]
        chunks = _fuse_modalities([], visual_frames)
        assert len(chunks) == 1
        assert "[Visual 0:00] Frame one" in chunks[0].text
        assert "[Visual 0:05] Frame two" in chunks[0].text

    def test_no_visual(self):
        audio_segments = [
            {"start": 0.0, "end": 10.0, "text": "hello", "speaker": "SPEAKER_00"},
        ]
        chunks = _fuse_modalities(audio_segments, [])
        assert len(chunks) == 1
        assert chunks[0].text == "hello"
        mf = chunks[0].modal_features
        assert len(mf["video_segment"]["visual_frames"]) == 0

    def test_visual_frames_after_last_audio(self):
        audio_segments = [
            {"start": 0.0, "end": 10.0, "text": "hello", "speaker": "SPEAKER_00"},
        ]
        visual_frames = [
            {"timestamp": 5.0, "description": "In audio range"},
            {"timestamp": 15.0, "description": "After audio"},
        ]
        chunks = _fuse_modalities(audio_segments, visual_frames)
        # Should have 2 chunks: one for audio + first visual, one trailing visual
        assert len(chunks) == 2
        assert "[Visual 0:05] In audio range" in chunks[0].text
        assert "[Visual 0:15] After audio" in chunks[1].text


# ---------------------------------------------------------------------------
# _visual_only_chunks
# ---------------------------------------------------------------------------

class TestVisualOnlyChunks:
    def test_grouping(self):
        frames = [
            {"timestamp": 0.0, "description": "Frame one"},
            {"timestamp": 5.0, "description": "Frame two"},
        ]
        chunks = _visual_only_chunks(frames)
        assert len(chunks) == 1
        assert "[Visual 0:00] Frame one" in chunks[0].text

    def test_empty(self):
        assert _visual_only_chunks([]) == []


# ---------------------------------------------------------------------------
# _build_uir
# ---------------------------------------------------------------------------

class TestBuildUir:
    def test_video_uir(self):
        uir = _build_uir(
            doc_id="doc_1",
            video_path=Path("/tmp/test.mp4"),
            transcription={
                "segments": [{"start": 0, "end": 5, "text": "hi"}],
                "language": "en",
                "all_text": "hi",
                "duration_seconds": 10.0,
            },
            visual_frames=[{"timestamp": 2.0, "description": "A frame"}],
            chunks=[
                ChunkDraft(
                    text="hi",
                    token_count=1,
                    page=1,
                    bbox=(0, 0, 1000, 1000),
                    confidence=1.0,
                    modal_features={
                        "video_segment": {
                            "start": 0.0,
                            "end": 5.0,
                            "speaker": "SPEAKER_00",
                            "visual_frames": [{"timestamp": 2.0, "description": "A frame"}],
                        }
                    },
                )
            ],
            model="openai/whisper-small",
            video_meta={"duration_seconds": 10.0, "width": 1920, "height": 1080, "fps": 30.0},
        )
        assert uir["modal_type"] == "video"
        assert uir["source"]["format"] == "MP4"
        meta = uir["metadata"]
        assert meta["modal_features"]["video"]["duration_seconds"] == 10.0
        assert meta["modal_features"]["video"]["width"] == 1920
        assert meta["modal_features"]["video"]["height"] == 1080
        assert meta["modal_features"]["video"]["fps"] == 30.0
        assert meta["modal_features"]["video"]["frame_count"] == 1

    def test_visual_only_uir(self):
        uir = _build_uir(
            doc_id="doc_2",
            video_path=Path("/tmp/test.mov"),
            transcription={"segments": [], "language": None, "all_text": "", "duration_seconds": 5.0},
            visual_frames=[{"timestamp": 1.0, "description": "A frame"}],
            chunks=[
                ChunkDraft(
                    text="[Visual 0:01] A frame",
                    token_count=5,
                    page=1,
                    bbox=(0, 0, 1000, 1000),
                    confidence=1.0,
                    modal_features={
                        "video_segment": {
                            "start": 1.0,
                            "end": 1.0,
                            "speaker": "UNKNOWN",
                            "visual_frames": [{"timestamp": 1.0, "description": "A frame"}],
                        }
                    },
                )
            ],
            model="openai/whisper-small",
            video_meta={"duration_seconds": 5.0},
        )
        assert uir["modal_type"] == "video"
        assert uir["metadata"]["chunk_count"] == 1


# ---------------------------------------------------------------------------
# _build_umr
# ---------------------------------------------------------------------------

class TestBuildUmr:
    def test_video_umr(self):
        uir = {
            "modal_type": "video",
            "source": {"filename": "test.mp4", "format": "MP4"},
            "metadata": {
                "date": "2024-01-01T00:00:00+00:00",
                "language": "en",
                "modal_features": {
                    "video": {
                        "duration_seconds": 10.0,
                        "width": 1920,
                        "height": 1080,
                        "fps": 30.0,
                        "frame_count": 2,
                    }
                }
            },
            "provenance": {"extraction": {"model": "openai/whisper-small"}},
            "structure": {
                "root": {
                    "children": [
                        {
                            "type": "figure",
                            "title": "Transcription + Visual",
                            "children": [
                                {
                                    "type": "chunk",
                                    "text": "Hello world",
                                    "token_count": 2,
                                    "modal_features": {
                                        "video_segment": {
                                            "start": 0.0,
                                            "end": 5.0,
                                            "speaker": "SPEAKER_00",
                                            "visual_frames": [
                                                {"timestamp": 2.0, "description": "A person waves"}
                                            ],
                                        }
                                    },
                                }
                            ],
                        }
                    ]
                }
            },
        }
        md = _build_umr(uir)
        assert "# Video: test.mp4" in md
        assert "Duration: 10.0s" in md
        assert "Resolution: 1920x1080" in md
        assert "FPS: 30.00" in md
        assert "Frames sampled: 2" in md
        assert "0:00 - 0:05" in md
        assert "SPEAKER_00" in md
        assert "[Visual 0:02] A person waves" in md
        assert "Hello world" in md

    def test_empty_video(self):
        uir = {
            "modal_type": "video",
            "source": {"filename": "empty.mp4"},
            "metadata": {"date": "2024-01-01T00:00:00+00:00"},
            "provenance": {"extraction": {"model": "m"}},
            "structure": {"root": {"children": []}},
        }
        md = _build_umr(uir)
        assert "# Video: empty.mp4" in md
        assert "_No content extracted" in md


# ---------------------------------------------------------------------------
# _get_video_metadata (ffprobe-based, mocked)
# ---------------------------------------------------------------------------

class TestGetVideoMetadata:
    def test_metadata_parsing(self, monkeypatch):
        import json as json_mod

        def fake_ffprobe_json(path: Path) -> dict[str, Any]:
            return {
                "format": {"duration": "120.5"},
                "streams": [
                    {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1"},
                    {"codec_type": "audio"},
                ],
            }

        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._ffprobe_json", fake_ffprobe_json
        )
        meta = _get_video_metadata(Path("/fake/video.mp4"))
        assert meta["duration_seconds"] == 120.5
        assert meta["width"] == 1920
        assert meta["height"] == 1080
        assert meta["fps"] == 30.0
        assert meta["has_audio"] is True

    def test_no_audio(self, monkeypatch):
        def fake_ffprobe_json(path: Path) -> dict[str, Any]:
            return {
                "format": {"duration": "60.0"},
                "streams": [
                    {"codec_type": "video", "width": 1280, "height": 720, "r_frame_rate": "24/1"},
                ],
            }

        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._ffprobe_json", fake_ffprobe_json
        )
        meta = _get_video_metadata(Path("/fake/video.mp4"))
        assert meta["has_audio"] is False
        assert meta["fps"] == 24.0

    def test_ffprobe_failure(self, monkeypatch):
        def fake_ffprobe_json(path: Path) -> dict[str, Any]:
            raise RuntimeError("ffprobe not found")

        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._ffprobe_json", fake_ffprobe_json
        )
        meta = _get_video_metadata(Path("/fake/video.mp4"))
        assert meta == {}


# ---------------------------------------------------------------------------
# run_video_pipeline error paths
# ---------------------------------------------------------------------------

class TestRunVideoPipeline:
    def test_missing_file(self, tmp_path: Path):
        result = run_video_pipeline(
            tmp_path / "nonexistent.mp4",
            output_dir=tmp_path / "out",
            dry_run=True,
        )
        assert result.error is not None
        assert "not found" in result.error.lower() or "File not found" in result.error

    def test_ffmpeg_not_available(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "fake.mp4"
        p.write_text("not a video")
        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._ffmpeg_available", lambda: False
        )
        result = run_video_pipeline(p, output_dir=tmp_path / "out", dry_run=True)
        assert result.error is not None
        assert "ffmpeg" in result.error.lower()

    def test_successful_dry_run(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "test.mp4"
        p.write_text("fake video")

        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._ffmpeg_available", lambda: True
        )
        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._get_video_metadata",
            lambda path: {"duration_seconds": 10.0, "has_audio": True, "width": 1920, "height": 1080, "fps": 30.0},
        )
        def _mock_extract_audio(path, output_wav):
            output_wav.write_text("fake audio")
            return True

        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._extract_audio",
            _mock_extract_audio,
        )
        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._sample_frames",
            lambda path, out_dir, interval_seconds: [{"timestamp": 2.0, "path": tmp_path / "frame.jpg"}],
        )
        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._caption_frames",
            lambda frames, device=None: [{"timestamp": 2.0, "description": "A frame", "path": tmp_path / "frame.jpg"}],
        )
        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.transcribe_audio",
            lambda *a, **k: {
                "segments": [{"start": 0.0, "end": 10.0, "text": "hello world"}],
                "language": "en",
                "all_text": "hello world",
                "duration_seconds": 10.0,
            },
        )
        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.diarize_audio",
            lambda *a, **k: [],
        )
        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.align_segments",
            lambda segments, speakers: [{"start": 0.0, "end": 10.0, "text": "hello world", "speaker": "UNKNOWN"}],
        )

        result = run_video_pipeline(p, output_dir=tmp_path / "out", dry_run=True)
        assert result.error is None
        assert result.chunk_count > 0
        assert result.frame_count == 1
        assert result.frame_descriptions == 1
        assert result.transcription_length > 0

    def test_no_audio_fallback(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "silent.mp4"
        p.write_text("fake video")

        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._ffmpeg_available", lambda: True
        )
        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._get_video_metadata",
            lambda path: {"duration_seconds": 10.0, "has_audio": False, "width": 1920, "height": 1080, "fps": 30.0},
        )
        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._sample_frames",
            lambda path, out_dir, interval_seconds: [{"timestamp": 2.0, "path": tmp_path / "frame.jpg"}],
        )
        monkeypatch.setattr(
            "uir_pipeline.video_pipeline._caption_frames",
            lambda frames, device=None: [{"timestamp": 2.0, "description": "A frame", "path": tmp_path / "frame.jpg"}],
        )

        result = run_video_pipeline(p, output_dir=tmp_path / "out", dry_run=True)
        assert result.error is None
        assert result.chunk_count > 0
        assert result.transcription_length == 0
