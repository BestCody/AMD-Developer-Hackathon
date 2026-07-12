"""Tests for src.uir_pipeline.audio_pipeline.

These tests mock the heavy vLLM and pyannote dependencies so they run fast
without GPU or model downloads. End-to-end audio coverage (with real models)
lives in ``tests/integration/`` (slow, gated).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from uir_pipeline.audio_pipeline import (
    AudioPipelineResult,
    align_segments,
    chunk_transcript_segments,
    _build_umr,
    _build_uir,
    _get_audio_metadata,
    _audio_mime_type,
    _bundle_segments_for_chunks,
    run_audio_pipeline,
)
from uir_pipeline.chunk import ChunkDraft


# ---------------------------------------------------------------------------
# _audio_mime_type
# ---------------------------------------------------------------------------

class TestAudioMimeType:
    def test_mp3(self):
        assert _audio_mime_type(".mp3") == "audio/mpeg"

    def test_wav(self):
        assert _audio_mime_type(".wav") == "audio/wav"

    def test_m4a(self):
        assert _audio_mime_type(".m4a") == "audio/mp4"

    def test_flac(self):
        assert _audio_mime_type(".flac") == "audio/flac"

    def test_ogg(self):
        assert _audio_mime_type(".ogg") == "audio/ogg"

    def test_aac(self):
        assert _audio_mime_type(".aac") == "audio/aac"

    def test_wma(self):
        assert _audio_mime_type(".wma") == "audio/x-ms-wma"

    def test_unknown(self):
        assert _audio_mime_type(".xyz") == "audio/unknown"


# ---------------------------------------------------------------------------
# AudioPipelineResult
# ---------------------------------------------------------------------------

class TestAudioPipelineResult:
    def test_dataclass_fields(self):
        r = AudioPipelineResult(
            uir_id="u",
            out_path=Path("out.json"),
            umr_path=Path("out.md"),
            transcription_length=42,
            chunk_count=3,
            entity_count=0,
            model_used="openai/whisper-small",
            elapsed_seconds=1.2,
            language_detected="en",
            duration_seconds=60.0,
            speaker_count=2,
            error=None,
        )
        assert r.transcription_length == 42
        assert r.chunk_count == 3
        assert r.speaker_count == 2


# ---------------------------------------------------------------------------
# _get_audio_metadata (pydub-based)
# ---------------------------------------------------------------------------

class TestGetAudioMetadata:
    def test_wav_file(self, tmp_path: Path):
        import wave

        p = tmp_path / "test.wav"
        with wave.open(str(p), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\x00" * 44100 * 2 * 2)  # 2 seconds of silence

        meta = _get_audio_metadata(p)
        assert meta["duration_seconds"] == pytest.approx(2.0, abs=0.1)
        assert meta["sample_rate"] == 44100
        assert meta["channels"] == 1

    def test_nonexistent_file_returns_empty(self, tmp_path: Path):
        p = tmp_path / "does_not_exist.wav"
        meta = _get_audio_metadata(p)
        assert meta == {}

    def test_text_file_returns_empty(self, tmp_path: Path):
        p = tmp_path / "not_audio.txt"
        p.write_text("hello")
        meta = _get_audio_metadata(p)
        assert meta == {}


# ---------------------------------------------------------------------------
# align_segments
# ---------------------------------------------------------------------------

class TestAlignSegments:
    def test_no_speaker_segments(self):
        tsegs = [
            {"start": 0.0, "end": 10.0, "text": "hello"},
        ]
        aligned = align_segments(tsegs, [])
        assert aligned == [{"start": 0.0, "end": 10.0, "text": "hello", "speaker": "UNKNOWN"}]

    def test_single_overlap(self):
        tsegs = [
            {"start": 0.0, "end": 10.0, "text": "hello"},
        ]
        ssegs = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
        ]
        aligned = align_segments(tsegs, ssegs)
        assert aligned[0]["speaker"] == "SPEAKER_00"

    def test_dominant_speaker_by_overlap(self):
        tsegs = [
            {"start": 0.0, "end": 10.0, "text": "hello"},
        ]
        ssegs = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]
        aligned = align_segments(tsegs, ssegs)
        # SPEAKER_01 has 7s overlap, SPEAKER_00 has 3s.
        assert aligned[0]["speaker"] == "SPEAKER_01"

    def test_no_overlap_falls_to_unknown(self):
        tsegs = [
            {"start": 0.0, "end": 5.0, "text": "hello"},
        ]
        ssegs = [
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00"},
        ]
        aligned = align_segments(tsegs, ssegs)
        assert aligned[0]["speaker"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# _bundle_segments_for_chunks
# ---------------------------------------------------------------------------

class TestBundleSegmentsForChunks:
    def test_empty_segments(self):
        result = _bundle_segments_for_chunks([], 100, 200)
        assert result == []

    def test_single_segment(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hello world", "speaker": "A"}]
        result = _bundle_segments_for_chunks(segs, 100, 200)
        assert len(result) == 1
        assert result[0][0]["text"] == "hello world"

    def test_two_segments_fit_in_one_chunk(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "world", "speaker": "A"},
        ]
        # Very high token limits so both fit.
        result = _bundle_segments_for_chunks(segs, 1000, 2000)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_two_segments_split_when_too_large(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "a " * 500, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "b " * 500, "speaker": "A"},
        ]
        # Low token limit forces split.
        result = _bundle_segments_for_chunks(segs, 10, 20)
        # Each segment is ~500 words which is > 20 tokens, so each goes standalone
        # because _bundle_segments_for_chunks emits oversized segments as
        # standalone chunks.
        assert len(result) == 2


# ---------------------------------------------------------------------------
# chunk_transcript_segments
# ---------------------------------------------------------------------------

class TestChunkTranscriptSegments:
    def test_empty_segments(self):
        assert chunk_transcript_segments([]) == []

    def test_single_segment_creates_one_draft(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hello world", "speaker": "A"},
        ]
        drafts = chunk_transcript_segments(segs, target_tokens=100, max_tokens=200)
        assert len(drafts) == 1
        assert drafts[0].text == "hello world"
        assert drafts[0].page == 1
        assert drafts[0].bbox == (0, 0, 1000, 1000)
        assert drafts[0].modal_features["audio_segment"]["speaker"] == "A"
        assert drafts[0].modal_features["audio_segment"]["start"] == 0.0
        assert drafts[0].modal_features["audio_segment"]["end"] == 1.0

    def test_multiple_segments_bundled(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "world", "speaker": "A"},
            {"start": 2.0, "end": 3.0, "text": "foo", "speaker": "A"},
        ]
        drafts = chunk_transcript_segments(segs, target_tokens=1000, max_tokens=2000)
        assert len(drafts) == 1
        assert "hello" in drafts[0].text
        assert "world" in drafts[0].text
        assert "foo" in drafts[0].text

    def test_dominant_speaker_computed(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"},
            {"start": 1.0, "end": 2.0, "text": "world", "speaker": "A"},
            {"start": 2.0, "end": 3.0, "text": "foo", "speaker": "B"},
        ]
        drafts = chunk_transcript_segments(segs, target_tokens=1000, max_tokens=2000)
        assert len(drafts) == 1
        # A appears twice, B once -> dominant is A
        assert drafts[0].modal_features["audio_segment"]["speaker"] == "A"

    def test_timing_spans_all_segments(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"},
            {"start": 5.0, "end": 6.0, "text": "world", "speaker": "A"},
        ]
        drafts = chunk_transcript_segments(segs, target_tokens=1000, max_tokens=2000)
        assert len(drafts) == 1
        assert drafts[0].modal_features["audio_segment"]["start"] == 0.0
        assert drafts[0].modal_features["audio_segment"]["end"] == 6.0


# ---------------------------------------------------------------------------
# _build_uir
# ---------------------------------------------------------------------------

class TestBuildUir:
    def test_basic_uir_structure(self, tmp_path: Path):
        p = tmp_path / "test.mp3"
        p.write_text("not real audio")

        chunks = [
            ChunkDraft(
                text="hello world",
                token_count=3,
                page=1,
                bbox=(0, 0, 1000, 1000),
                confidence=1.0,
                modal_features={
                    "audio_segment": {"start": 0.0, "end": 1.0, "speaker": "A"},
                    "text": {"token_count": 3, "chunk_strategy": "segment-aware"},
                },
            ),
        ]

        transcription = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            "language": "en",
            "duration_seconds": 60.0,
        }

        uir = _build_uir(
            doc_id="doc_123",
            audio_path=p,
            transcription=transcription,
            chunks=chunks,
            model="openai/whisper-small",
            audio_meta={"duration_seconds": 60.0, "sample_rate": 44100, "channels": 2},
        )

        assert uir["uiR_version"] == "1.0"
        assert uir["id"] == "doc_123"
        assert uir["modal_type"] == "audio"
        assert uir["source"]["format"] == "MP3"
        assert uir["source"]["route"] == "audio"
        assert uir["metadata"]["title"].startswith("Audio transcription")
        assert uir["metadata"]["modal_features"]["audio"]["duration_seconds"] == 60.0
        assert uir["metadata"]["modal_features"]["audio"]["sample_rate"] == 44100
        assert uir["metadata"]["modal_features"]["audio"]["channels"] == 2

        # Structure
        root = uir["structure"]["root"]
        assert root["type"] == "document"
        assert len(root["children"]) == 1
        assert root["children"][0]["type"] == "figure"
        assert root["children"][0]["title"] == "Transcription"
        chunk_nodes = root["children"][0]["children"]
        assert len(chunk_nodes) == 1
        assert chunk_nodes[0]["type"] == "chunk"
        assert chunk_nodes[0]["text"] == "hello world"
        assert chunk_nodes[0]["modal_features"]["audio_segment"]["speaker"] == "A"

    def test_empty_chunks(self, tmp_path: Path):
        p = tmp_path / "test.wav"
        p.write_text("not real audio")

        uir = _build_uir(
            doc_id="doc_456",
            audio_path=p,
            transcription={"segments": [], "duration_seconds": 0},
            chunks=[],
            model="openai/whisper-small",
            audio_meta={},
        )

        assert uir["modal_type"] == "audio"
        root = uir["structure"]["root"]
        assert len(root["children"]) == 0


# ---------------------------------------------------------------------------
# _build_umr
# ---------------------------------------------------------------------------

class TestBuildUmr:
    def test_basic_umr(self):
        uir = {
            "uiR_version": "1.0",
            "id": "doc_123",
            "modal_type": "audio",
            "source": {
                "uri": "file:///test.mp3",
                "filename": "test.mp3",
                "format": "MP3",
                "route": "audio",
                "mime_type": "audio/mpeg",
                "size_bytes": 0,
                "checksum": "",
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            "metadata": {
                "title": "Audio transcription: test.mp3",
                "author": None,
                "page_count": 1,
                "chunk_count": 1,
                "language": "en",
                "format": "MP3",
                "modal_features": {
                    "audio": {
                        "duration_seconds": 60.0,
                        "sample_rate": 44100,
                        "channels": 2,
                        "language_detected": "en",
                        "model": "openai/whisper-small",
                        "segments": [],
                    },
                },
            },
            "structure": {
                "root": {
                    "id": "doc_123",
                    "type": "document",
                    "title": "Audio: test.mp3",
                    "children": [
                        {
                            "id": "fig_123",
                            "type": "figure",
                            "title": "Transcription",
                            "children": [
                                {
                                    "id": "chunk_123",
                                    "type": "chunk",
                                    "text": "hello world",
                                    "token_count": 3,
                                    "page": 1,
                                    "bounding_box": [0, 0, 1000, 1000],
                                    "confidence": 1.0,
                                    "modal_features": {
                                        "audio_segment": {
                                            "start": 0.0,
                                            "end": 1.0,
                                            "speaker": "SPEAKER_00",
                                        },
                                    },
                                }
                            ],
                        }
                    ],
                }
            },
            "semantics": {"entities": [], "relationships": [], "topics": []},
            "provenance": {
                "extraction": {
                    "model": "openai/whisper-small",
                    "version": "1.0",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                },
                "normalization": {
                    "version": "1.0",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                },
            },
        }

        md = _build_umr(uir)
        assert "# Audio: test.mp3" in md
        assert "Format: MP3" in md
        assert "Duration: 60.0s" in md
        assert "Sample rate: 44100 Hz" in md
        assert "Channels: 2" in md
        assert "SPEAKER_00" in md
        assert "0:00 - 0:01" in md
        assert "hello world" in md

    def test_empty_transcription(self):
        uir = {
            "uiR_version": "1.0",
            "id": "doc_123",
            "modal_type": "audio",
            "source": {"filename": "test.mp3", "format": "MP3"},
            "metadata": {"title": "test", "page_count": 1, "language": "en", "format": "MP3"},
            "structure": {
                "root": {
                    "id": "doc_123",
                    "type": "document",
                    "title": "Audio: test.mp3",
                    "children": [],
                }
            },
            "semantics": {"entities": [], "relationships": [], "topics": []},
            "provenance": {
                "extraction": {"model": "m", "version": "1.0", "timestamp": "2024-01-01T00:00:00+00:00"},
                "normalization": {"version": "1.0", "timestamp": "2024-01-01T00:00:00+00:00"},
            },
        }

        md = _build_umr(uir)
        assert "# Audio: test.mp3" in md
        assert "No transcription extracted" in md


# ---------------------------------------------------------------------------
# run_audio_pipeline (mocked)
# ---------------------------------------------------------------------------

class TestRunAudioPipeline:
    def test_successful_run(self, tmp_path: Path, monkeypatch):
        import wave

        p = tmp_path / "test.wav"
        with wave.open(str(p), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\x00" * 44100 * 2)  # 1 second

        out_dir = tmp_path / "out"

        # Mock transcription
        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.transcribe_audio",
            lambda *a, **k: {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello world"},
                ],
                "language": "en",
                "all_text": "hello world",
                "duration_seconds": 1.0,
            },
        )

        # Mock diarization (skip by returning empty)
        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.diarize_audio",
            lambda *a, **k: [],
        )

        result = run_audio_pipeline(p, out_dir, model_id="mock-model")

        assert result.error is None
        assert result.uir_id != ""
        assert result.chunk_count == 1
        assert result.transcription_length == 11  # "hello world"
        assert result.model_used == "mock-model"
        assert result.language_detected == "en"
        assert result.duration_seconds == 1.0
        assert result.speaker_count is None  # no diarization -> None
        assert result.out_path.exists()
        assert result.umr_path.exists()

        # Validate UIR JSON
        uir = json.loads(result.out_path.read_text(encoding="utf-8"))
        assert uir["modal_type"] == "audio"
        assert uir["metadata"]["modal_features"]["audio"]["duration_seconds"] == 1.0

        # Validate UMR markdown
        umr = result.umr_path.read_text(encoding="utf-8")
        assert "hello world" in umr
        assert "UNKNOWN" in umr

    def test_file_not_found(self, tmp_path: Path):
        p = tmp_path / "does_not_exist.wav"
        out_dir = tmp_path / "out"
        result = run_audio_pipeline(p, out_dir)
        assert result.error is not None
        assert "File not found" in result.error
        assert result.chunk_count == 0

    def test_transcription_failure(self, tmp_path: Path, monkeypatch):
        import wave

        p = tmp_path / "test.wav"
        with wave.open(str(p), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\x00" * 44100 * 2)

        out_dir = tmp_path / "out"

        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.transcribe_audio",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vLLM failed")),
        )

        result = run_audio_pipeline(p, out_dir)
        assert result.error is not None
        assert "vLLM failed" in result.error

    def test_dry_run_does_not_write(self, tmp_path: Path, monkeypatch):
        import wave

        p = tmp_path / "test.wav"
        with wave.open(str(p), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\x00" * 44100 * 2)

        out_dir = tmp_path / "out"

        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.transcribe_audio",
            lambda *a, **k: {
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
                "language": "en",
                "all_text": "hello",
                "duration_seconds": 1.0,
            },
        )
        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.diarize_audio",
            lambda *a, **k: [],
        )

        result = run_audio_pipeline(p, out_dir, dry_run=True)
        assert result.error is None
        assert not result.out_path.exists()
        assert not result.umr_path.exists()

    def test_progress_callback(self, tmp_path: Path, monkeypatch):
        import wave

        p = tmp_path / "test.wav"
        with wave.open(str(p), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\x00" * 44100 * 2)

        out_dir = tmp_path / "out"

        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.transcribe_audio",
            lambda *a, **k: {
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
                "language": "en",
                "all_text": "hello",
                "duration_seconds": 1.0,
            },
        )
        monkeypatch.setattr(
            "uir_pipeline.audio_pipeline.diarize_audio",
            lambda *a, **k: [],
        )

        stages = []
        def _on_progress(stage, pct, **meta):
            stages.append((stage, pct))

        result = run_audio_pipeline(p, out_dir, on_progress=_on_progress)
        assert result.error is None
        assert stages
        # Should include at least validate and done stages
        assert any(s == "validate" for s, _ in stages)
        assert any(s == "done" for s, _ in stages)
