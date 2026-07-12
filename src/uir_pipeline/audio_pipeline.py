"""audio_pipeline -- audio transcription to UIR + UMR using vLLM Whisper + pyannote diarization.

This module is the high-level orchestrator for processing a single audio
file through vLLM-served Whisper transcription and pyannote.audio speaker
diarization, producing the standard UIR (Universal Intermediate Representation)
and UMR (Universal Markdown Representation) outputs.

Flow:
    1. Validate audio format and extract metadata (pydub).
    2. Transcribe audio using vLLM-served Whisper model.
    3. Run speaker diarization using pyannote.audio (optional).
    4. Align transcription segments with speaker labels.
    5. Chunk transcript segments (speaker-aware chunking).
    6. Enrich chunks (spaCy NER).
    7. Embed chunks (BGE-small).
    8. Build UIRV1 with modal_type="audio".
    9. Build UMR markdown with speaker labels and timestamps.
    10. Write outputs: {doc_id}.uir.json + {doc_id}.umr.md.

All audio formats (MP3, WAV, M4A, FLAC, OGG, AAC, WMA) are supported.
ffmpeg must be on PATH for pydub to handle non-WAV formats.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uir_pipeline.chunk import (
    ChunkDraft,
    DEFAULT_CHUNK_OVERLAP_PCT,
    DEFAULT_CHUNK_TARGET_TOKENS,
    MAX_CHUNK_TOKENS,
)
from uir_pipeline.utils import (
    deterministic_node_id,
    count_tokens,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_WHISPER_MODEL_ID: str = os.environ.get("WHISPER_MODEL_ID", "openai/whisper-small")
_DIARIZATION_MODEL_ID: str = os.environ.get(
    "DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-3.1"
)
_DIARIZATION_DISABLE: bool = os.environ.get(
    "DIARIZATION_DISABLE", "0"
).strip().lower() in ("1", "true", "yes", "on")
_WHISPER_LANGUAGE: str | None = os.environ.get("WHISPER_LANGUAGE") or None


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class AudioPipelineResult:
    """Shape returned by :func:`run_audio_pipeline`."""

    uir_id: str
    out_path: Path
    umr_path: Path
    transcription_length: int
    chunk_count: int
    entity_count: int
    model_used: str
    elapsed_seconds: float
    language_detected: str | None = None
    duration_seconds: float | None = None
    speaker_count: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Lazy model singletons
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()

_DIARIZATION_CACHE: dict[tuple[str, str], Any] = {}
_DIARIZATION_LOCK = threading.Lock()


def _get_vllm_whisper(
    model_id: str = _WHISPER_MODEL_ID,
    *,
    device: str | None = None,
    force_reload: bool = False,
) -> Any:
    """Lazy-load the vLLM LLM with a Whisper model.

    Returns a vLLM :class:`LLM` instance. Cached per ``(model_id, device)``
    after first load so repeated calls in a long-running server stay cheap.

    Device resolution follows :mod:`uir_pipeline.device`.
    The vLLM Whisper API is evolving. This loader uses the standard vLLM
    ``LLM`` class with the model_id. The exact generation API may need
    adjustment for different vLLM versions.
    """
    if device is None:
        from uir_pipeline.device import get_device

        device = get_device()

    cache_key = (model_id, device)
    cached = None if force_reload else _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None and not force_reload:
            return cached

        logger.info(
            "loading vLLM Whisper model_id=%s device=%s "
            "(cold cache; first run may download weights)",
            model_id,
            device,
        )

        try:
            from vllm import LLM
        except ImportError as exc:
            logger.error("vLLM not installed: %s", exc)
            raise RuntimeError(
                "vLLM is required for audio transcription. "
                "Install with: pip install vllm>=0.7.3"
            ) from exc

        # vLLM handles device selection internally via CUDA_VISIBLE_DEVICES.
        # For AMD ROCm, device="cuda" works via the HIP compatibility layer.
        # dtype selection: fp16 on CUDA/ROCm, fp32 on CPU/MPS.
        dtype = "float16" if device == "cuda" else "float32"

        try:
            # vLLM 0.7.3+ Whisper support: load the model directly.
            # The exact arguments may vary by vLLM version; the LLM class
            # auto-detects the model architecture.
            llm = LLM(
                model=model_id,
                dtype=dtype,
            )
        except Exception as exc:
            logger.error("vLLM model load failed: %s", exc)
            raise RuntimeError(
                f"Failed to load vLLM Whisper model {model_id}: {exc}"
            ) from exc

        _MODEL_CACHE[cache_key] = llm
        return llm


def _get_transformers_whisper(
    model_id: str = _WHISPER_MODEL_ID,
    *,
    device: str | None = None,
    force_reload: bool = False,
) -> tuple[Any, Any]:
    """Lazy-load Whisper via HuggingFace transformers (Darwin / CPU fallback).

    Returns a ``(model, processor)`` pair. Cached per ``(model_id, device)``.
    This is the fallback path when vLLM is unavailable (e.g. macOS, CPU-only).
    """
    if device is None:
        from uir_pipeline.device import get_device

        device = get_device()

    cache_key = (model_id, device, "transformers")
    cached = None if force_reload else _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None and not force_reload:
            return cached

        logger.info(
            "loading transformers Whisper model_id=%s device=%s "
            "(cold cache; first run may download weights)",
            model_id,
            device,
        )

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        torch_dtype = torch.float32 if device in ("cpu", "mps") else torch.float16
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        _MODEL_CACHE[cache_key] = (model, processor)
        return _MODEL_CACHE[cache_key]


def _get_pyannote_pipeline(
    model_id: str = _DIARIZATION_MODEL_ID,
    *,
    device: str | None = None,
    force_reload: bool = False,
) -> Any:
    """Lazy-load the pyannote.audio speaker diarization pipeline.

    Returns a pyannote :class:`Pipeline` instance. Cached per ``(model_id, device)``.
    """
    if device is None:
        from uir_pipeline.device import get_device

        device = get_device()

    cache_key = (model_id, device)
    cached = None if force_reload else _DIARIZATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _DIARIZATION_LOCK:
        cached = _DIARIZATION_CACHE.get(cache_key)
        if cached is not None and not force_reload:
            return cached

        logger.info(
            "loading pyannote diarization model_id=%s device=%s",
            model_id,
            device,
        )

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            logger.warning("pyannote.audio not installed: %s", exc)
            raise RuntimeError(
                "pyannote.audio is required for speaker diarization. "
                "Install with: pip install pyannote.audio>=3.1"
            ) from exc

        try:
            pipeline = Pipeline.from_pretrained(model_id)
            if hasattr(pipeline, "to"):
                pipeline.to(device)
        except Exception as exc:
            logger.error("pyannote pipeline load failed: %s", exc)
            raise RuntimeError(
                f"Failed to load pyannote model {model_id}: {exc}"
            ) from exc

        _DIARIZATION_CACHE[cache_key] = pipeline
        return pipeline


# ---------------------------------------------------------------------------
# Audio metadata extraction
# ---------------------------------------------------------------------------


def _get_audio_metadata(path: Path) -> dict[str, Any]:
    """Extract audio file metadata: duration, sample_rate, channels.

    Uses pydub. Requires ffmpeg on PATH for non-WAV formats.
    Fail-soft: returns empty dict on any exception.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.warning("pydub not installed; audio metadata unavailable")
        return {}

    try:
        audio = AudioSegment.from_file(str(path))
        return {
            "duration_seconds": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "sample_width": audio.sample_width,
        }
    except Exception as exc:
        logger.warning(
            "audio metadata extraction failed for %s: %s", path.name, exc
        )
        return {}


def _audio_mime_type(ext: str) -> str:
    """Map audio extension to MIME type."""
    mapping = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma",
    }
    return mapping.get(ext.lower(), "audio/unknown")


# ---------------------------------------------------------------------------
# Transcription via vLLM Whisper
# ---------------------------------------------------------------------------


def _transcribe_with_transformers(
    audio_path: Path,
    *,
    model_id: str = _WHISPER_MODEL_ID,
    device: str | None = None,
    language: str | None = _WHISPER_LANGUAGE,
) -> dict[str, Any]:
    """Transcribe audio using HuggingFace transformers (Darwin / CPU fallback)."""
    import platform
    import torch
    from transformers import pipeline

    if device is None:
        from uir_pipeline.device import get_device
        device = get_device()

    logger.info(
        "transcribing %s with transformers Whisper (%s) on %s",
        audio_path.name, model_id, platform.system()
    )

    torch_dtype = torch.float32 if device in ("cpu", "mps") else torch.float16
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )

    kwargs = {"return_timestamps": True}
    if language:
        kwargs["generate_kwargs"] = {"language": language}
    result = pipe(str(audio_path), **kwargs)

    segments = []
    all_text = ""
    if "chunks" in result:
        for chunk in result["chunks"]:
            ts = chunk.get("timestamp", (0.0, None))
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start + 5.0
            text = chunk.get("text", "")
            segments.append({"start": start, "end": end, "text": text})
            all_text += text + " "
    elif "text" in result:
        all_text = result["text"]
        segments.append({"start": 0.0, "end": 0.0, "text": all_text})

    meta = _get_audio_metadata(audio_path)
    duration = meta.get("duration_seconds", 0.0)
    if segments and duration > 0:
        segments[-1]["end"] = max(segments[-1]["end"], duration)

    return {
        "segments": segments,
        "language": language,
        "language_probability": None,
        "all_text": all_text.strip(),
        "duration_seconds": duration,
    }


def transcribe_audio(
    audio_path: Path,
    *,
    model_id: str = _WHISPER_MODEL_ID,
    device: str | None = None,
    language: str | None = _WHISPER_LANGUAGE,
) -> dict[str, Any]:
    """Transcribe an audio file using vLLM-served Whisper (Linux) or transformers (Darwin).

    Returns a dict with keys:
        segments: list of {start, end, text} segments (approximate)
        language: detected language code (or None)
        all_text: full joined transcription text
        duration_seconds: audio duration (approximate)
    """
    import platform

    # On Darwin (macOS) vLLM is typically unavailable; use transformers fallback.
    if platform.system() == "Darwin":
        return _transcribe_with_transformers(
            audio_path,
            model_id=model_id,
            device=device,
            language=language,
        )

    t0 = time.monotonic()
    llm = _get_vllm_whisper(model_id=model_id, device=device)

    audio_path_str = str(audio_path)
    logger.info(
        "transcribing %s with vLLM Whisper (%s)", audio_path.name, model_id
    )

    try:
        # vLLM Whisper API -- the exact prompt format depends on vLLM version.
        # For vLLM 0.7.3+, audio is typically passed as a string path or a
        # special multimodal prompt. We try common patterns.

        # Primary pattern: direct string path as prompt (vLLM treats it as
        # an audio file path).
        outputs = llm.generate([audio_path_str])
        text = outputs[0].outputs[0].text

    except Exception as exc:
        logger.warning("vLLM primary generate API failed: %s", exc)
        try:
            # Alternative: dict with audio key (used in some vLLM versions).
            outputs = llm.generate([{"audio": audio_path_str}])
            text = outputs[0].outputs[0].text
        except Exception as exc2:
            logger.error("vLLM transcription failed: %s", exc2)
            raise RuntimeError(
                f"vLLM transcription failed: {exc2}"
            ) from exc2

    # vLLM Whisper returns plain text. We create a single segment covering
    # the full duration. Timing is approximate; refined later if we have
    # duration metadata.
    elapsed = time.monotonic() - t0
    logger.info(
        "transcription done in %.2fs; text length=%d", elapsed, len(text)
    )

    # Try to get audio duration for segment timing.
    meta = _get_audio_metadata(audio_path)
    duration = meta.get("duration_seconds", 0.0)

    return {
        "segments": [
            {
                "start": 0.0,
                "end": duration,
                "text": text,
            }
        ],
        "language": language,
        "language_probability": None,
        "all_text": text,
        "duration_seconds": duration,
    }


# ---------------------------------------------------------------------------
# Speaker diarization via pyannote.audio
# ---------------------------------------------------------------------------


def diarize_audio(
    audio_path: Path,
    *,
    model_id: str = _DIARIZATION_MODEL_ID,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run speaker diarization on an audio file.

    Returns a list of {start, end, speaker} segments.
    Fail-soft: returns ``[]`` if pyannote is unavailable or disabled.
    """
    if _DIARIZATION_DISABLE:
        logger.info("speaker diarization disabled via DIARIZATION_DISABLE")
        return []

    try:
        pipeline = _get_pyannote_pipeline(model_id=model_id, device=device)
    except RuntimeError as exc:
        logger.warning("diarization skipped: %s", exc)
        return []

    logger.info("running speaker diarization on %s", audio_path.name)
    try:
        diarization = pipeline(str(audio_path))
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )
        logger.info(
            "diarization found %d speaker segments, %d unique speakers",
            len(segments),
            len({s["speaker"] for s in segments}),
        )
        return segments
    except Exception as exc:
        logger.warning("diarization failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Align transcription + diarization
# ---------------------------------------------------------------------------


def align_segments(
    transcription_segments: list[dict[str, Any]],
    speaker_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign a dominant speaker to each transcription segment.

    For each transcription segment, find the speaker segment with the
    largest time overlap. If no speaker segments, all get ``'UNKNOWN'``.
    """
    if not speaker_segments:
        return [
            {**seg, "speaker": "UNKNOWN"} for seg in transcription_segments
        ]

    aligned = []
    for tseg in transcription_segments:
        t_start = float(tseg.get("start", 0))
        t_end = float(tseg.get("end", 0))

        best_speaker = "UNKNOWN"
        max_overlap = 0.0

        for sseg in speaker_segments:
            s_start = sseg["start"]
            s_end = sseg["end"]
            overlap = max(0.0, min(t_end, s_end) - max(t_start, s_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = sseg["speaker"]

        aligned.append({**tseg, "speaker": best_speaker})

    return aligned


# ---------------------------------------------------------------------------
# Speaker-aware chunking
# ---------------------------------------------------------------------------


def _bundle_segments_for_chunks(
    segments: list[dict[str, Any]],
    target_tokens: int,
    max_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Greedy bundle segments into chunks, preserving boundaries.

    Mirrors the paragraph-bundling logic in :func:`chunk._bundle_paragraphs_for_chunks`
    but tracks per-segment metadata. Each segment is treated as a paragraph.
    Returns a list of chunk-groups, where each group is a list of segments.
    """
    chunks: list[list[dict[str, Any]]] = []
    buf: list[dict[str, Any]] = []
    buf_tokens = 0

    def _flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            chunks.append(list(buf))
            buf = []
            buf_tokens = 0

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        seg_tokens = count_tokens(seg_text)

        if seg_tokens > max_tokens:
            # Oversized single segment: flush buffer, then emit this segment
            # standalone. We do not recursively halve here; the BGE embedder
            # will truncate at 512 tokens if needed.
            _flush()
            chunks.append([seg])
            continue

        if buf_tokens + seg_tokens <= max_tokens:
            buf.append(seg)
            buf_tokens += seg_tokens
            if buf_tokens >= target_tokens:
                _flush()
        else:
            _flush()
            buf.append(seg)
            buf_tokens = seg_tokens
            if buf_tokens >= target_tokens:
                _flush()

    _flush()
    return chunks


def _with_overlap_segments(
    chunk_groups: list[list[dict[str, Any]]],
    overlap_pct: int,
) -> list[list[dict[str, Any]]]:
    """Re-stitch chunk groups with a small tail overlap.

    Simplified overlap for audio transcripts: include the last segment of
    the previous group in the next group if it fits within the token limit.
    """
    if not chunk_groups or overlap_pct <= 0:
        return list(chunk_groups)

    out: list[list[dict[str, Any]]] = [chunk_groups[0]]

    for prev_group, curr_group in zip(chunk_groups, chunk_groups[1:]):
        if not prev_group or not curr_group:
            out.append(curr_group)
            continue

        tail_seg = prev_group[-1]
        tail_text = tail_seg.get("text", "")
        tail_tokens = count_tokens(tail_text)

        curr_text = "\n\n".join(
            s.get("text", "").strip() for s in curr_group
        )
        curr_tokens = count_tokens(curr_text)

        if curr_tokens + tail_tokens <= MAX_CHUNK_TOKENS:
            # Prepend tail segment to current group (deduplicated later in
            # the caller by segment identity, but for now we just overlap).
            new_group = [tail_seg] + curr_group
            out.append(new_group)
        else:
            out.append(curr_group)

    return out


def chunk_transcript_segments(
    segments: list[dict[str, Any]],
    *,
    target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    overlap_pct: int = DEFAULT_CHUNK_OVERLAP_PCT,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[ChunkDraft]:
    """Chunk transcript segments into :class:`ChunkDraft` with audio timing metadata.

    Each draft carries ``modal_features.audio_segment`` with ``start``,
    ``end``, and ``speaker`` keys.
    """
    target_tokens = max(256, min(target_tokens, MAX_CHUNK_TOKENS))
    overlap_pct = max(0, min(overlap_pct, 50))
    max_tokens = max(target_tokens, min(max_tokens, MAX_CHUNK_TOKENS))

    if not segments:
        return []

    chunk_groups = _bundle_segments_for_chunks(segments, target_tokens, max_tokens)
    chunk_groups = _with_overlap_segments(chunk_groups, overlap_pct)

    drafts: list[ChunkDraft] = []
    for group in chunk_groups:
        if not group:
            continue

        texts = [
            s.get("text", "").strip() for s in group if s.get("text", "").strip()
        ]
        if not texts:
            continue

        combined_text = "\n\n".join(texts)
        start_time = min(float(s.get("start", 0)) for s in group)
        end_time = max(float(s.get("end", 0)) for s in group)

        # Dominant speaker: most frequent in the group.
        speaker_counts: dict[str, int] = {}
        for s in group:
            spk = s.get("speaker", "UNKNOWN")
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        dominant_speaker = max(
            speaker_counts, key=speaker_counts.get, default="UNKNOWN"
        )

        modal_features: dict[str, dict[str, object]] = {
            "audio_segment": {
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "speaker": dominant_speaker,
            },
            "text": {
                "token_count": count_tokens(combined_text),
                "chunk_strategy": "segment-aware-bge-tokenizer",
            },
        }

        drafts.append(
            ChunkDraft(
                text=combined_text,
                token_count=count_tokens(combined_text),
                page=1,  # audio has no pages; use 1
                bbox=(0, 0, 1000, 1000),
                confidence=1.0,
                modal_features=modal_features,
            )
        )

    return drafts


# ---------------------------------------------------------------------------
# UIR builder
# ---------------------------------------------------------------------------


def _build_uir(
    doc_id: str,
    audio_path: Path,
    transcription: dict[str, Any],
    chunks: list[ChunkDraft],
    model: str,
    audio_meta: dict[str, Any],
    *,
    entities: list[dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
    topics: list[str] | None = None,
    vectors: Any | None = None,
) -> dict[str, Any]:
    """Assemble a UIRV1-compatible plain dict for an audio result."""
    now = datetime.now(timezone.utc)
    source_name = audio_path.name
    duration = transcription.get("duration_seconds")
    chunk_count = len(chunks)

    # Build chunk nodes (plain dicts, matching the UIR structure).
    chunk_nodes: list[dict[str, Any]] = []
    chunk_ids: list[str] = []
    for i, ck in enumerate(chunks):
        ck_id = deterministic_node_id("chunk", doc_id, i, ck.text[:64])
        chunk_ids.append(ck_id)

        mf = dict(ck.modal_features) if ck.modal_features else {}
        if vectors is not None and i < len(vectors.vectors):
            mf["vector"] = {
                "dim": vectors.dim,
                "model": "BAAI/bge-small-en-v1.5",
                "chunk_index": i,
                "embedding": [round(float(v), 6) for v in vectors.vectors[i]],
            }

        chunk_nodes.append(
            {
                "id": ck_id,
                "type": "chunk",
                "text": ck.text,
                "token_count": ck.token_count,
                "page": 1,
                "bounding_box": [0, 0, 1000, 1000],
                "confidence": 1.0,
                "modal_features": mf,
            }
        )

    # Wire chunk neighbours (consecutive).
    for i, cn in enumerate(chunk_nodes):
        mf = cn["modal_features"]
        if i > 0:
            mf["preceding_chunk_id"] = {"chunk_id": chunk_ids[i - 1]}
        if i < len(chunk_nodes) - 1:
            mf["following_chunk_id"] = {"chunk_id": chunk_ids[i + 1]}

    # Build structure root.
    root_children: list[dict[str, Any]] = []
    if chunk_nodes:
        root_children.append(
            {
                "id": deterministic_node_id("figure", doc_id, "transcription"),
                "type": "figure",
                "title": "Transcription",
                "children": chunk_nodes,
            }
        )

    return {
        "uiR_version": "1.0",
        "id": doc_id,
        "modal_type": "audio",
        "source": {
            "uri": audio_path.resolve().as_uri(),
            "filename": source_name,
            "format": audio_path.suffix.lstrip(".").upper() or "WAV",
            "route": "audio",
            "mime_type": _audio_mime_type(audio_path.suffix),
            "size_bytes": audio_path.stat().st_size if audio_path.exists() else 0,
            "checksum": "",  # TODO: compute SHA256
            "timestamp": now.isoformat(),
        },
        "metadata": {
            "title": f"Audio transcription: {source_name}",
            "author": None,
            "page_count": 1,
            "chunk_count": chunk_count,
            "language": transcription.get("language") or "en",
            "format": audio_path.suffix.lstrip(".").upper() or "WAV",
            "modal_features": {
                "audio": {
                    "duration_seconds": duration,
                    "sample_rate": audio_meta.get("sample_rate"),
                    "channels": audio_meta.get("channels"),
                    "language_detected": transcription.get("language"),
                    "language_probability": transcription.get(
                        "language_probability"
                    ),
                    "model": model,
                    "segments": transcription.get("segments", []),
                },
            },
        },
        "structure": {
            "root": {
                "id": deterministic_node_id("doc", doc_id),
                "type": "document",
                "title": f"Audio: {source_name}",
                "children": root_children,
                "intent_filter": None,
            },
        },
        "semantics": {
            "entities": [e for e in (entities or [])],
            "relationships": [r for r in (relationships or [])],
            "topics": [t for t in (topics or [])],
        },
        "provenance": {
            "extraction": {
                "model": model,
                "version": "1.0",
                "timestamp": now.isoformat(),
            },
            "normalization": {
                "version": "1.0",
                "timestamp": now.isoformat(),
            },
        },
    }


# ---------------------------------------------------------------------------
# UMR builder
# ---------------------------------------------------------------------------


def _build_umr(uir_dict: dict[str, Any]) -> str:
    """Render an audio-analysis UIR dict into a clean Markdown string."""
    src = uir_dict.get("source", {})
    meta = uir_dict.get("metadata", {})
    root = uir_dict.get("structure", {}).get("root", {})
    prov = uir_dict.get("provenance", {}).get("extraction", {})
    audio_meta = (meta.get("modal_features") or {}).get("audio", {})

    lines: list[str] = []
    lines.append(f"# Audio: {src.get('filename', 'unknown')}")
    lines.append("")

    fmt = src.get("format", "?")
    model = prov.get("model", "?")
    dur = audio_meta.get("duration_seconds")
    lang = audio_meta.get("language_detected")
    sr = audio_meta.get("sample_rate")
    ch = audio_meta.get("channels")
    ts = meta.get("date", "?")

    eyebrow_parts = [f"Format: {fmt}", f"Model: {model}"]
    if dur is not None:
        eyebrow_parts.append(f"Duration: {float(dur):.1f}s")
    if lang:
        eyebrow_parts.append(f"Language: {lang}")
    if sr:
        eyebrow_parts.append(f"Sample rate: {int(sr)} Hz")
    if ch:
        eyebrow_parts.append(f"Channels: {int(ch)}")
    eyebrow_parts.append(f"Analysed: {ts}")

    lines.append(f"> *{' · '.join(eyebrow_parts)}*")
    lines.append("")

    for fig in root.get("children", []):
        if fig.get("type") == "figure":
            fig_title = fig.get("title", "")
            lines.append(f"## {fig_title}")
            lines.append("")
            for chunk in fig.get("children", []):
                if chunk.get("type") == "chunk":
                    text = chunk.get("text", "")
                    mf = chunk.get("modal_features", {})
                    audio_seg = mf.get("audio_segment", {})
                    speaker = audio_seg.get("speaker", "UNKNOWN")
                    start = audio_seg.get("start", 0)
                    end = audio_seg.get("end", 0)

                    start_fmt = f"{int(start) // 60}:{int(start) % 60:02d}"
                    end_fmt = f"{int(end) // 60}:{int(end) % 60:02d}"

                    lines.append(f"> **{speaker} · {start_fmt} - {end_fmt}**")
                    if text:
                        lines.append(text)
                    lines.append("")

    if not any(
        c.get("type") == "chunk"
        for fig in root.get("children", [])
        for c in fig.get("children", [])
    ):
        lines.append("_No transcription extracted from this audio._")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_audio_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_id: str | None = None,
    language: str | None = None,
    dry_run: bool = False,
    on_progress: Any | None = None,
    include_semantics: bool = False,
) -> AudioPipelineResult:
    """Process a single audio file through the vLLM Whisper + diarization pipeline.

    Args:
        input_path: Path to the audio file (MP3, WAV, M4A, FLAC, OGG, AAC, WMA).
        output_dir: Directory to write ``{doc_id}.uir.json`` and
            ``{doc_id}.umr.md``.
        model_id: Whisper model ID override (default from ``WHISPER_MODEL_ID`` env).
        language: Optional language code override (ISO 639-1, e.g. "en").
        dry_run: If True, don't write output files (simulate only).
        on_progress: Optional callback ``fn(stage: str, percent: int)``.
        include_semantics: If True, emit NER entities/relationships in the UIR.

    Returns:
        An :class:`AudioPipelineResult` with paths and metadata.
    """
    t0 = time.monotonic()
    p = Path(input_path)
    out_dir = Path(output_dir)

    resolved_model_id = model_id or _WHISPER_MODEL_ID
    resolved_language = language or _WHISPER_LANGUAGE

    def _progress(stage: str, pct: int, **meta: Any) -> None:
        logger.info("audio_pipeline.stage %s (%d%%) meta=%s", stage, pct, meta)
        if on_progress is not None:
            try:
                on_progress(stage, pct, **meta)
            except Exception:
                pass

    # Derive deterministic doc ID from the audio file URI.
    from uir_pipeline.embed import derive_doc_id

    doc_id = derive_doc_id(p.resolve().as_uri())

    # Stage 1: validate and extract metadata.
    _progress("validate", 5)
    if not p.exists():
        return AudioPipelineResult(
            uir_id="",
            out_path=out_dir / "ERROR",
            umr_path=out_dir / "ERROR",
            transcription_length=0,
            chunk_count=0,
            entity_count=0,
            model_used=resolved_model_id,
            elapsed_seconds=time.monotonic() - t0,
            error=f"File not found: {p}",
        )

    audio_meta = _get_audio_metadata(p)

    # Stage 2: transcribe with vLLM Whisper.
    _progress("transcribe", 30)
    try:
        transcription = transcribe_audio(
            p,
            model_id=resolved_model_id,
            language=resolved_language,
        )
    except Exception as exc:
        logger.error("transcription failed: %s", exc)
        return AudioPipelineResult(
            uir_id=doc_id,
            out_path=out_dir / f"{doc_id}.uir.json",
            umr_path=out_dir / f"{doc_id}.umr.md",
            transcription_length=0,
            chunk_count=0,
            entity_count=0,
            model_used=resolved_model_id,
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )

    all_text = transcription.get("all_text", "")
    transcription_len = len(all_text)
    logger.info(
        "transcription: %d chars, %d segments",
        transcription_len,
        len(transcription.get("segments", [])),
    )

    # Stage 3: speaker diarization.
    _progress("diarize", 50)
    speaker_segments: list[dict[str, Any]] = []
    try:
        speaker_segments = diarize_audio(p)
    except Exception as exc:
        logger.warning("diarization failed (fail-soft): %s", exc)

    # Stage 4: align segments.
    _progress("align", 55)
    segments = align_segments(transcription.get("segments", []), speaker_segments)
    unique_speakers = len({s.get("speaker", "UNKNOWN") for s in segments})

    # Stage 5: chunking.
    _progress("chunk", 70)
    chunk_drafts = chunk_transcript_segments(segments)
    logger.info("chunked into %d chunks", len(chunk_drafts))

    # Stage 6: enrich (NER).
    _progress("enrich", 80)
    entity_count = 0
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    topics: list[str] = []
    if include_semantics and chunk_drafts:
        try:
            from uir_pipeline.enrich import enrich_chunks

            enrichment = enrich_chunks([c.text for c in chunk_drafts])
            entities = [
                {
                    "text": e.text,
                    "type": e.type,
                    "confidence": e.confidence,
                }
                for e in enrichment.entities
            ]
            relationships = [
                {
                    "from": r.from_text,
                    "to": r.to_text,
                    "type": r.type,
                    "confidence": r.confidence,
                }
                for r in enrichment.relationships
            ]
            topics = list(enrichment.topics)
            entity_count = len(entities)
        except Exception as exc:
            logger.warning("enrich failed (fail-soft): %s", exc)

    # Stage 7: embed.
    _progress("embed", 90)
    vectors = None
    if chunk_drafts:
        try:
            from uir_pipeline.embed import embed_texts

            vectors = embed_texts([c.text for c in chunk_drafts])
        except Exception as exc:
            logger.warning("embed failed (fail-soft): %s", exc)

    # Stage 8: build UIR.
    _progress("assemble_uir", 95)
    uir_dict = _build_uir(
        doc_id=doc_id,
        audio_path=p,
        transcription=transcription,
        chunks=chunk_drafts,
        model=resolved_model_id,
        audio_meta=audio_meta,
        entities=entities if include_semantics else None,
        relationships=relationships if include_semantics else None,
        topics=topics if include_semantics else None,
        vectors=vectors,
    )

    # Stage 9: build UMR.
    _progress("assemble_umr", 97)
    umr_text = _build_umr(uir_dict)

    # Stage 10: write outputs.
    out_path = out_dir / f"{doc_id}.uir.json"
    umr_path = out_dir / f"{doc_id}.umr.md"

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(uir_dict, indent=2), encoding="utf-8")
        umr_path.write_text(umr_text, encoding="utf-8")
        logger.info("wrote %s and %s", out_path.name, umr_path.name)
    else:
        logger.info(
            "dry-run: would write %s and %s", out_path.name, umr_path.name
        )

    _progress("done", 100)
    elapsed = time.monotonic() - t0

    return AudioPipelineResult(
        uir_id=doc_id,
        out_path=out_path,
        umr_path=umr_path,
        transcription_length=transcription_len,
        chunk_count=len(chunk_drafts),
        entity_count=entity_count,
        model_used=resolved_model_id,
        elapsed_seconds=round(elapsed, 3),
        language_detected=transcription.get("language"),
        duration_seconds=audio_meta.get("duration_seconds"),
        speaker_count=unique_speakers if speaker_segments else None,
    )


__all__ = [
    "AudioPipelineResult",
    "run_audio_pipeline",
    "transcribe_audio",
    "diarize_audio",
    "align_segments",
    "chunk_transcript_segments",
]
