"""video_pipeline -- video to UIR + UMR pipeline using audio extraction + Whisper + frame sampling + Florence-2.

This module is the high-level orchestrator for processing a single video
file through:
    1. ffmpeg audio extraction -> transcribe with Whisper (reuses audio_pipeline)
    2. ffmpeg frame sampling at adaptive intervals
    3. Florence-2 caption per sampled frame (reuses caption._get_florence2)
    4. Temporal fusion: audio chunks carry visual descriptions from frames
       that fall within their time window
    5. Enrich (spaCy NER), embed (BGE-small)
    6. Build UIRV1 with modal_type="video" and UMR Markdown

The total incremental memory cost is negligible: audio uses the existing
Whisper path, and Florence-2 is already loaded for image uploads.

Works on Darwin (macOS) via the same transformers Whisper fallback and
MPS-accelerated Florence-2 that the audio and image pipelines already use.
ffmpeg is required (same as the audio pipeline's pydub dependency).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
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
_WHISPER_LANGUAGE: str | None = os.environ.get("WHISPER_LANGUAGE") or None


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class VideoPipelineResult:
    """Shape returned by :func:`run_video_pipeline`."""

    uir_id: str
    out_path: Path
    umr_path: Path
    transcription_length: int
    chunk_count: int
    entity_count: int
    frame_count: int
    frame_descriptions: int
    model_used: str
    elapsed_seconds: float
    duration_seconds: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


def _ffprobe_json(path: Path) -> dict[str, Any]:
    """Run ffprobe and return the JSON stream info as a dict."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def _get_video_metadata(path: Path) -> dict[str, Any]:
    """Extract video metadata: duration, width, height, fps, has_audio.

    Uses ffprobe. Fail-soft: returns empty dict on any exception.
    """
    try:
        data = _ffprobe_json(path)
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", path.name, exc)
        return {}

    fmt = data.get("format", {})
    streams = data.get("streams", [])

    duration = None
    try:
        duration = float(fmt.get("duration", 0))
    except (ValueError, TypeError):
        pass

    width = height = fps = None
    has_audio = False
    for s in streams:
        if s.get("codec_type") == "video" and width is None:
            width = s.get("width")
            height = s.get("height")
            r_frame_rate = s.get("r_frame_rate", "0/1")
            try:
                num, den = r_frame_rate.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
        if s.get("codec_type") == "audio":
            has_audio = True

    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": has_audio,
    }


def _extract_audio(path: Path, output_wav: Path) -> bool:
    """Extract audio track to WAV (16kHz mono) via ffmpeg.

    Returns True on success.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            logger.warning(
                "ffmpeg audio extraction failed for %s: %s",
                path.name,
                result.stderr.decode("utf-8", errors="replace")[:200],
            )
            return False
        return output_wav.exists() and output_wav.stat().st_size > 0
    except Exception as exc:
        logger.warning("ffmpeg audio extraction exception for %s: %s", path.name, exc)
        return False


def _choose_interval(duration: float) -> float:
    """Adaptive frame interval based on video duration.

    Short (<60s): every 5s. Medium (60s-300s): every 10s. Long (>300s): cap at 20 frames total.
    """
    if duration <= 60:
        return 5.0
    if duration <= 300:
        return 10.0
    return max(30.0, duration / 20)


def _sample_frames(
    path: Path,
    output_dir: Path,
    *,
    interval_seconds: float = 5.0,
) -> list[dict[str, Any]]:
    """Extract frames at uniform intervals via ffmpeg.

    Returns a list of {timestamp, path} dicts sorted by timestamp.
    """
    pattern = output_dir / "frame_%04d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(path),
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", "2",
        str(pattern),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            logger.warning(
                "ffmpeg frame sampling failed for %s: %s",
                path.name,
                result.stderr.decode("utf-8", errors="replace")[:200],
            )
            return []
    except Exception as exc:
        logger.warning("ffmpeg frame sampling exception for %s: %s", path.name, exc)
        return []

    frames: list[dict[str, Any]] = []
    # ffmpeg names frames frame_0001.jpg, frame_0002.jpg, ...
    # The timestamp of frame N is (N-1) * interval_seconds
    for fpath in sorted(output_dir.glob("frame_*.jpg")):
        try:
            stem = fpath.stem  # "frame_0001"
            idx = int(stem.split("_")[-1])
            ts = (idx - 1) * interval_seconds
            frames.append({"timestamp": ts, "path": fpath})
        except (ValueError, IndexError):
            continue
    return frames


# ---------------------------------------------------------------------------
# Frame captioning via Florence-2
# ---------------------------------------------------------------------------


def _caption_frames(
    frames: list[dict[str, Any]],
    *,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run Florence-2 on each sampled frame.

    Reuses caption._get_florence2(). Returns [{timestamp, description, path}, ...].
    """
    if not frames:
        return []

    try:
        from uir_pipeline.caption import _get_florence2, DEFAULT_PROMPT
    except Exception as exc:
        logger.warning("Florence-2 caption module unavailable: %s", exc)
        return []

    try:
        processor, model = _get_florence2(device=device)
    except Exception as exc:
        logger.warning("Florence-2 model load failed: %s", exc)
        return []

    import torch
    from PIL import Image

    results: list[dict[str, Any]] = []
    for frame in frames:
        fpath = frame["path"]
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as exc:
            logger.warning("Could not open frame %s: %s", fpath, exc)
            continue

        try:
            inputs = processor(
                text=DEFAULT_PROMPT,
                images=img,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )

            description = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            # Post-process: remove the prompt prefix if present
            try:
                parsed = processor.post_process_generation(
                    description, task=DEFAULT_PROMPT, image_size=(img.width, img.height)
                )
                if isinstance(parsed, dict):
                    description = parsed.get(DEFAULT_PROMPT, description)
                else:
                    description = str(parsed) if parsed is not None else description
            except Exception as exc:
                logger.debug("post_process_generation failed for frame %s: %s", fpath, exc)
                # keep the raw description

            results.append({
                "timestamp": frame["timestamp"],
                "description": description.strip(),
                "path": fpath,
            })
        except Exception as exc:
            logger.warning("Florence-2 caption failed for frame %s: %s", fpath, exc)
            continue

    return results


# ---------------------------------------------------------------------------
# Temporal fusion: audio segments + visual frames -> chunks
# ---------------------------------------------------------------------------


def _fuse_modalities(
    audio_segments: list[dict[str, Any]],
    visual_frames: list[dict[str, Any]],
    *,
    target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    overlap_pct: int = DEFAULT_CHUNK_OVERLAP_PCT,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[ChunkDraft]:
    """Bundle audio segments into chunks, then prepend visual frames that fall
    within each chunk's time window.

    Each chunk gets modal_features.video_segment with
    {start, end, speaker, visual_frames: [{timestamp, description}]}.
    """
    from uir_pipeline.audio_pipeline import chunk_transcript_segments

    # First, chunk the audio segments using the existing audio pipeline logic.
    audio_chunks = chunk_transcript_segments(
        audio_segments,
        target_tokens=target_tokens,
        overlap_pct=overlap_pct,
        max_tokens=max_tokens,
    )

    if not audio_chunks:
        # No audio: create visual-only chunks, one per frame or grouped by time.
        return _visual_only_chunks(visual_frames, max_tokens=max_tokens)

    # Build visual frame lookup: for each frame, find which audio chunk(s) it falls into.
    # A frame at timestamp T falls into chunk C if chunk_start <= T <= chunk_end.
    drafts: list[ChunkDraft] = []
    for ck in audio_chunks:
        # Get the audio chunk's time range from modal_features
        mf = dict(ck.modal_features) if ck.modal_features else {}
        audio_seg = mf.get("audio_segment", {})
        ck_start = float(audio_seg.get("start", 0)) if audio_seg else 0.0
        ck_end = float(audio_seg.get("end", 0)) if audio_seg else 0.0
        speaker = audio_seg.get("speaker", "UNKNOWN") if audio_seg else "UNKNOWN"

        # Find visual frames in this time window
        window_frames = [
            f for f in visual_frames
            if ck_start <= f["timestamp"] <= ck_end
        ]

        # Build the combined text: visual descriptions first, then audio transcript
        visual_parts: list[str] = []
        for f in window_frames:
            ts = f["timestamp"]
            mins = int(ts) // 60
            secs = int(ts) % 60
            visual_parts.append(f"[Visual {mins}:{secs:02d}] {f['description']}")

        audio_text = ck.text.strip()

        combined_text = "\n\n".join(visual_parts + [audio_text]) if audio_text else "\n\n".join(visual_parts)

        # Build modal_features
        modal_features: dict[str, Any] = {
            "video_segment": {
                "start": round(ck_start, 3),
                "end": round(ck_end, 3),
                "speaker": speaker,
                "visual_frames": [
                    {
                        "timestamp": round(f["timestamp"], 3),
                        "description": f["description"],
                    }
                    for f in window_frames
                ],
            },
            "text": {
                "token_count": count_tokens(combined_text),
                "chunk_strategy": "video-fusion-bge-tokenizer",
            },
        }

        drafts.append(
            ChunkDraft(
                text=combined_text,
                token_count=count_tokens(combined_text),
                page=1,
                bbox=(0, 0, 1000, 1000),
                confidence=1.0,
                modal_features=modal_features,
            )
        )

    # Append any visual frames that fall after the last audio chunk
    if audio_chunks and visual_frames:
        last_ck = audio_chunks[-1]
        mf = dict(last_ck.modal_features) if last_ck.modal_features else {}
        audio_seg = mf.get("audio_segment", {})
        last_end = float(audio_seg.get("end", 0)) if audio_seg else 0.0

        trailing_frames = [
            f for f in visual_frames
            if f["timestamp"] > last_end
        ]
        if trailing_frames:
            # Add trailing frames to the last chunk (or create a new one if too many)
            visual_parts = []
            for f in trailing_frames:
                ts = f["timestamp"]
                mins = int(ts) // 60
                secs = int(ts) % 60
                visual_parts.append(f"[Visual {mins}:{secs:02d}] {f['description']}")
            combined_text = "\n\n".join(visual_parts)

            modal_features = {
                "video_segment": {
                    "start": round(last_end, 3),
                    "end": round(max(f["timestamp"] for f in trailing_frames), 3),
                    "speaker": "UNKNOWN",
                    "visual_frames": [
                        {
                            "timestamp": round(f["timestamp"], 3),
                            "description": f["description"],
                        }
                        for f in trailing_frames
                    ],
                },
                "text": {
                    "token_count": count_tokens(combined_text),
                    "chunk_strategy": "video-fusion-bge-tokenizer",
                },
            }
            drafts.append(
                ChunkDraft(
                    text=combined_text,
                    token_count=count_tokens(combined_text),
                    page=1,
                    bbox=(0, 0, 1000, 1000),
                    confidence=1.0,
                    modal_features=modal_features,
                )
            )

    return drafts


def _visual_only_chunks(
    visual_frames: list[dict[str, Any]],
    *,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[ChunkDraft]:
    """Create chunks from visual frames only (no audio).

    Groups frames into chunks that fit within max_tokens.
    """
    if not visual_frames:
        return []

    chunks: list[list[dict[str, Any]]] = []
    buf: list[dict[str, Any]] = []
    buf_tokens = 0

    for f in visual_frames:
        desc = f["description"]
        desc_tokens = count_tokens(desc)
        if desc_tokens > max_tokens:
            # Oversized single description
            if buf:
                chunks.append(list(buf))
                buf = []
                buf_tokens = 0
            chunks.append([f])
            continue

        if buf_tokens + desc_tokens <= max_tokens:
            buf.append(f)
            buf_tokens += desc_tokens
        else:
            if buf:
                chunks.append(list(buf))
            buf = [f]
            buf_tokens = desc_tokens

    if buf:
        chunks.append(buf)

    drafts: list[ChunkDraft] = []
    for group in chunks:
        if not group:
            continue
        visual_parts = []
        for f in group:
            ts = f["timestamp"]
            mins = int(ts) // 60
            secs = int(ts) % 60
            visual_parts.append(f"[Visual {mins}:{secs:02d}] {f['description']}")
        combined_text = "\n\n".join(visual_parts)

        start_time = min(f["timestamp"] for f in group)
        end_time = max(f["timestamp"] for f in group)

        modal_features = {
            "video_segment": {
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "speaker": "UNKNOWN",
                "visual_frames": [
                    {
                        "timestamp": round(f["timestamp"], 3),
                        "description": f["description"],
                    }
                    for f in group
                ],
            },
            "text": {
                "token_count": count_tokens(combined_text),
                "chunk_strategy": "video-visual-only-bge-tokenizer",
            },
        }

        drafts.append(
            ChunkDraft(
                text=combined_text,
                token_count=count_tokens(combined_text),
                page=1,
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
    video_path: Path,
    transcription: dict[str, Any],
    visual_frames: list[dict[str, Any]],
    chunks: list[ChunkDraft],
    model: str,
    video_meta: dict[str, Any],
    *,
    entities: list[dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
    topics: list[str] | None = None,
    vectors: Any | None = None,
) -> dict[str, Any]:
    """Assemble a UIRV1-compatible plain dict for a video result."""
    now = datetime.now(timezone.utc)
    source_name = video_path.name
    duration = video_meta.get("duration_seconds")
    chunk_count = len(chunks)
    frame_count = len(visual_frames)

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
                "title": "Transcription + Visual",
                "children": chunk_nodes,
            }
        )

    return {
        "uiR_version": "1.0",
        "id": doc_id,
        "modal_type": "video",
        "source": {
            "uri": video_path.resolve().as_uri(),
            "filename": source_name,
            "format": video_path.suffix.lstrip(".").upper() or "MP4",
            "route": "video",
            "mime_type": f"video/{video_path.suffix.lstrip('.').lower() or 'mp4'}",
            "size_bytes": video_path.stat().st_size if video_path.exists() else 0,
            "checksum": "",  # TODO: compute SHA256
            "timestamp": now.isoformat(),
        },
        "metadata": {
            "title": f"Video analysis: {source_name}",
            "author": None,
            "page_count": 1,
            "chunk_count": chunk_count,
            "language": transcription.get("language") or "en",
            "format": video_path.suffix.lstrip(".").upper() or "MP4",
            "modal_features": {
                "video": {
                    "duration_seconds": duration,
                    "width": video_meta.get("width"),
                    "height": video_meta.get("height"),
                    "fps": video_meta.get("fps"),
                    "frame_count": frame_count,
                    "audio_model": model,
                    "vision_model": "microsoft/Florence-2-base",
                },
            },
        },
        "structure": {
            "root": {
                "id": deterministic_node_id("doc", doc_id),
                "type": "document",
                "title": f"Video: {source_name}",
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
    """Render a video-analysis UIR dict into a clean Markdown string."""
    src = uir_dict.get("source", {})
    meta = uir_dict.get("metadata", {})
    root = uir_dict.get("structure", {}).get("root", {})
    prov = uir_dict.get("provenance", {}).get("extraction", {})
    video_meta = (meta.get("modal_features") or {}).get("video", {})

    lines: list[str] = []
    lines.append(f"# Video: {src.get('filename', 'unknown')}")
    lines.append("")

    fmt = src.get("format", "?")
    model = prov.get("model", "?")
    dur = video_meta.get("duration_seconds")
    lang = meta.get("language", "?")
    ts = meta.get("date", "?")
    frame_count = video_meta.get("frame_count", 0)
    width = video_meta.get("width")
    height = video_meta.get("height")
    fps = video_meta.get("fps")

    eyebrow_parts = [f"Format: {fmt}", f"Audio model: {model}", f"Vision model: microsoft/Florence-2-base"]
    if dur is not None:
        eyebrow_parts.append(f"Duration: {float(dur):.1f}s")
    if lang:
        eyebrow_parts.append(f"Language: {lang}")
    if width and height:
        eyebrow_parts.append(f"Resolution: {width}x{height}")
    if fps:
        eyebrow_parts.append(f"FPS: {fps:.2f}")
    if frame_count:
        eyebrow_parts.append(f"Frames sampled: {frame_count}")
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
                    video_seg = mf.get("video_segment", {})
                    speaker = video_seg.get("speaker", "UNKNOWN")
                    start = video_seg.get("start", 0)
                    end = video_seg.get("end", 0)
                    visual_frames = video_seg.get("visual_frames", [])

                    start_fmt = f"{int(start) // 60}:{int(start) % 60:02d}"
                    end_fmt = f"{int(end) // 60}:{int(end) % 60:02d}"

                    lines.append(f"> **{start_fmt} - {end_fmt}** · {speaker}")

                    # Render visual frames inline
                    for vf in visual_frames:
                        vts = vf.get("timestamp", 0)
                        vdesc = vf.get("description", "")
                        v_mins = int(vts) // 60
                        v_secs = int(vts) % 60
                        lines.append(f"> [Visual {v_mins}:{v_secs:02d}] {vdesc}")

                    lines.append("")
                    if text:
                        lines.append(text)
                    lines.append("")

    if not any(
        c.get("type") == "chunk"
        for fig in root.get("children", [])
        for c in fig.get("children", [])
    ):
        lines.append("_No content extracted from this video._")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_video_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_id: str | None = None,
    language: str | None = None,
    dry_run: bool = False,
    on_progress: Any | None = None,
    include_semantics: bool = False,
) -> VideoPipelineResult:
    """Process a single video file through the audio + frame + vision pipeline.

    Args:
        input_path: Path to the video file (MP4, AVI, MOV, WebM, MKV, etc.).
        output_dir: Directory to write ``{doc_id}.uir.json`` and
            ``{doc_id}.umr.md``.
        model_id: Whisper model ID override (default from ``WHISPER_MODEL_ID`` env).
        language: Optional language code override for Whisper (ISO 639-1, e.g. "en").
        dry_run: If True, don't write output files (simulate only).
        on_progress: Optional callback ``fn(stage: str, percent: int)``.
        include_semantics: If True, emit NER entities/relationships in the UIR.

    Returns:
        A :class:`VideoPipelineResult` with paths and metadata.
    """
    t0 = time.monotonic()
    p = Path(input_path)
    out_dir = Path(output_dir)

    resolved_model_id = model_id or _WHISPER_MODEL_ID
    resolved_language = language or _WHISPER_LANGUAGE

    def _progress(stage: str, pct: int, **meta: Any) -> None:
        logger.info("video_pipeline.stage %s (%d%%) meta=%s", stage, pct, meta)
        if on_progress is not None:
            try:
                on_progress(stage, pct, **meta)
            except Exception:
                pass

    from uir_pipeline.embed import derive_doc_id

    doc_id = derive_doc_id(p.resolve().as_uri())

    # Stage 0: validate ffmpeg availability.
    _progress("validate", 5)
    if not p.exists():
        return VideoPipelineResult(
            uir_id="",
            out_path=out_dir / "ERROR",
            umr_path=out_dir / "ERROR",
            transcription_length=0,
            chunk_count=0,
            entity_count=0,
            frame_count=0,
            frame_descriptions=0,
            model_used=resolved_model_id,
            elapsed_seconds=time.monotonic() - t0,
            error=f"File not found: {p}",
        )

    if not _ffmpeg_available():
        return VideoPipelineResult(
            uir_id=doc_id,
            out_path=out_dir / f"{doc_id}.uir.json",
            umr_path=out_dir / f"{doc_id}.umr.md",
            transcription_length=0,
            chunk_count=0,
            entity_count=0,
            frame_count=0,
            frame_descriptions=0,
            model_used=resolved_model_id,
            elapsed_seconds=time.monotonic() - t0,
            error="ffmpeg is not available on PATH. Install ffmpeg to process video files.",
        )

    video_meta = _get_video_metadata(p)
    duration = video_meta.get("duration_seconds")
    has_audio = video_meta.get("has_audio", False)

    # Work in a temporary directory for audio extraction and frame sampling.
    with tempfile.TemporaryDirectory(prefix="uir_video_") as tmpdir:
        tmp_path = Path(tmpdir)
        audio_path = tmp_path / "audio.wav"
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Stage 1: extract audio.
        _progress("extract_audio", 15)
        audio_extracted = False
        if has_audio:
            audio_extracted = _extract_audio(p, audio_path)
            if not audio_extracted:
                logger.warning("audio extraction failed for %s; falling back to visual-only", p.name)

        # Stage 2: sample frames.
        _progress("sample_frames", 25)
        interval = _choose_interval(duration or 60.0)
        frames = _sample_frames(p, frames_dir, interval_seconds=interval)
        logger.info(
            "sampled %d frames from %s at %.1fs interval",
            len(frames), p.name, interval,
        )

        # Stage 3: transcribe audio (if available).
        _progress("transcribe", 40)
        transcription: dict[str, Any] = {"segments": [], "language": None, "all_text": "", "duration_seconds": duration}
        if audio_extracted and audio_path.exists():
            try:
                from uir_pipeline.audio_pipeline import transcribe_audio
                transcription = transcribe_audio(
                    audio_path,
                    model_id=resolved_model_id,
                    language=resolved_language,
                )
            except Exception as exc:
                logger.warning("transcription failed for %s: %s", p.name, exc)
                transcription = {"segments": [], "language": None, "all_text": "", "duration_seconds": duration}

        all_text = transcription.get("all_text", "")
        transcription_len = len(all_text)
        logger.info(
            "transcription: %d chars, %d segments",
            transcription_len,
            len(transcription.get("segments", [])),
        )

        # Stage 4: speaker diarization (optional, same as audio pipeline).
        _progress("diarize", 50)
        speaker_segments: list[dict[str, Any]] = []
        if audio_extracted and audio_path.exists():
            try:
                from uir_pipeline.audio_pipeline import diarize_audio, align_segments
                speaker_segments = diarize_audio(audio_path)
                segments = align_segments(transcription.get("segments", []), speaker_segments)
            except Exception as exc:
                logger.warning("diarization failed (fail-soft): %s", exc)
                segments = transcription.get("segments", [])
        else:
            segments = transcription.get("segments", [])

        # Stage 5: caption frames with Florence-2.
        _progress("caption_frames", 60)
        visual_frames: list[dict[str, Any]] = []
        if frames:
            try:
                visual_frames = _caption_frames(frames)
            except Exception as exc:
                logger.warning("frame captioning failed (fail-soft): %s", exc)
        logger.info("captioned %d/%d frames", len(visual_frames), len(frames))

        # Stage 6: fuse modalities into chunks.
        _progress("fuse", 70)
        chunk_drafts = _fuse_modalities(
            segments,
            visual_frames,
        )
        logger.info("fused into %d chunks", len(chunk_drafts))

        # Stage 7: enrich (NER).
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
                    {"text": e.text, "type": e.type, "confidence": e.confidence}
                    for e in enrichment.entities
                ]
                relationships = [
                    {"from": r.from_text, "to": r.to_text, "type": r.type, "confidence": r.confidence}
                    for r in enrichment.relationships
                ]
                topics = list(enrichment.topics)
                entity_count = len(entities)
            except Exception as exc:
                logger.warning("enrich failed (fail-soft): %s", exc)

        # Stage 8: embed.
        _progress("embed", 90)
        vectors = None
        if chunk_drafts:
            try:
                from uir_pipeline.embed import embed_texts
                vectors = embed_texts([c.text for c in chunk_drafts])
            except Exception as exc:
                logger.warning("embed failed (fail-soft): %s", exc)

        # Stage 9: build UIR.
        _progress("assemble_uir", 95)
        uir_dict = _build_uir(
            doc_id=doc_id,
            video_path=p,
            transcription=transcription,
            visual_frames=visual_frames,
            chunks=chunk_drafts,
            model=resolved_model_id,
            video_meta=video_meta,
            entities=entities if include_semantics else None,
            relationships=relationships if include_semantics else None,
            topics=topics if include_semantics else None,
            vectors=vectors,
        )

        # Stage 10: build UMR.
        _progress("assemble_umr", 98)
        umr_text = _build_umr(uir_dict)

        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{doc_id}.uir.json"
            umr_path = out_dir / f"{doc_id}.umr.md"
            out_path.write_text(json.dumps(uir_dict, indent=2), encoding="utf-8")
            umr_path.write_text(umr_text, encoding="utf-8")
        else:
            out_path = out_dir / f"{doc_id}.uir.json"
            umr_path = out_dir / f"{doc_id}.umr.md"

    _progress("done", 100)
    elapsed = time.monotonic() - t0
    logger.info(
        "video_pipeline done in %.2fs: %d chunks, %d frames, %d chars",
        elapsed, len(chunk_drafts), len(visual_frames), transcription_len,
    )

    return VideoPipelineResult(
        uir_id=doc_id,
        out_path=out_path,
        umr_path=umr_path,
        transcription_length=transcription_len,
        chunk_count=len(chunk_drafts),
        entity_count=entity_count,
        frame_count=len(frames),
        frame_descriptions=len(visual_frames),
        model_used=resolved_model_id,
        elapsed_seconds=elapsed,
        duration_seconds=duration,
        error=None,
    )
