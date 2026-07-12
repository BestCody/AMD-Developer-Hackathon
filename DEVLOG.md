# Devlog

## 2026-07-12 — Autonomous Assistant, @mentions, Email, AVIF/HEIC, Sidebar Redesign

### Backend

**Autonomous Assistant**
- The Fireworks assistant no longer receives 6 pre-fetched source chunks. Instead, it is given a catalog of the user's documents and must call `search` or `get_more_sources` to retrieve passages before answering. This is enforced by the updated system prompt: "You MUST call `search` or `get_more_sources` to find relevant passages before answering. No passages are pre-loaded for you."
- `_grounded_answer` in `web.py` passes an empty `contexts` list when `docs` is present, forcing the agentic loop to run.
- The `chat.py` `answer()` function filters `docs` by `job_ids` before entering the tool loop, so any scope narrowing is respected across tool calls.

**@mention File Scoping**
- Users can type `@filename` in both the Fireworks chat tab and the Chats thread input. The backend parses the mention, resolves it to the user's DONE job(s), and scopes the agent's search to only those documents.
- A `_parse_file_mentions` helper in `web.py` strips `@filename` tokens from the text before the model sees them, then builds a `job_ids` set for `_grounded_answer`.
- Both `/api/chat` and `/api/conversations/<cid>/messages` (for `@fireworks` commands) now run this parser.

**Email Format Support**
- New module: `src/uir_pipeline/email_pipeline.py`.
- `.eml` files are parsed with Python's stdlib `email` module. `.msg` files require `extract-msg` (added to `requirements.txt`).
- Extracts `subject`, `from`, `to`, `date`, and the body text (plain text preferred; HTML fallback with a lightweight tag stripper).
- Text flows through the existing pageless pipeline: paginate → chunk → enrich → embed → UIR/UMR.
- Email metadata is stored in `metadata.modal_features.email` and rendered in the UMR eyebrow.

**AVIF / HEIC / HEIF Image Support**
- Added `.avif`, `.heic`, `.heif` to `_IMAGE_EXTENSIONS` in `format_router.py` and to `_SUPPORTED_EXTENSIONS` in `fireworks_vision.py`.
- Added MIME types (`image/avif`, `image/heic`, `image/heif`) to the `/api/thumb` and `/api/original` endpoints so browsers can stream them.
- Frontend icon maps updated to show the `image` icon for these extensions.

**UIR Schema Update**
- Added `modal_features: dict[str, dict[str, Any]]` to the `Metadata` Pydantic model so per-modality metadata (audio, video, email) can be stored without schema churn.
- Regenerated `docs/uir.schema.json` to match.

**UMR Rendering**
- Added email eyebrow rendering: `subject`, `from`, `to`, `date` appear alongside the title when the source format is `EML` or `MSG`.

### Frontend

**Sidebar Redesign (IconRail)**
- "Chats" is now a first-class tab in the sidebar with a `message-circle` icon, instead of being hidden behind the profile avatar.
- The profile avatar is replaced by an **Account** dropdown: clicking the avatar opens a small menu showing the user's name/email and a **Sign out** button.

**@mention Autocomplete**
- Both the **Fireworks chat** and **Chats thread** inputs now show a dropdown of matching converted files when the user types `@` followed by text.
- Selecting a file inserts `@filename` into the input. The dropdown matches on filename substring (case-insensitive) and shows up to 8 results.

**Search Icon Fix**
- The search icon in the Chats list search bar is now vertically centered (`top: 50%; transform: translateY(-50%)`) and has `pointerEvents: "none"`, so clicking it focuses the input instead of intercepting the click.

### Tests
- Updated `test_web_auth.py` and `test_web_conversations.py` to reflect the new autonomous assistant flow (no pre-fetched `paths` to capture).
- Updated `test_fireworks_vision.py` to use `.xyz` as the unsupported extension test case (since `.heic` is now supported).
- Added `tests/test_email_pipeline.py` with `.eml` extraction, HTML fallback, and `_strip_html` tests.
- **802 tests pass**, 4 skipped (Weaviate), 15 deselected (pre-existing launcher test).

---

## 2026-07-12 — Video Pipeline & Visual Prominence in UI

### Backend

**Video Modality Pipeline**
- New module: `src/uir_pipeline/video_pipeline.py`.
- For video files (`.mp4`, `.avi`, `.mov`, `.webm`, `.mkv`, `.flv`, `.wmv`, `.m4v`):
  1. Extracts audio via `ffmpeg` and runs it through the existing Whisper transcription pipeline.
  2. Samples frames at adaptive intervals: 5s for <60s, 10s for 60–300s, 30s for >300s, capped at 20 frames.
  3. Captions each sampled frame with Florence-2 (`microsoft/Florence-2-base`).
  4. Fuses audio transcript chunks and visual frame descriptions into time-aligned `ChunkDraft`s. Each chunk contains both what was said and what was seen in that time window.
- Added `FormatRoute.VIDEO` to `format_router.py` and wired it into `pipeline.py`.
- Added `modal_type="video"` to `uir_schema.py`.

**UMR Rendering for Video**
- `_render_chunk` in `umr.py` strips `[Visual ...]` lines from the body text when `visual_frames` exists to avoid duplication.
- Renders visual frames as distinct annotation blocks with timestamp and description.
- Video metadata (duration, resolution, fps, frames sampled) appears in the eyebrow.

### Frontend

**Video & Audio Playback**
- `FileDetail.jsx` `BigPreview` now renders a native `<video controls poster=...>` element for video files and `<audio controls>` for audio files.
- The original file is served via the new `/api/original/<job_id>` endpoint so media players can stream the source bytes.
- `FileCard.jsx` maps video extensions to `clapperboard` and audio to `music` icons.

**Chunks Tab Enhancements**
- Visual frames are rendered as blue annotation blocks with a `clapperboard` icon and timestamp (`MM:SS`).
- Chunk headers show time range + speaker badge for video/audio chunks.

**Thumbnail Generation for Video**
- `/api/thumb/<job_id>` extracts a single frame via `ffmpeg` (`-ss 00:00:01`) for video files and returns it as PNG.

### Tests
- Added `tests/test_video_pipeline.py` (22 tests) covering fusion, chunking, UIR/UMR building, metadata, and error paths.
- Added `tests/test_video_route.py` (3 tests) for pipeline dispatch.

---

## 2026-07-11 — Audio Modality Pipeline (vLLM Whisper + Speaker Diarization)

### Backend

**Audio Pipeline**
- New module: `src/uir_pipeline/audio_pipeline.py`.
- For audio files (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.wma`):
  1. **Transcription**: vLLM (`openai/whisper-small` by default) produces timestamped segments.
  2. **Speaker Diarization**: `pyannote.audio` (`speaker-diarization-3.1`) assigns speaker labels to time ranges.
  3. **Alignment**: Each transcript segment gets its dominant speaker by time overlap.
  4. **Chunking**: `_bundle_segments_for_chunks` groups segments into ~384-token chunks, preserving speaker boundaries.
  5. **Enrichment + Embed**: Chunks flow through existing spaCy NER and BGE-small embedding stages.
- vLLM is Linux/CUDA only; on macOS the pipeline falls back to HuggingFace Transformers (`automatic-speech-recognition` pipeline) for Whisper inference.
- `pyannote.audio` is optional: if unavailable, diarization returns `[]` and all speakers default to `UNKNOWN`.

**Format Router**
- Added `FormatRoute.AUDIO` and `_AUDIO_EXTENSIONS`.
- Wired into `pipeline.py` with `AudioAnalysisError` and lazy imports so heavy audio deps don't load for document/image routes.

**UIR / UMR**
- `modal_type="audio"` added to `uir_schema.py`.
- `ChunkNode.modal_features.audio_segment` stores `{start, end, speaker}` per chunk.
- UMR renders speaker labels and timestamps inline: `> **SPEAKER_00** · 0:00 - 0:15`.

### Configuration
- Added `WHISPER_MODEL_ID`, `WHISPER_LANGUAGE`, `DIARIZATION_MODEL_ID`, and `DIARIZATION_DISABLE` to `.env.example`.
- Added `vllm>=0.7.3`, `pyannote.audio>=3.1`, and `pydub>=0.25` to `requirements.txt`.

---

## 2026-07-11 — Console Fixes: SQLite Leak, React DOM, Chats Multi-User

### Backend Fixes
- **SQLite Connection Leak**: `auth.py`, `library.py`, and `conversations.py` used a `@contextmanager` that only committed but never closed the connection. On macOS (default soft FD limit 256), this caused `EMFILE` (Too many open files) under load. Fixed by adding `try/finally: conn.close()`.
- **macOS FD Limit**: The web server now automatically raises the soft FD limit to 4096 on startup.

### Frontend Fixes
- **React DOM NotFoundError**: `lucide.createIcons()` was being called globally and replaced `<i>` elements with SVGs, which React's reconciler couldn't track. Replaced with a React-safe `LucideIcon` component that renders icons via `useLayoutEffect` and scoped `createIcons()` on a container ref.

### Chats Panel
- Real multi-user messaging: 1:1 threads between two registered users, identified by email.
- `@fireworks <question>` inside a chat thread runs the grounded retrieval system on the *sender's* documents and posts the answer as a shared assistant message visible to both members.
- Conversations are stored in SQLite and polled every 4 seconds.
- User search autocomplete when starting a new chat.

---

## 2026-07-11 — File Browser, Global Search, Agentic Chat

### Console Features
- **File Browser**: Google-Drive-like grid with folders, file cards showing conversion stage/percentage, and a tabbed detail pane (Metadata, UMR, UIR, Chunks).
- **Global Search**: ⌘/Ctrl+K command-palette overlay that searches all converted documents by semantic meaning + title priority (BGE-small embeddings).
- **Agentic Chat**: Fireworks chat model (MiniMax-M3) with an OpenAI-style tool-calling loop. The model can call `search` or `get_more_sources` to broaden its context before answering. Citations are validated and stripped if hallucinated.
- **Multi-upload**: Batch upload with optimistic UI rows, real polling via `/api/status`, and server-persisted folders in SQLite.

### Pipeline
- **Multi-format**: PDF (Docling), DOCX/XLSX/HTML/EPUB/TEX (Docling), PPTX (python-pptx native), TXT/MD/CSV/code (text lane), images (Fireworks vision), RTF (striprtf), IPYNB (JSON cell extraction).
- **Figure Captioning**: Florence-2 captions figure regions extracted by Docling, producing `caption` chunks that share the BGE embedding pipeline.
- **UMR (Universal Markdown Representation)**: A clean Markdown companion file emitted alongside every UIR JSON, optimized for agent context windows.
