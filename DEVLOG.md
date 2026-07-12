# Devlog

## 2026-07-12 — Autonomous Assistant, @mentions, Email, AVIF/HEIC, Sidebar Redesign

**Autonomous Assistant** — Agent no longer receives 6 pre-fetched chunks. Given document catalog; must call `search`/`get_more_sources` to retrieve passages before answering. `chat.py` and `web.py` updated; `answer()` filters `docs` by `job_ids` so tool calls respect scope.

**@mention File Scoping** — Typing `@filename` in Fireworks chat or Chats threads scopes retrieval to that document. `_parse_file_mentions()` in `web.py` resolves mentions to `job_ids`, strips them from prompt, and passes to `_grounded_answer`. Frontend autocomplete added to both inputs (substring match, max 8 results).

**Email Support** — New `email_pipeline.py`. `.eml` via stdlib `email`; `.msg` via `extract-msg` (added to `requirements.txt`). Extracts subject, from, to, date, body text. Flows through text lane: paginate → chunk → enrich → embed. Metadata stored in `metadata.modal_features.email`; rendered in UMR eyebrow.

**AVIF/HEIC/HEIF** — Added to `_IMAGE_EXTENSIONS` and `fireworks_vision.py` supported extensions. MIME types added to `/api/thumb` and `/api/original`. Frontend icon maps show `image` icon.

**UIR Schema** — Added `modal_features: dict` to `Metadata` model; regenerated `docs/uir.schema.json`.

**Sidebar Redesign** — Chats is now a first-class sidebar tab (`message-circle` icon). Profile avatar replaced by Account dropdown (name/email + Sign out).

**Search Icon Fix** — Vertically centered with `pointerEvents: "none"` so clicks pass through to input.

**Tests** — 802 pass. Updated `test_web_auth.py`, `test_web_conversations.py`, `test_fireworks_vision.py`. Added `test_email_pipeline.py`.

---

## 2026-07-12 — Video Pipeline & Visual Prominence

**Video Pipeline** — `video_pipeline.py`: ffmpeg extracts audio → Whisper transcription. Frames sampled adaptively (5s/10s/30s, max 20) → Florence-2 captions. Audio and visual descriptions fused into time-aligned chunks. `FormatRoute.VIDEO` added; `modal_type="video"` in schema.

**Frontend Playback** — `BigPreview` renders `<video controls>` and `<audio controls>`. `/api/original/<job_id>` serves source bytes. Chunks tab renders visual frames as blue timestamped blocks. Video thumbnails via ffmpeg frame extraction.

**UMR** — Strips `[Visual ...]` lines from body when `visual_frames` exists; renders video metadata eyebrow.

---

## 2026-07-11 — Audio Pipeline (Whisper + Diarization)

**Audio Pipeline** — `audio_pipeline.py`: vLLM Whisper transcribes; `pyannote.audio` diarizes speakers. Segments aligned by time overlap, chunked by speaker boundary, then enriched + embedded. macOS falls back to HuggingFace Transformers. Diarization optional (defaults to `UNKNOWN`).

**UIR/UMR** — `modal_type="audio"`; `modal_features.audio_segment` stores `{start, end, speaker}`. UMR renders speaker labels and timestamps inline.

---

## 2026-07-11 — Console Fixes & Multi-User Chats

**Fixes** — SQLite connection leak resolved (`conn.close()` in `auth.py`, `library.py`, `conversations.py`) fixing EMFILE on macOS. React DOM `NotFoundError` fixed by replacing global `lucide.createIcons()` with React-safe `LucideIcon` using `useLayoutEffect`.

**Chats** — Real 1:1 threads by email. `@fireworks` inside a thread runs grounded retrieval on sender's documents and posts answer as shared assistant message. SQLite-backed, 4s polling.

---

## 2026-07-11 — File Browser, Global Search, Agentic Chat

**Console** — File browser grid with folders, tabbed detail pane (Metadata/UMR/UIR/Chunks), global search (⌘K, semantic + title priority), agentic chat with tool-calling loop (`search`/`get_more_sources`), multi-upload with optimistic rows and real polling.

**Pipeline** — Multi-format routing (PDF/DOCX/PPTX/TXT/MD/CSV/RTF/IPYNB/code/images). Florence-2 figure captioning. UMR companion markdown for agent consumption.
