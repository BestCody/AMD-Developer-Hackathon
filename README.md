# MonadLabs UIR Pipeline

MonadLabs converts documents, images, audio, and text into a Universal Intermediate
Representation (UIR v1.0). The output contains structured chunks, source
metadata, optional embeddings, and a readable UMR Markdown companion file.

The repository includes:

- A Python conversion pipeline
- A command-line interface
- An authenticated browser console with a file browser, global search, and
  multi-user chat
- A `@fireworks` document assistant with grounded answers, citations, and an
  agentic tool-calling loop
- Optional Weaviate storage

## How it works

```text
input file
   |
   +-- PDF, DOCX, XLSX, HTML, EPUB, TEX -> Docling
   +-- PPTX                              -> python-pptx
   +-- text, Markdown, CSV, source code -> direct text extraction
   +-- images                            -> Fireworks vision
   +-- audio (MP3, WAV, M4A, FLAC, etc.) -> vLLM Whisper + pyannote diarization
   +-- email (.eml, .msg)               -> MIME / Outlook parser
   +-- video (.mp4, .mov, etc.)          -> ffmpeg + Whisper + Florence-2 frame fusion
   |
   v
typed regions -> chunks -> enrichment -> embeddings -> UIR JSON + UMR Markdown
```

PDFs use Docling with the `pypdfium2` backend. Born-digital PDFs first run
without OCR. Scanned PDFs can be retried with OCR when `DOCLING_OCR=auto`.

Audio files are transcribed with vLLM-served Whisper (openai/whisper-small by
default), optionally speaker-diarized with pyannote.audio, chunked by utterance
boundary, enriched with NER, and embedded with BGE-small just like document
chunks. The resulting transcript includes per-chunk speaker labels and timestamps
in the UMR Markdown companion.

Video files are processed by extracting the audio track (transcribed via Whisper),
sampling frames at adaptive intervals, and captioning each frame with Florence-2.
Audio transcripts and visual descriptions are fused into time-aligned chunks, so
each chunk contains both what was said and what was seen in that time window.
This avoids the heavy memory requirements of dedicated video VLMs (18+ GB for
Qwen-Omni-3B) by reusing the existing audio and image pipelines.

## Browser console

The console is a single-page application with three tabs and a global
command-palette search overlay.

### Upload tab — file browser

1. Create an account or sign in.
2. Drop a file into the upload area or pick a folder from the left tree.
3. Watch the real pipeline stage and percentage on the file card.
4. Files open only on an explicit click, not automatically mid-upload.
5. Folders are server-persisted (SQLite) and survive restarts.
6. Click a finished file to open a tabbed detail pane: **Metadata**, **UMR**
   (Markdown), **UIR** (JSON), and **Chunks**.
7. PDFs render a thumbnail preview via the built-in `/api/thumb` endpoint.

### Fireworks tab — grounded Q&A

Ask questions about your converted documents. The assistant uses a Fireworks
chat model (MiniMax-M3) with a retrieval + agentic loop:

- The model is fully autonomous: it is given the list of your documents and
  must call `search` / `get_more_sources` to find relevant passages before
  answering.  No passages are pre-loaded.
- Answers are rendered as **Markdown** (bold, lists, code, tables) with
  **DOMPurify** sanitization.
- Every answer shows **tool-step chips** (e.g. "Searched 'invoices' — 5
  sources") and an expandable **citations** panel with the source chunks.

### Chats tab — multi-user messaging

1. Start a conversation by entering a teammate's email. An **autocomplete**
   dropdown suggests registered users as you type.
2. Message back and forth. Messages are stored in SQLite and polled every 4
   seconds.
3. Type `@fireworks <question>` to ask your own documents from inside a chat.
   The question and the assistant's answer are posted into the shared thread
   so both members see them.
4. You can also `@mention` a converted file (e.g. `@report.pdf`) to scope the
   assistant's search to that specific document. An autocomplete dropdown
   suggests matching files as you type.
4. Each conversation shows the peer's **full email**, a **Member / Pending**
   badge (whether they have signed up), and the last message preview.

### Global search (⌘/Ctrl+K from any tab)

A command-palette overlay searches **all** converted documents by **semantic
meaning + title priority** (BGE-small embeddings). Title-matching documents
rank above content-only matches. Results show the document title, page
number, and a scored snippet. Clicking a result jumps to the file in the
Upload tab.

The interface is displayed at 75% scale to match the Aperture console design.

## Requirements

- Python 3.10-3.13
- About 4 GB of free memory for reliable Docling model loading
- Docker Desktop only if using Weaviate
- Tesseract only when OCR fallback is needed
- A Fireworks API key for image conversion and assistant responses
- **ffmpeg** on PATH for audio metadata extraction (pydub uses it for
  non-WAV formats) and video frame sampling
- **vLLM** is Linux/CUDA only; on macOS the audio and video pipelines fall back to
  HuggingFace Transformers for Whisper inference
- `extract-msg` for `.msg` (Outlook) parsing; `.eml` uses the Python stdlib `email` module

Docling and PyTorch can consume significant memory. On an 8 GB computer,
conversion may fail while loading the table model if other applications leave
only about 1 GB free. The error usually contains:

```text
DefaultCPUAllocator: not enough memory
```

Closing memory-heavy applications can help, but a low-memory Docling mode or a
lighter PDF fallback is the more reliable long-term solution.

## Pipeline architecture decisions (why we built it this way)

### The UIR contract: one schema, any modality

We didn't want a separate pipeline per file type. Every input—PDF, image, audio, video, email—converges into the **same** UIR v1.0 schema:

- A `StructureNode` tree (sections, figures, tables, chunks)
- `ChunkNode` leaves with `{text, page, bounding_box, confidence, modal_features}`
- `modal_features` is a free-form dict per modality: `vector` for embeddings, `audio_segment` for timestamps, `video` for frame captions, `email` for headers

This means the **retrieval layer**, **enrichment layer**, and **agentic layer** are modality-agnostic. A video chunk and a PDF paragraph are scored the same way (BGE cosine + BM25-lite fallback).

### Why Docling as the single backend

We started with pdfplumber + a custom layout classifier, then replaced both with **IBM Docling** (MIT license). Docling emits pre-typed sections / tables / figures / math natively, so downstream chunks come out spatially-aware and column-correct instead of flattened prose. The previous pdfplumber path was retired because it couldn't preserve column structure on multi-column PDFs. We keep a single backend to reduce memory overhead and avoid silent divergence between test fixtures and production paths.

### Why Florence-2 instead of a heavier vision model

For images and video frames we use **microsoft/Florence-2-base** (~1.5 GB). The prompt is a single `<MORE_DETAILED_CAPTION>` tag, which is fast and produces deterministic, structured descriptions. We don't need a conversational VLM here; the caption becomes a `chunk.text` entry that flows through the same BGE embedding and NER pipeline as any document paragraph.

### Video: fusion, not a monolithic VLM

Dedicated video VLMs (e.g., Qwen-Omni-3B) require 18+ GB of GPU memory. We avoid that by **decomposing video into two pipelines we already have**:

1. **Audio track** → Whisper → timestamped transcript segments (same pipeline as audio files)
2. **Frame sampling** → adaptive intervals (5s for <60s, 10s for 60–300s, 30s for >300s, capped at 20 frames) → Florence-2 captions
3. **Fusion** — each audio chunk is annotated with visual frame descriptions that fall within its time window, producing a `ChunkNode` that contains both *what was said* and *what was seen*.

This reuses zero new model weights and peaks at ~4 GB GPU for Whisper + Florence-2 combined.

### Why the agent is autonomous (no pre-fetched context)

Most RAG systems embed the top-k passages into the prompt unconditionally. We deliberately removed that. The model is given the **document catalog** and two tools (`search`, `get_more_sources`) and must decide when to retrieve. This matches how a human researcher works: you don't dump 6 paragraphs on someone and ask them to answer; you let them search, then answer. The system prompt enforces the rule: *"No passages are pre-loaded for you."*

If the user @mentions a file (`@report.pdf`), the backend parses the mention, resolves it to the job ID, and narrows the tool search space to only that document. The model never sees the full file content; it only receives the chunks it explicitly retrieves.

### Why BGE-small + BM25-lite fallback

We use **BAAI/bge-small-en-v1.5** (384-dim) for dense embeddings. It's small enough to run on CPU, fast to batch, and produces strong semantic similarity for document retrieval. When BGE is unavailable (e.g., a cold-start machine without the model cached), we fall back to a **BM25-lite** scorer that tokenizes the query and chunks, computes a simplified TF-IDF score, and still ranks by relevance. The fallback guarantees the pipeline never breaks even if the embedding stage fails.

### Why SQLite instead of a document store for metadata

Jobs, folders, conversations, and auth are stored in **SQLite** (with WAL mode enabled). This gives us:
- Zero external dependencies for the core product
- Instant persistence on every job state change (no async flush risk)
- A single `.db` file that can be copied, inspected, or migrated
- The in-memory job registry is rebuilt from the database on startup, so uploads survive server restarts

Weaviate is optional and only used for the vector layer; all metadata stays in SQLite.

## How the AI agent works

### Retrieval pipeline

When a user asks a question, the backend walks every `ChunkNode` across every UIR JSON the user owns:

1. **Embed the query** with BGE-small (or tokenize for BM25-lite)
2. **Score every chunk** by cosine similarity (or text score)
3. **Title boost** — if the document title matches a query token, every passage in that document gets `+0.30` (this lifts title matches above content-only matches without breaking intra-document ordering)
4. **Floor filter** — chunks below `0.58` cosine are dropped (measured on a 267-chunk corpus; 0.58 sits below the worst answer-bearing chunk and above every off-topic query)
5. **Return top-k** (default 6 for chat, 8 for search) sorted best-first

### Agentic tool-calling loop

The Fireworks chat model (MiniMax-M3) receives a system prompt that teaches it to ground every claim in a `[n]` citation and to call tools when it needs more context. The loop works like this:

1. **Initial state** — user question + document catalog (no passages)
2. **Model decides** — if it needs evidence, it calls `search(query)` or `get_more_sources(query)`
3. **Tool executes** — the backend runs the retrieval pipeline above, appends the results to the conversation with continuous citation numbers, and returns control to the model
4. **Repeat** — up to 4 tool-call iterations; after the cap, `tool_choice="none"` forces a final answer
5. **Citation validation** — the backend strips any `[n]` markers that point at passages never supplied, so hallucinated citations are removed before the user sees them

### Why this beats "dump top-k into the prompt"

- **Token efficiency** — the model only pays for passages it actually needs, not 6 irrelevant chunks
- **Multi-hop reasoning** — the model can search for "revenue", then search for "Q3" after reading the first result, mimicking how a human would investigate
- **Citation honesty** — every claim is checked against the passages that were actually provided; hallucinated `[7]` when only 3 passages were given is stripped automatically

## Setup

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Copy-Item .env.example .env
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
```

Set at least these values in `.env` when using the related features:

```dotenv
FIREWORKS_API_KEY=your-key
SECRET_KEY=replace-with-a-long-random-value
DOCLING_OCR=auto
```

The root launcher reads the process environment. If your shell does not load
`.env` automatically, export the values before starting the server.

## Run the browser console

```powershell
python web.py
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050).

The server binds only to the current computer by default. To use a different
port:

```powershell
$env:PORT="8080"
python web.py
```

LAN access must be enabled explicitly because the console has accounts and
session cookies. Use TLS for any network that is not fully trusted.

## Run a conversion from the command line

Convert one file without Weaviate:

```powershell
python pipeline.py tests/fixtures/sample_pdfs/flat_text.pdf `
  --output-data data/output/ `
  --skip-weaviate
```

Convert a directory:

```powershell
python pipeline.py tests/fixtures/sample_pdfs/ `
  --output-data data/output/ `
  --skip-weaviate
```

Each successful conversion writes:

```text
doc_<id>.uir.json
doc_<id>.umr.md
```

## Supported formats

| Route | Formats | Extractor |
|---|---|---|
| PDF | `.pdf` | Docling with `pypdfium2` |
| Office/document | `.docx`, `.xlsx`, `.html`, `.tex`, `.epub` | Docling |
| Presentation | `.pptx` | `python-pptx` |
| Text | `.txt`, `.md`, `.csv`, `.tsv`, `.rtf`, `.ipynb`, source files | Direct extraction |
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`, `.avif`, `.heic`, `.heif` | Fireworks vision |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.wma` | vLLM Whisper + pyannote diarization |
| Video | `.mp4`, `.avi`, `.mov`, `.webm`, `.mkv`, `.flv`, `.wmv`, `.m4v` | ffmpeg audio + frame sampling + Whisper + Florence-2 |
| Email | `.eml`, `.msg` | MIME / Outlook parser, chunked as text |

Legacy `.doc`, `.ppt`, and `.xls` files are recognized but rejected because
they are not safely convertible by the current routes.

Image uploads require `FIREWORKS_API_KEY`. Without it, the upload is accepted
but the conversion job fails when the vision stage starts.

Audio uploads require vLLM (Linux) or Transformers (macOS fallback). Speaker
diarization is optional and falls back to "UNKNOWN" labels if pyannote.audio is
not installed or fails to load.

Video uploads require `ffmpeg` on PATH (same as audio). The video pipeline extracts
the audio track for Whisper transcription, samples frames at adaptive intervals,
and captions each frame with Florence-2. This is lightweight compared to dedicated
video VLMs: zero new model loads, ~4 GB total GPU for Whisper + Florence-2 on a
short video. On macOS, Whisper runs via Transformers and Florence-2 via MPS.

## Weaviate

Start the optional local vector database:

```powershell
docker compose up -d
```

The browser console skips Weaviate by default. The CLI uses it unless
`--skip-weaviate` is supplied.

## Tests

Run the normal suite:

```powershell
python -m pytest
```

Useful focused commands:

```powershell
python -m pytest -m "not slow"
python -m pytest -m slow
python -m pytest tests/test_console_assets.py
python -m pytest tests/test_web.py
python -m pytest tests/integration/
```

Some integration tests download model weights or require a running Weaviate
container.

## Main modules

### Backend (Python)

| Module | Purpose |
|---|---|
| `pipeline.py` | Runs the complete conversion flow |
| `format_router.py` | Selects the extraction route for each input |
| `docling_extract.py` | Wraps Docling and validates complete conversion |
| `image_pipeline.py` | Converts images through Fireworks vision |
| `audio_pipeline.py` | Transcribes audio via vLLM Whisper + pyannote diarization |
| `video_pipeline.py` | Extracts audio + samples frames from video; fuses Whisper transcripts with Florence-2 captions into time-aligned chunks |
| `chunk.py` | Produces token-sized document chunks |
| `enrich.py` | Adds entities and relationships |
| `embed.py` | Creates BGE-small embeddings |
| `search.py` | Semantic + title-priority passage search over UIR documents |
| `uir_schema.py` | Defines and validates UIR v1.0 |
| `umr.py` | Produces readable UMR Markdown |
| `chat.py` | Retrieves chunks, agentic tool-calling loop, grounded answers |
| `auth.py` | Stores accounts, verifies passwords, user search |
| `conversations.py` | Stores multi-user chat threads and messages |
| `library.py` | SQLite-backed folders and job persistence |
| `web.py` | Flask routes, isolated conversion worker, file browser API |

### Frontend (JSX)

Frontend files are under `static/console/`:

| File | Purpose |
|---|---|
| `app.jsx` | Root component, session bootstrap, upload orchestration |
| `FileBrowser.jsx` | Document grid, folder tree, dropzone, file detail pane |
| `FileCard.jsx` | Conversion progress card with stage and percentage |
| `FileTree.jsx` | Collapsible left-side folder tree |
| `FileDetail.jsx` | Tabbed detail: Metadata, UMR, UIR, Chunks, thumbnail |
| `CopilotChat.jsx` | Fireworks Q&A with markdown, citations, tool-step chips |
| `ChatsPanel.jsx` | Conversation list, thread, new-chat email autocomplete |
| `GlobalSearch.jsx` | Command-palette overlay: semantic search across documents |
| `IconRail.jsx` | Left navigation rail with Upload, Fireworks, Chats, Search |
| `Markdown.jsx` | `marked` + `DOMPurify` renderer for chat answers |
| `AuthScreens.jsx` | Sign-in and sign-up forms |
| `LucideIcon.jsx` | React-safe Lucide icon wrapper (avoids DOM-reconciliation errors) |
| `api.js` | Fetch wrapper for all backend endpoints |

The shared Aperture design-system styles are under `static/ds/`, and the page
template is `templates/console.html`.

## API overview

The console backend exposes these authenticated endpoints (in addition to the
auth routes above):

| Endpoint | Method | Description |
|---|---|---|
| `/api/run` | `POST` | Upload a file, start a conversion job |
| `/api/status/<id>` | `GET` | Poll job stage, percentage, and result |
| `/api/result/<id>` | `GET` | Full or intent-filtered UIR JSON |
| `/api/umr/<id>` | `GET` | Markdown UMR companion |
| `/api/download/<id>` | `GET` | Download the full UIR JSON |
| `/api/thumb/<id>` | `GET` | PNG thumbnail of the first PDF page |
| `/api/jobs` | `GET` / `PATCH` / `DELETE` | List, move, or delete a job |
| `/api/folders` | `GET` / `POST` / `PATCH` / `DELETE` | Folder CRUD |
| `/api/search` | `POST` | Semantic + title-priority passage search |
| `/api/chat` | `POST` | Grounded Q&A with tool-calling agent loop |
| `/api/conversations` | `GET` / `POST` | List or start a chat thread |
| `/api/conversations/<id>/messages` | `GET` / `POST` | Read or send messages |
| `/api/users/search` | `GET` | Autocomplete: registered users by email prefix |

## Current limitations

- Docling may not fit comfortably on an 8 GB machine while browsers and other
  development tools are open. On macOS the default soft FD limit is 256; the
  server now raises this automatically, but very large model loads may still
  strain memory.
- Image conversion and assistant answers depend on an external Fireworks API.
- The browser server persists jobs and folders in SQLite, so they survive
  restarts. The in-memory job queue is rebuilt on startup from the database.
- The Flask development server is intended for local testing, not public
  deployment.
- AMD ROCm support is designed into device selection but has not been fully
  validated on the target AMD cloud hardware.
- vLLM (Whisper inference) is Linux/CUDA only. macOS falls back to HuggingFace
  Transformers, which is slower and uses more memory for the same model size.
- pyannote.audio speaker diarization requires a HuggingFace token for some
  model weights; if unavailable, the pipeline falls back to "UNKNOWN" speaker
  labels.

## Project references

- [PLAN.md](./PLAN.md) - implementation plan and decisions
- [PLAN_TIER3.md](./PLAN_TIER3.md) - image-aware pipeline work
- [INSTRUCTIONS.md](./INSTRUCTIONS.md) - original project requirements
- [docs/uir.schema.json](./docs/uir.schema.json) - exported UIR schema

## License

TBD.
