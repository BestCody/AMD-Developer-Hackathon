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

- The model can call `search` and `get_more_sources` to find more passages
  before answering.
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
  non-WAV formats)
- **vLLM** is Linux/CUDA only; on macOS the audio pipeline falls back to
  HuggingFace Transformers for Whisper inference

Docling and PyTorch can consume significant memory. On an 8 GB computer,
conversion may fail while loading the table model if other applications leave
only about 1 GB free. The error usually contains:

```text
DefaultCPUAllocator: not enough memory
```

Closing memory-heavy applications can help, but a low-memory Docling mode or a
lighter PDF fallback is the more reliable long-term solution.

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
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff` | Fireworks vision |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.wma` | vLLM Whisper + pyannote diarization |

Legacy `.doc`, `.ppt`, and `.xls` files are recognized but rejected because
they are not safely convertible by the current routes.

Image uploads require `FIREWORKS_API_KEY`. Without it, the upload is accepted
but the conversion job fails when the vision stage starts.

Audio uploads require vLLM (Linux) or Transformers (macOS fallback). Speaker
diarization is optional and falls back to "UNKNOWN" labels if pyannote.audio is
not installed or fails to load.

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
