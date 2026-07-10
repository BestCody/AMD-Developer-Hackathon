# MonadLabs UIR Pipeline

MonadLabs converts documents, images, and text into a Universal Intermediate
Representation (UIR v1.0). The output contains structured chunks, source
metadata, optional embeddings, and a readable UMR Markdown companion file.

The repository includes:

- A Python conversion pipeline
- A command-line interface
- An authenticated browser console
- A Gemini-named document assistant with grounded answers and citations
- Optional Weaviate storage

> The console calls the assistant **Gemini**, but the current chat and image
> backends use Fireworks models. The UI name does not change the model provider.

## How it works

```text
input file
   |
   +-- PDF, DOCX, XLSX, HTML, EPUB, TEX -> Docling
   +-- PPTX                              -> python-pptx
   +-- text, Markdown, CSV, source code -> direct text extraction
   +-- images                            -> Fireworks vision
   |
   v
typed regions -> chunks -> enrichment -> embeddings -> UIR JSON + UMR Markdown
```

PDFs use Docling with the `pypdfium2` backend. Born-digital PDFs first run
without OCR. Scanned PDFs can be retried with OCR when `DOCLING_OCR=auto`.

## Browser console

The console supports this flow:

1. Create an account or sign in.
2. Drop a file into the upload area.
3. Watch the real pipeline stage and percentage.
4. On success, see a centered checkmark confirmation.
5. After 1.8 seconds, return to the upload area while the document remains in
   the Documents panel.
6. Open a converted document to view its UIR or UMR result.
7. Ask Gemini questions about converted documents. Answers include the source
   chunks sent to the model.

The interface is displayed at 75% scale to match the Aperture console design.

## Requirements

- Python 3.10-3.13
- About 4 GB of free memory for reliable Docling model loading
- Docker Desktop only if using Weaviate
- Tesseract only when OCR fallback is needed
- A Fireworks API key for image conversion and assistant responses

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

Legacy `.doc`, `.ppt`, and `.xls` files are recognized but rejected because
they are not safely convertible by the current routes.

Image uploads require `FIREWORKS_API_KEY`. Without it, the upload is accepted
but the conversion job fails when the vision stage starts.

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

| Module | Purpose |
|---|---|
| `pipeline.py` | Runs the complete conversion flow |
| `format_router.py` | Selects the extraction route for each input |
| `docling_extract.py` | Wraps Docling and validates complete conversion |
| `image_pipeline.py` | Converts images through Fireworks vision |
| `chunk.py` | Produces token-sized document chunks |
| `enrich.py` | Adds entities and relationships |
| `embed.py` | Creates BGE-small embeddings |
| `uir_schema.py` | Defines and validates UIR v1.0 |
| `umr.py` | Produces readable UMR Markdown |
| `chat.py` | Retrieves chunks and produces grounded answers |
| `auth.py` | Stores accounts and verifies passwords |
| `conversations.py` | Stores user conversations |
| `web.py` | Provides the Flask routes and isolated conversion worker |

Frontend files are under `static/console/`, the shared Aperture styles are
under `static/ds/`, and the page template is `templates/console.html`.

## Current limitations

- Docling may not fit comfortably on an 8 GB machine while browsers and other
  development tools are open.
- Image conversion and assistant answers depend on an external Fireworks API.
- The browser server keeps job state in memory. Restarting it clears the job
  list, although previously written output files remain on disk.
- The Flask development server is intended for local testing, not public
  deployment.
- AMD ROCm support is designed into device selection but has not been fully
  validated on the target AMD cloud hardware.

## Project references

- [PLAN.md](./PLAN.md) - implementation plan and decisions
- [PLAN_TIER3.md](./PLAN_TIER3.md) - image-aware pipeline work
- [INSTRUCTIONS.md](./INSTRUCTIONS.md) - original project requirements
- [docs/uir.schema.json](./docs/uir.schema.json) - exported UIR schema

## License

TBD.
