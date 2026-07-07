# Phase 1 PDF → UIR Pipeline (MVP)

Pipeline that ingests PDF documents and emits **Universal Intermediate Representation (UIR v1.0)** JSON plus chunk-level vector embeddings into a Weaviate index.

This is the MVP for an AMD-hackathon project (Phase 1, multimodal unification system). For the **full plan**, see [`PLAN.md`](./PLAN.md). For the **upstream spec**, see [`INSTRUCTIONS.md`](./INSTRUCTIONS.md).

> Status: 🔨 Phase A (Bootstrap) in progress. Repository initialized and pushed to GitHub: BestCody/AMD-Developer-Hackathon. See `PLAN.md` §9 for the per-phase exit criteria.

---

## Quickstart (dev, macOS Apple Silicon)

Requires Python 3.10–3.13 (Python 3.14 wheels lag for some ML libraries — see "Python" below), Docker Desktop (≥ 4 GB RAM), and the `tesseract` CLI binary on `PATH`.

```bash
# 1. Install system dependencies (Homebrew)
brew install tesseract

# 2. Create a venv + install
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Bring up Weaviate via Docker Compose (PLAN.md \u00a79 Phase C).
#    Cold pull (first time on this machine): downloads the ~150 MB
#    `cr.weaviate.io/.../weaviate:1.26.4` image (1-3 min on slow links).
#    Warm restart: ~30 s.
#    If the curl below fails immediately after `up -d`, wait ~30 s
#    and retry -- the container is still booting past the healthcheck's
#    `start_period`.
docker compose up -d
# Smoke test:
curl -fsS http://localhost:18080/v1/meta | head -20

# 4. Drop a PDF here:
cp /path/to/file.pdf data/input/

# 5. Run the pipeline (Phase L/M — currently a stub that prints args):
python pipeline.py data/input/file.pdf --output-data data/output/

# 6. Run the Phase A.5 de-risking spike (validates LayoutLMv3 on MPS):
python scripts/spike_layoutlmv3.py

# 7. Run the Phase N web UI (defaults to LAN-visible on :5000):
python web.py                                 # browse from anywhere on the LAN
HOST=127.0.0.1 python web.py                  # loopback-only (private dev)
PORT=8080 python web.py                       # custom port
```

The intended exit state is a working CLI that emits
`data/output/{doc_id}.uir.json` and upserts chunk embeddings into Weaviate.
The pipeline is being built phase-by-phase — see `PLAN.md` §9.

---

## Python compatibility

- **Target**: 3.10, 3.11, 3.12, 3.13.
- **3.14+**: many wheels exist, but PyTorch / transformers / easyocr /
  spacy have shipped 3.14 wheels late or in pre-release — installation
  may fail with `Could not find a version that satisfies the requirement`.
  If that happens: `brew install python@3.12`, then `python3.12 -m venv .venv`.

---

## First-run model downloads (one-time)

The first run downloads ~700 MB of weights to local caches:

| Model | Used by | Size | Cache dir |
|---|---|---|---|
| `microsoft/layoutlmv3-base` | `layout.py` | ~500 MB | `~/.cache/huggingface/` |
| EasyOCR English (`english_g2`) | `ocr.py` | ~100 MB | `~/.EasyOCR/model/` |
| `BAAI/bge-small-en-v1.5` | `embed.py` | ~133 MB | `~/.cache/huggingface/` |
| spaCy `en_core_web_sm` | `enrich.py` | ~40 MB | spacy's default |

---

## Hardware

- **Development (Phase 1 MVP)**: Apple M4 Pro (macOS). The pipeline
  prefers MPS GPU for `transformers`/`sentence-transformers`, falls back
  to CPU when MPS kernels are missing.
- **Production (Phase 2)**: AMD cloud GPUs (MI250X / MI300X, ROCm). Same
  code runs after switching to a `rocm/pytorch` Docker image and setting
  `CUDA_VISIBLE_DEVICES=0`.

See `PLAN.md` §4 for the AMD migration plan and §6 for the rationale
behind the EasyOCR / pdfplumber / BGE-small choices.

---

## Repo layout

See `PLAN.md` §7 for the canonical view. Short version:

```
phase1-pipeline/
├── PLAN.md                       ← plan & decision log
├── INSTRUCTIONS.md               ← upstream spec (read-only)
├── pipeline.py                   ← CLI entrypoint
├── src/uir_pipeline/             ← package modules
└── tests/                        ← unit + integration tests
```

---

## Known risks (recorded but currently OK)

- **easyocr + numpy 2.x.** easyocr 1.7.x ships numpy-2 support, but a few
  internal codepaths assume pre-2.x APIs. If you see `np.float was removed`
  or `np.bool was removed` at runtime, pin `numpy<2.0` in `requirements.txt`
  and `pip install --force-reinstall --no-deps easyocr`. Empirically we
  install cleanly on Python 3.14 / arm64 with `numpy-2.5.1`.
- **LayoutLMv3 on MPS.** text+bbox branch + forward pass validated by the
  spike. Multimodal branch (Detectron2) is deliberately out of scope (see
  PLAN.md §9 Phase A.5).
- **Python 3.13+ only.** All wheels resolved on Python 3.14.2 in mid-2026.
  Older Pythons (3.10–3.12) are theoretically supported but unverified.

## Tests

```bash
pytest tests/                                  # unit tests (no model downloads)
pytest -m "not slow" tests/                    # skip integration tests
pytest -m slow tests/integration/             # integration tests (require fixtures)
```

`@pytest.mark.slow` integration tests run only against real sample PDFs
in `tests/fixtures/sample_pdfs/`. The MVP covers three profile fixtures:
flat text, dense table, multi-column.

---

## License

TBD.
