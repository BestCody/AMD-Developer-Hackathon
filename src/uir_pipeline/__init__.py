"""uir_pipeline -- Phase 1 PDF \u2192 UIR pipeline package (PLAN.md \u00a77).

Submodules:
    -- uir_schema      Pydantic v2 models for the UIR JSON contract (Phase B).
    -- weaviate_store  Weaviate client helper + URL resolution (Phase C).
                       Phase K (``embed.py``) extends this with upsert
                       logic, but adds its own module name (``embed.py``).
    -- device          cuda > mps > cpu hardware selector (Phase D).
                       Eager-importable; every model module needs it.

Heavy-dep modules (``ingest``, ``ocr``, ``layout``, ``tables``, ``chunk``,
``enrich``, ``embed``, ``pipeline``) are NOT imported here -- they pull
in PyTorch, transformers, easyocr, spaCy, pypdf, etc. Import them
explicitly when you need them:

    from uir_pipeline.ingest import ingest, DocumentInput
    from uir_pipeline.ocr    import default_engine, OCREngine
    from uir_pipeline.layout import LayoutClassifier, LayoutLMv3Backend
"""
from uir_pipeline import uir_schema  # noqa: F401  (always available)
from uir_pipeline import weaviate_store  # noqa: F401  (Phase C; optional)
from uir_pipeline import device  # noqa: F401  (Phase D; eagerly imported)
