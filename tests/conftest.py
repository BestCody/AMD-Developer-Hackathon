"""tests/conftest.py -- shared pytest fixtures for the UIR pipeline tests.

Provides:
    -- ``skip_if_no_weaviate`` fixture: used by tests that require a live
       Weaviate container (``tests/test_weaviate_store.py::TestWeaviateLive``)
       so CI without docker doesn't pollute with errors.
    -- ``tmp_data_dir``: per-test scratch dir for output / logs / fixtures.
    -- The ``slow`` pytest marker for integration tests (default skip in
       the suite's pytest.ini).
"""
from __future__ import annotations

import os
import shutil

# Default test fast_path to pdfplumber so the pytest run never pulls
# 2 GB of HuggingFace weights on CI. Individual tests can opt in to
# the docling branch explicitly via ``monkeypatch.setenv("UIR_FAST_PATH",
# "docling")`` or by passing ``fast_path="docling"`` to ``run()``
# directly. PLAN §17 §OCR follow-up.
os.environ.setdefault("UIR_FAST_PATH", "pdfplumber")
from pathlib import Path

import pytest


# ----------------------------------------------------------------------------
# Pytest configuration
# ----------------------------------------------------------------------------

def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark tests under tests/integration/ as ``slow``."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def skip_if_no_weaviate():
    """Skip the decorated test (or fixture) when Weaviate is unreachable.

    Used by :class:`tests.test_weaviate_store.TestWeaviateLive` to gate
    live integration tests behind an actual docker-compose-up Weaviate.
    """
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:18080/v1/meta", timeout=1) as r:
            ok = (r.status == 200)
    except Exception:
        ok = False
    if not ok:
        pytest.skip("Weaviate is not reachable at http://localhost:18080")


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """A scratch directory tree with ``input/``, ``output/``, ``logs/`` subdirs."""
    base = tmp_path / "data"
    (base / "input").mkdir(parents=True)
    (base / "output").mkdir(parents=True)
    (base / "logs").mkdir(parents=True)
    return base


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to ``tests/fixtures``. Returns the directory itself; creates it
    on first use if missing.
    """
    p = Path(__file__).parent / "fixtures"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(autouse=True)
def _reset_logging_handlers():
    """Avoid log-handler leakage between tests (configure() is idempotent
    but an autouse fixture gives belt-and-suspenders cleanup).
    """
    import logging
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_uir_owned", False):
            try:
                root.removeHandler(h)
                h.close()
            except Exception:
                pass


@pytest.fixture
def teardown_root_handlers():
    """Force-clear all uir-owned handlers after the test.

    Use when a test installs a per-doc handler that needs to be flushed.
    """
    import logging
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_uir_owned", False):
            try:
                root.removeHandler(h)
                h.close()
            except Exception:
                pass
