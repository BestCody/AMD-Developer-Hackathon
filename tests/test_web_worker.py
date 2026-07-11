"""test_web_worker.py -- warm-worker lifecycle + FD limit hygiene.

These tests run the worker in-process so they stay fast. They do not load
Docling or torch.
"""
from __future__ import annotations

import pytest

from uir_pipeline.web import _WarmWorker


def test_warm_worker_discard_is_repeatable():
    """Calling _discard twice must not raise (covers queue-leak cleanup)."""
    worker = _WarmWorker()
    # First discard on a fresh worker is a no-op (no proc yet).
    worker._discard()
    # Second discard is also a no-op.
    worker._discard()
    assert worker._proc is None


def test_warm_worker_discard_after_spawn_cleans_queues():
    """Spawn, then discard — queues close without leaking."""
    worker = _WarmWorker()
    worker._spawn()
    assert worker._proc is not None
    worker._discard()
    assert worker._proc is None
    worker._discard()
    assert worker._proc is None


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="resource module is Unix-only",
)
def test_create_app_raises_fd_limit_on_unix(tmp_path, monkeypatch):
    """create_app should raise the soft NOFILE limit to the hard cap."""
    import resource
    monkeypatch.setenv("SECRET_KEY", "test-secret-not-random")
    from uir_pipeline.web import create_app

    soft_before, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    app = create_app(
        upload_dir=tmp_path / "up",
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        execution="thread",
    )
    # Avoid linter warnings about unused app
    _ = app
    soft_after, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and soft_before < hard:
        assert soft_after == hard
    else:
        assert soft_after == soft_before
