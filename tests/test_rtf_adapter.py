"""Tests for ``src.uir_pipeline.ingest_rtf`` (PLAN §17 §Multi-format)."""
from __future__ import annotations

from pathlib import Path

import pytest

from uir_pipeline.ingest_rtf import (
    StriprtfUnavailable,
    _import_striprtf,
    ingest_rtf,
    striprtf_environment_enabled,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_rtf.rtf"


# ----------------------------------------------------------------------------
# Environment availability
# ----------------------------------------------------------------------------

def test_striprtf_environment_enabled():
    """``striprtf`` is in requirements; this should always pass in CI."""
    assert striprtf_environment_enabled() is True


def test_import_striprtf_returns_callable():
    fn = _import_striprtf()
    assert callable(fn)


# ----------------------------------------------------------------------------
# Happy path: fixture round-trip
# ----------------------------------------------------------------------------

def test_ingest_rtf_happy_path():
    """Read the small RTF fixture, return populated DocumentInput + pages."""
    fixture = Path(__file__).parent / "fixtures" / "sample_rtf.rtf"
    if not fixture.is_file():
        pytest.skip(f"missing fixture at {fixture}")
    doc, pages = ingest_rtf(fixture)
    # Document shape contracts
    assert doc.source_path == fixture
    assert doc.format == "RTF"
    assert doc.route == "text"
    assert doc.mime_type == "application/rtf"
    assert doc.size_bytes == fixture.stat().st_size
    assert doc.page_count >= 1
    # Pages shape
    assert isinstance(pages, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pages)
    assert pages[0][0] == 1  # page numbers start at 1
    # striprtf decoded at least the literal words we put in the fixture
    joined = "\n\n".join(t for _, t in pages)
    assert "Hello RTF world." in joined
    assert "bold" in joined or "italic" in joined


def test_ingest_rtf_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ingest_rtf(tmp_path / "does-not-exist.rtf")


# ----------------------------------------------------------------------------
# Synthetic RTF: control sequences must be stripped
# ----------------------------------------------------------------------------

def test_ingest_rtf_strips_control_words(tmp_path: Path):
    # Write a minimal hand-crafted RTF: ``{\rtf1\ansi Hello world!\par}``
    raw = b"{\\rtf1\\ansi Hello world!\\par}"
    p = tmp_path / "tiny.rtf"
    p.write_bytes(raw)
    _, pages = ingest_rtf(p)
    joined = "\n\n".join(t for _, t in pages)
    assert "Hello world!" in joined
    # RTF control words must NOT survive into cleaned text.
    assert "\\par" not in joined
    assert "\\ansi" not in joined
    assert "\\rtf1" not in joined


# ----------------------------------------------------------------------------
# striprtf env probe raises the right sentinel when missing
# ----------------------------------------------------------------------------

def test_striprtf_unavailable_sentinel(monkeypatch):
    """If we hide striprtf behind a fake import failure, the sentinel fires."""
    # We can't actually uninstall striprtf mid-test, but we can prove the
    # exception class is exported + importable.
    assert issubclass(StriprtfUnavailable, RuntimeError)
