"""Tests for email extraction pipeline (.eml, .msg)."""
from __future__ import annotations

import pytest
from pathlib import Path

from uir_pipeline.email_pipeline import extract_email, _strip_html


def test_extract_eml(tmp_path: Path) -> None:
    src = tmp_path / "test.eml"
    src.write_text(
        'Subject: Hello\r\n'
        'From: alice@example.com\r\n'
        'To: bob@example.com\r\n'
        'Date: Mon, 01 Jan 2024 00:00:00 +0000\r\n'
        'Content-Type: text/plain\r\n'
        '\r\n'
        'Hello, world!',
        encoding="utf-8",
    )
    result = extract_email(src)
    assert result["text"] == "Hello, world!"
    meta = result["metadata"]
    assert meta["subject"] == "Hello"
    assert meta["from"] == "alice@example.com"
    assert meta["to"] == "bob@example.com"


def test_extract_eml_html_fallback(tmp_path: Path) -> None:
    src = tmp_path / "test_html.eml"
    src.write_text(
        'Subject: HTML Mail\r\n'
        'From: sender@example.com\r\n'
        'To: receiver@example.com\r\n'
        'Content-Type: text/html\r\n'
        '\r\n'
        '<html><body><p>Hello HTML</p></body></html>',
        encoding="utf-8",
    )
    result = extract_email(src)
    assert "Hello HTML" in result["text"]


def test_strip_html() -> None:
    assert _strip_html("<p>Hello</p>") == "Hello"
    assert _strip_html("a &amp; b") == "a & b"


def test_msg_requires_extract_msg(tmp_path: Path) -> None:
    extract_msg = pytest.importorskip("extract_msg")
    src = tmp_path / "test.msg"
    # Create a minimal fake .msg file; extract_msg will likely fail on a
    # fake file, but we test that it *requires* the module.
    src.write_bytes(b"fake msg")
    with pytest.raises(ValueError):
        extract_email(src)
