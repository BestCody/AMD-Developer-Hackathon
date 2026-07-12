"""email_pipeline -- extract text and metadata from email files.

``.eml`` (MIME/rfc822) uses the stdlib ``email`` module.
``.msg`` (Outlook) uses ``extract_msg`` when available.

Both produce a flat text body + header metadata that flows through the
existing text route (paginate_pageless → chunk → enrich → embed).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_email(path: Path) -> dict[str, Any]:
    """Return ``{"text": str, "metadata": dict}`` from ``.eml`` or ``.msg``.

    Raises ``ValueError`` when the file cannot be parsed or ``extract_msg``
    is required but missing.  The caller (``pipeline.run``) catches this and
    surfaces it as an error result so the upload is not swallowed silently.
    """
    ext = path.suffix.lower()
    if ext == ".eml":
        return _extract_eml(path)
    if ext == ".msg":
        return _extract_msg(path)
    raise ValueError(f"unsupported email extension: {ext}")


def _extract_eml(path: Path) -> dict[str, Any]:
    import email
    from email import policy

    raw = path.read_bytes()
    msg = email.message_from_bytes(raw, policy=policy.default)

    subject = str(msg.get("Subject", ""))
    from_ = str(msg.get("From", ""))
    to = str(msg.get("To", ""))
    date = str(msg.get("Date", ""))

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    body = part.get_content()
                except Exception:
                    pass
                break
            elif ctype == "text/html" and not body:
                # Fallback to HTML if no plain text; strip tags crudely.
                try:
                    html = part.get_content()
                    body = _strip_html(html)
                except Exception:
                    pass
    else:
        try:
            body = msg.get_content()
        except Exception:
            pass

    return {
        "text": body.strip() if body else "",
        "metadata": {
            "subject": subject,
            "from": from_,
            "to": to,
            "date": date,
        },
    }


def _extract_msg(path: Path) -> dict[str, Any]:
    try:
        import extract_msg
    except ImportError as exc:
        raise ValueError(
            "`.msg` files require `extract-msg`. Install it with:\n"
            "    pip install extract-msg>=0.50"
        ) from exc

    try:
        msg = extract_msg.Message(str(path))
        body = msg.body or ""
        subject = msg.subject or ""
        from_ = msg.sender or ""
        to = msg.to or ""
        date = str(msg.date) if msg.date else ""
    except Exception as exc:
        raise ValueError(f"could not parse {path.name}: {exc}") from exc

    return {
        "text": body.strip(),
        "metadata": {
            "subject": subject,
            "from": from_,
            "to": to,
            "date": date,
        },
    }


def _strip_html(html: str) -> str:
    """Very light HTML tag stripper for email body fallback."""
    import re
    # Remove tags
    text = re.sub(r"<[^>]+>", "", html)
    # Unescape common entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    return text
