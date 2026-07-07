"""logging_config -- per-document JSON-or-text logger configuration.

PLAN.md \u00a79 Phase M: per-doc JSON line logs plus stdout at configured
level. We use stdlib ``logging`` with a tiny JSON formatter (no third-party
``python-json-logger`` dependency) and a per-doc file handler that the
orchestrator attaches at run time.

Configuration is environment-driven (matches .env.example):
    LOG_LEVEL=INFO  (DEBUG | INFO | WARNING | ERROR)
    LOG_FORMAT=json (json | text)

``configure(level, fmt)`` is idempotent -- calling it twice is a no-op
for already-configured root loggers.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

_VALID_LEVELS: Final[frozenset[str]] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
_VALID_FORMATS: Final[frozenset[str]] = frozenset({"json", "text"})

_DEFAULT_LEVEL: Final[str] = "INFO"
_DEFAULT_FORMAT: Final[str] = "json"


# ----------------------------------------------------------------------------
# JSON line formatter (stdlib-only; no python-json-logger dep)
# ----------------------------------------------------------------------------

class _JsonLineFormatter(logging.Formatter):
    """One-line JSON record per logging call.

    We avoid ``python-json-logger`` because (a) it isn't in requirements.txt
    and (b) the spec only needs ``message / level / time / logger``.
    """

    RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        rec = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Carry structured ``extra={"doc_id": ...}``-style fields forward.
        for k, v in record.__dict__.items():
            if k in self.RESERVED or k.startswith("_"):
                continue
            rec[k] = v
        if record.exc_info:
            rec["exc"] = self.formatException(record.exc_info)
        # ensure_ascii=False so non-ASCII (e.g. embedded Spanish) round-trips.
        return json.dumps(rec, ensure_ascii=False)


_TEXT_FMT = "%(asctime)sZ %(levelname)s %(name)s: %(message)s"
_TEXT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def configure(level: str | None = None, fmt: str | None = None) -> logging.Logger:
    """Idempotently configure the root logger from env (or override args).

    Returns the ``uir_pipeline`` logger so call sites can attach
    per-doc handlers via :func:`attach_doc_log`.
    """
    raw_level = (level or os.environ.get("LOG_LEVEL") or _DEFAULT_LEVEL).upper()
    raw_fmt = (fmt or os.environ.get("LOG_FORMAT") or _DEFAULT_FORMAT).lower()
    if raw_level not in _VALID_LEVELS:
        raw_level = _DEFAULT_LEVEL
    if raw_fmt not in _VALID_FORMATS:
        raw_fmt = _DEFAULT_FORMAT

    root = logging.getLogger()
    root.setLevel(raw_level)

    # Remove prior stdout handlers we previously installed so re-configure
    # doesn't accumulate. Third-party handlers (e.g. from dependencies)
    # are left alone.
    for h in list(root.handlers):
        if getattr(h, "_uir_owned", False):
            root.removeHandler(h)

    handler: logging.Handler
    if raw_fmt == "json":
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_JsonLineFormatter())
    else:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_TEXT_FMT, datefmt=_TEXT_DATEFMT))
    handler._uir_owned = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    package_logger = logging.getLogger("uir_pipeline")
    return package_logger


def attach_doc_log(
    doc_id: str,
    log_dir: str | Path,
    package_logger: logging.Logger | None = None,
) -> logging.FileHandler:
    """Attach a per-doc file handler that writes ``log_dir/{doc_id}.log``.

    Returns the handler so the orchestrator can ``handler.close()`` and
    free the FD at end-of-document.
    """
    root = logging.getLogger()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"{doc_id}.log"

    fh = logging.FileHandler(out_path, mode="w", encoding="utf-8")
    fh.setFormatter(_JsonLineFormatter())
    fh.setLevel(root.level or logging.INFO)
    fh._uir_owned = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    if package_logger is not None:
        package_logger.info(
            "logging attached",
            extra={"doc_id": doc_id, "log_path": str(out_path)},
        )
    return fh


def detach_doc_log(handler: logging.FileHandler) -> None:
    """Remove ``handler`` from the root logger and close its stream."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if h is handler:
            root.removeHandler(h)
            handler.close()
            return
    handler.close()


__all__ = [
    "attach_doc_log",
    "configure",
    "detach_doc_log",
]
