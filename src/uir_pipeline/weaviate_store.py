"""weaviate_store -- connection helper for the local Weaviate instance.

This module is the Phase C infrastructure layer (PLAN.md \\u00a79 Phase C).
Phase K (``embed.py`` + ``weaviate_store.py``) extends it with upsert and
retrieval helpers for the ``UIRChunks_v1`` and ``UIRParentDoc_v1``
collections.

Phase C exit (per PLAN.md \\u00a79): ``docker compose up -d`` brings Weaviate;
``curl http://localhost:18080/v1/meta`` returns 200; image is
``cr.weaviate.io/semitechnologies/weaviate:1.26.4`` (ARM64 verified).

Network note:
    -- Weaviate's container HTTP port is 8080; the **host** port is 18080
       because dev machines often have another listener on 8080.
    -- ``WEAVIATE_URL=http://localhost:18080`` is the default in ``.env``;
       override the env var to point at a different host/port.
    -- gRPC stays on host :50051 since that port is usually free.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Optional python-dotenv. We deliberately don't hard-depend on it: at
# process start the user can choose to load .env via their own bootstrap
# (the pipeline CLI). `get_weaviate_url()` works without it.
try:
    from dotenv import load_dotenv  # type: ignore
    _HAS_DOTENV = True
except ImportError:  # pragma: no cover
    _HAS_DOTENV = False


# ----------------------------------------------------------------------------
# Public constants
# ----------------------------------------------------------------------------

# Authoritative env var names.
WEAVIATE_URL_ENV = "WEAVIATE_URL"
WEAVIATE_API_KEY_ENV = "WEAVIATE_API_KEY"

# Host-side fallback when WEAVIATE_URL is unset.
# Note: this is the host port mapping in docker-compose.yml (\u00b618080:8080\u00b7).
# We DO NOT match Weaviate's container-side default (8080) because that
# port is host-local on the dev machine and frequently in use.
DEFAULT_WEAVIATE_URL = "http://localhost:18080"

# gRPC port stays at the conventional 50051; rarely conflicts.
DEFAULT_GRPC_PORT = 50051


# ----------------------------------------------------------------------------
# Dotenv handling
# ----------------------------------------------------------------------------

def _maybe_load_dotenv(project_root: Path | None = None) -> None:
    """Load ``.env`` from project root if a ``.env`` file is present.

    Idempotent. ``override=False`` so process env wins over .env.
    """
    if not _HAS_DOTENV:
        return
    candidate = project_root or Path(__file__).resolve().parent.parent.parent
    load_dotenv(candidate / ".env", override=False)


# ----------------------------------------------------------------------------
# Env resolution
# ----------------------------------------------------------------------------

def get_weaviate_url() -> str:
    """Resolve the Weaviate URL from env or fall back to ``DEFAULT_WEAVIATE_URL``."""
    return os.environ.get(WEAVIATE_URL_ENV, DEFAULT_WEAVIATE_URL)


def get_weaviate_api_key() -> str | None:
    """Resolve the Weaviate API key from env.

    Returns ``None`` for the MVP anonymous-dev case. Empty strings and
    whitespace are coerced to ``None`` so users with an unset key (\u00b7\u00b7)
    don't accidentally hit the auth path.
    """
    raw = os.environ.get(WEAVIATE_API_KEY_ENV, "")
    return raw.strip() or None


# ----------------------------------------------------------------------------
# URL parsing
# ----------------------------------------------------------------------------

def parse_url(url: str) -> tuple[str, int, bool]:
    """Parse ``scheme://host[:port][/path]`` into ``(host, port, secure)``.

    Defaults: ``port=443`` for https, ``port=80`` for http when absent.
    ``secure=True`` for https and wss; ``False`` otherwise.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "http").lower()
    host = parsed.hostname or "localhost"
    if parsed.port is not None:
        port = parsed.port
    elif scheme in {"https", "wss"}:
        port = 443
    else:
        port = 80
    secure = scheme in {"https", "wss"}
    return host, port, secure


# ----------------------------------------------------------------------------
# Client construction
# ----------------------------------------------------------------------------

def get_client(
    url: str | None = None,
    api_key: str | None = None,
    grpc_port: int = DEFAULT_GRPC_PORT,
) -> Any:
    """Return an open ``weaviate.WeaviateClient``.

    Lazy-imports ``weaviate`` so this module stays importable in CI
    minimal envs where the client is not installed.

    Caller is responsible for ``client.close()`` (typically via a context
    manager that the caller wires up). The client is NOT thread-safe.

    Auth:
        -- Anonymous dev (MVP): passes ``auth_credentials=None`` to
           ``connect_to_local``. Weaviate needs
           ``AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true`` in its env.
        -- API-key mode (prod): wraps the key in ``AuthApiKey(...)``.
    """
    import weaviate  # type: ignore

    _maybe_load_dotenv()
    resolved_url = url or get_weaviate_url()
    resolved_key = api_key if api_key is not None else get_weaviate_api_key()
    host, http_port, secure = parse_url(resolved_url)

    auth_credentials = (
        weaviate.auth.AuthApiKey(resolved_key) if resolved_key else None
    )

    # ``connect_to_local`` handles localhost / arbitrary host:port + gRPC
    # wiring in one call. It does NOT accept custom HTTPS paths,
    # certificates, or proxy headers -- production deployments that need
    # those should switch to ``weaviate.connect_to_custom`` (Phase 2).
    return weaviate.connect_to_local(
        host=host,
        port=http_port,
        grpc_port=grpc_port,
        auth_credentials=auth_credentials,
    )


# ----------------------------------------------------------------------------
# Reachability warm-ping
# ----------------------------------------------------------------------------

def reachable(client: Any | None = None, timeout: float = 3.0) -> bool:
    """Return ``True`` iff the Weaviate server responds to ``is_ready()``.

    Single-shot helper: if ``client`` is None, opens + closes a fresh
    client. If ``client`` is supplied, the caller owns its lifecycle and
    we never close it. Errors are swallowed (ImportError, network, auth)
    and converted to ``False`` so callers can use this as a guard. The
    original exception is logged at DEBUG so misconfig remains
    diagnosable.

    Implementation notes:
        -- ``c`` is pre-bound to ``None`` so the ``get_client()``
           assignment lives inside the ``try`` block. Otherwise an
           exception during client creation would raise before ``try``
           is entered, then crash in ``finally`` with a NameError on
           ``c`` instead of the genuine cause.
        -- ``should_close`` captures the lifecycle contract at entry:
           we own this client IFF we opened it (caller-passed clients
           are off-limits).
    """
    should_close = client is None
    c = None
    try:
        c = client if client is not None else get_client()
        return bool(c.is_ready())
    except Exception as exc:
        logging.debug("uir_pipeline.weaviate_store.reachable() failed: %s", exc)
        return False
    finally:
        if should_close and c is not None:
            try:
                c.close()
            except Exception:
                pass


__all__ = [
    "DEFAULT_GRPC_PORT",
    "DEFAULT_WEAVIATE_URL",
    "WEAVIATE_API_KEY_ENV",
    "WEAVIATE_URL_ENV",
    "get_client",
    "get_weaviate_api_key",
    "get_weaviate_url",
    "parse_url",
    "reachable",
]
