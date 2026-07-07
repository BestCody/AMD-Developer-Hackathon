"""tests/test_weaviate_store.py -- Weaviate connection helper.

Unit tests cover URL parsing, env-var resolution, and the parse_url
helper. A live integration test suite is gated behind a Weaviate-running
check via ``pytest.mark.skipif``, so the suite stays green on machines
without Weaviate.

Run the optional live block::

    docker compose up -d
    .venv/bin/python -m pytest tests/test_weaviate_store.py -v
"""
from __future__ import annotations

import pytest

from uir_pipeline.weaviate_store import (
    DEFAULT_GRPC_PORT,
    DEFAULT_WEAVIATE_URL,
    WEAVIATE_API_KEY_ENV,
    WEAVIATE_URL_ENV,
    get_weaviate_api_key,
    get_weaviate_url,
    parse_url,
    reachable,
)


# ----------------------------------------------------------------------------
# Pure helpers (no I/O)
# ----------------------------------------------------------------------------

class TestParseUrl:
    def test_parses_localhost_with_explicit_port(self):
        assert parse_url("http://localhost:18080") == ("localhost", 18080, False)

    def test_parses_https_default_port(self):
        assert parse_url("https://example.com") == ("example.com", 443, True)

    def test_parses_http_default_port(self):
        assert parse_url("http://h") == ("h", 80, False)

    def test_parses_ip_address(self):
        assert parse_url("http://127.0.0.1:8080") == ("127.0.0.1", 8080, False)

    def test_parses_ipv6(self):
        # urlparse parses the brackets off the hostname.
        host, port, secure = parse_url("http://[::1]:18080")
        assert host == "::1"
        assert port == 18080
        assert secure is False

    def test_explicit_https_port_overrides_default(self):
        assert parse_url("https://example.com:8443") == ("example.com", 8443, True)


class TestEnvResolution:
    def test_default_url_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(WEAVIATE_URL_ENV, raising=False)
        assert get_weaviate_url() == DEFAULT_WEAVIATE_URL

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv(WEAVIATE_URL_ENV, "http://h:9000")
        assert get_weaviate_url() == "http://h:9000"

    def test_api_key_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv(WEAVIATE_API_KEY_ENV, raising=False)
        assert get_weaviate_api_key() is None

    def test_api_key_returns_none_when_blank(self, monkeypatch):
        monkeypatch.setenv(WEAVIATE_API_KEY_ENV, "   ")
        assert get_weaviate_api_key() is None

    def test_api_key_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv(WEAVIATE_API_KEY_ENV, "secret-token-xyz")
        assert get_weaviate_api_key() == "secret-token-xyz"


# ----------------------------------------------------------------------------
# Public constants sanity
# ----------------------------------------------------------------------------

def test_default_url_uses_offset_port():
    """Phase C guardrail: the default URL must NOT be host :8080,
    because that port frequently conflicts on macOS dev machines."""
    assert "18080" in DEFAULT_WEAVIATE_URL
    assert DEFAULT_WEAVIATE_URL.endswith(":18080") or ":18080" in DEFAULT_WEAVIATE_URL


def test_default_grpc_port_is_50051():
    assert DEFAULT_GRPC_PORT == 50051


# ----------------------------------------------------------------------------
# Live integration tests (skipped when Weaviate is offline)
# ----------------------------------------------------------------------------

def _weaviate_reachable() -> bool:
    """Module-level probe used by ``skipif`` decorators at collection time.

    Wraps ``reachable()`` in a try/except so an ImportError of
    ``weaviate`` itself doesn't poison the skipif gating.
    """
    try:
        return bool(reachable())
    except Exception:
        return False


# Evaluate once at module load. Pytest only re-evaluates decorators at
# import time, so this is intentional.
_LIVE: bool = _weaviate_reachable()


_SKIP_LIVE_REASON = "Weaviate not reachable (start with `docker compose up -d`)"


@pytest.mark.skipif(not _LIVE, reason=_SKIP_LIVE_REASON)
class TestWeaviateLive:
    """Smoke tests that exercise the running server. Skipped when offline."""

    def test_is_ready(self):
        from uir_pipeline.weaviate_store import get_client
        client = get_client()
        try:
            assert client.is_ready() is True
        finally:
            client.close()

    def test_collections_list_is_queryable(self):
        from uir_pipeline.weaviate_store import get_client
        client = get_client()
        try:
            # ``collections.list_all`` returns a dict; just ensure it
            # doesn't raise and the value is iterable.
            listing = client.collections.list_all()
            assert listing is not None
        finally:
            client.close()

    def test_collection_create_delete_round_trip(self):
        from uir_pipeline.weaviate_store import get_client
        client = get_client()
        try:
            assert client.is_ready()
            # Name is version-stamped only (not run-stamped). To make
            # the create-assertion actually exercise the create path
            # (vs incidentally passing due to an orphan from a crashed
            # prior run), delete-first so the post-create assertion is
            # meaningful.
            stamp = pytest.__version__.replace(".", "")
            name = f"UIRPipelineSmokeTest_{stamp}"
            existing = client.collections.list_all()
            if name in existing:
                client.collections.delete(name)
            client.collections.create(name=name)
            assert name in client.collections.list_all(), (
                "collection was just created but not visible via list_all"
            )
            client.collections.delete(name)
            assert name not in client.collections.list_all()
        finally:
            client.close()

    def test_reachable_returns_true(self):
        # The module-level _LIVE check passed, so reachable() must agree.
        assert reachable() is True
