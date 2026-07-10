"""tests/test_weaviate_failsoft.py -- which upsert failures are fatal.

``pipeline.run`` fails soft when Weaviate simply isn't running (the CLI
defaults to ``skip_weaviate=False``, so a dev without ``docker compose up``
must still get their UIR JSON). It must NOT fail soft on a bug in our own
code -- that is how a schema mismatch shipped while every run logged
"done ... chunks=267" and stored nothing.

No server required.
"""
from __future__ import annotations

import pytest

from uir_pipeline.pipeline import _is_weaviate_unavailable

weaviate_exceptions = pytest.importorskip("weaviate.exceptions")


class TestTransportFailuresAreSoft:
    @pytest.mark.parametrize(
        "exc_name",
        [
            "WeaviateConnectionError",
            "WeaviateGRPCUnavailableError",
            "WeaviateStartUpError",
            "WeaviateTimeoutError",
        ],
    )
    def test_transport_errors_are_unavailable(self, exc_name):
        exc_cls = getattr(weaviate_exceptions, exc_name)
        assert _is_weaviate_unavailable(exc_cls("boom")) is True

    def test_missing_client_library_is_unavailable(self):
        assert _is_weaviate_unavailable(ImportError("no module named weaviate")) is True


class TestCodeDefectsAreFatal:
    @pytest.mark.parametrize(
        "exc",
        [
            AttributeError("'dict' object has no attribute 'textAnalyzer'"),
            TypeError("bad argument"),
            KeyError("vector"),
            RuntimeError("weaviate rejected 3/3 chunk objects"),
            ValueError("Not valid 'uuid'"),
        ],
    )
    def test_programming_errors_are_not_unavailable(self, exc):
        assert _is_weaviate_unavailable(exc) is False

    def test_generic_weaviate_error_is_not_unavailable(self):
        """A schema/validation error from the server is our bug, not absence."""
        exc = weaviate_exceptions.WeaviateInvalidInputError("bad property")
        assert _is_weaviate_unavailable(exc) is False

    def test_the_exact_historical_bug_is_fatal(self):
        """The real regression: raw-dict properties -> AttributeError.

        Had this been classified 'unavailable', the fix would have been
        swallowed right back into a warning.
        """
        exc = AttributeError("'dict' object has no attribute 'textAnalyzer'")
        assert _is_weaviate_unavailable(exc) is False
