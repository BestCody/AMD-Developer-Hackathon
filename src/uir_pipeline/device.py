"""device -- cuda > mps > cpu hardware selector (single source of truth).

PLAN \\u00a79 Phase D + \\u00a74 AMD migration: a single env-line toggle drives
hardware selection; downstream modules (``ocr``, ``layout``, ``embed``, ...)
import :func:`get_device` + :func:`torch_dtype` from here so the
machines-switch is one-line.

Exit criterion (PLAN \\u00a79 Phase D):
    -- select correctly on M-series (returns ``"mps"``)
    -- report ``"cpu"`` on CPU-only Linux
    -- expose :func:`torch_dtype` (``fp16`` on cuda-cuda-rocm, ``fp32`` on
       mps-cpu)
    -- tested with three unit cases

Usage::

    from uir_pipeline.device import get_device, torch_dtype
    dev = get_device()                 # "mps" on Mac, "cuda" on Linux+AMD
    dtype = torch_dtype(dev)           # torch.float16 on cuda, else .float32
    tensor = torch.zeros(2, 2, dtype=dtype, device=dev)

Environment:
    -- ``$DEVICE_PREFERENCE`` (default: ``"cuda,mps,cpu"``) sets the search
       order. Empty / whitespace-only falls back to default. Tokens not
       in ``("cuda","mps","cpu")`` are filtered out.

Testability:
    -- ``import torch`` happens at module load. We patch the module-level
       ``torch`` symbol via pytest monkeypatch to stub it for unit tests.
       The CPU-only (``"cpu"``) path never touches ``torch``.
"""
from __future__ import annotations

import os
from typing import Any, Final, Sequence

_VALID_BACKENDS: Final[tuple[str, ...]] = ("cuda", "mps", "cpu")
_DEFAULT_PREFERENCE: Final[tuple[str, ...]] = ("cuda", "mps", "cpu")
DEVICE_PREFERENCE_ENV: Final[str] = "DEVICE_PREFERENCE"

# Module-level ``import torch`` so downstream callers (and ``torch_dtype``
# return) reach the same singleton. ``None`` sentinel only fires in
# environments where torch isn't installed (rare, e.g. CI minimal).
try:
    import torch as torch  # type: ignore[redef]
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _parse_preference(raw: str | None) -> tuple[str, ...]:
    """Normalize ``$DEVICE_PREFERENCE`` to a validated backend tuple.

    Empty / whitespace / unrecognized tokens fall back to the canonical
    ``("cuda", "mps", "cpu")`` order.
    """
    if not raw:
        return _DEFAULT_PREFERENCE
    tokens = [t.strip().lower() for t in raw.split(",")]
    valid = tuple(t for t in tokens if t in _VALID_BACKENDS)
    return valid if valid else _DEFAULT_PREFERENCE


def default_preference() -> tuple[str, ...]:
    """Read ``$DEVICE_PREFERENCE`` (or return the default fallback)."""
    return _parse_preference(os.environ.get(DEVICE_PREFERENCE_ENV))


def is_available(backend: str) -> bool:
    """Return True iff the requested backend can be used right now.

    Raises :class:`ValueError` for unknown backend names so typos surface
    loudly.

    Backend rules:
        -- ``"cpu"`` is always available (no torch dependency).
        -- ``"cuda"`` requires ``torch`` import + ``torch.cuda.is_available()``.
        -- ``"mps"`` requires ``torch`` import + ``torch.backends.mps.is_available()``.
    """
    if backend == "cpu":
        return True
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown backend: {backend!r}; expect one of {_VALID_BACKENDS}"
        )
    if torch is None:
        return False
    try:
        if backend == "cuda":
            return bool(torch.cuda.is_available())
        if backend == "mps":
            return bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover -- defensive
        return False
    return False  # pragma: no cover -- unreachable


def get_device(preference: Sequence[str] | None = None) -> str:
    """Return the first available backend in the preference chain.

    Default preference comes from ``$DEVICE_PREFERENCE`` (or
    ``("cuda", "mps", "cpu")``). If none of the requested backends is
    available, falls back to ``"cpu"`` which is always available.

    Examples::

        get_device()                       # "mps" on M-series, "cpu" on CI
        get_device(["mps", "cpu"])         # explicit ordering
        get_device(["cpu"])                # never auto-accelerate

    Passing an explicit ``preference`` is the recommended test pattern
    so behavior is deterministic regardless of the host's actual
    hardware.
    """
    chain: tuple[str, ...] = (
        tuple(preference) if preference is not None else default_preference()
    )
    for backend in chain:
        if is_available(backend):
            return backend
    return "cpu"


def torch_dtype(device: str) -> Any:
    """Return the default torch dtype for ``device``.

    Policy (PLAN \\u00a79 Phase D):
        -- ``"cuda"`` -> :class:`torch.float16`: memory-bandwidth bound;
           fp16 fits larger batches on GPUs (NVIDIA + AMD ROCm via CUDA API).
        -- ``"mps"`` -> :class:`torch.float32`: MPS fp16 has stability issues
           on some kernels (Apple Silicon driver maturity).
        -- ``"cpu"`` -> :class:`torch.float32`: no fp16 benefit on CPU.
        -- ``"rocm"`` (alias for ``"cuda"``): same float16 reasoning.

    Raises :class:`RuntimeError` when ``torch`` isn't installed.
    """
    if torch is None:
        raise RuntimeError(
            "torch is not installed; cannot resolve torch_dtype. "
            "Install torch or skip dtype selection."
        )
    if device in ("cuda", "rocm"):
        return torch.float16
    return torch.float32


__all__ = [
    "DEVICE_PREFERENCE_ENV",
    "default_preference",
    "get_device",
    "is_available",
    "torch_dtype",
]
