"""tests/test_device.py -- hardware selector (Phase D).

Unit tests cover the public API of ``uir_pipeline.device``. We mock the
module-level ``torch`` symbol via ``monkeypatch`` to deterministically
test each branch without depending on the actual host. Tests exit fast
and never require real CUDA / MPS hardware.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from uir_pipeline import device as d


# ----------------------------------------------------------------------------
# Stubbed-torch fixtures
# ----------------------------------------------------------------------------

class _StubTorch:
    """Build a fake torch module that reports CUDA/MPS as the flags dictate.

    ``float16`` and ``float32`` are forwarded from the real torch module
    so ``d.torch_dtype()`` (which compares against these singletons) can
    be exercised against the stub. ``cuda.is_available()`` and
    ``backends.mps.is_available()`` are mocked per constructor flags.
    """
    def __init__(self, *, cuda: bool, mps: bool):
        import torch as real_torch  # only at construction time
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        )
        self.float16 = real_torch.float16
        self.float32 = real_torch.float32


@pytest.fixture
def stub_cuda_and_mps(monkeypatch):
    monkeypatch.setattr(d, "torch", _StubTorch(cuda=True, mps=True))


@pytest.fixture
def stub_mps_only(monkeypatch):
    monkeypatch.setattr(d, "torch", _StubTorch(cuda=False, mps=True))


@pytest.fixture
def stub_neither_accel(monkeypatch):
    monkeypatch.setattr(d, "torch", _StubTorch(cuda=False, mps=False))


@pytest.fixture
def stub_no_torch(monkeypatch):
    """Pretend torch isn't installed (simulates CPU-only CI)."""
    monkeypatch.setattr(d, "torch", None)


# ----------------------------------------------------------------------------
# get_device()
# ----------------------------------------------------------------------------

def test_get_device_returns_first_available_cuda(stub_cuda_and_mps):
    assert d.get_device(["cuda", "mps", "cpu"]) == "cuda"


def test_get_device_skips_unavailable_cuda(stub_mps_only):
    assert d.get_device(["cuda", "mps", "cpu"]) == "mps"


def test_get_device_returns_cpu_when_all_gpus_missing(stub_neither_accel):
    assert d.get_device(["cuda", "mps", "cpu"]) == "cpu"


def test_get_device_handles_absent_torch(stub_no_torch):
    assert d.get_device(["cuda", "mps", "cpu"]) == "cpu"


def test_get_device_honors_custom_preference_order(stub_mps_only):
    """Custom preference overrides default; cpu first -> cpu even with MPS."""
    assert d.get_device(["cpu", "cuda", "mps"]) == "cpu"


def test_get_device_empty_preference_uses_default(stub_mps_only):
    """None preference -> default chain -> ``mps`` when only MPS is up."""
    assert d.get_device() == "mps"


def test_get_device_always_returns_cuda_or_mps_or_cpu(stub_cuda_and_mps):
    val = d.get_device(["cuda", "mps", "cpu"])
    assert val in ("cuda", "mps", "cpu")


# ----------------------------------------------------------------------------
# is_available()
# ----------------------------------------------------------------------------

def test_is_available_cpu_always_true():
    """CPU doesn't require torch."""
    assert d.is_available("cpu") is True


def test_is_available_cpu_true_even_without_torch(stub_no_torch):
    assert d.is_available("cpu") is True


def test_is_available_cuda_when_torch_says_yes(stub_cuda_and_mps):
    assert d.is_available("cuda") is True


def test_is_available_cuda_when_torch_says_no(stub_mps_only):
    assert d.is_available("cuda") is False


def test_is_available_mps_when_torch_says_yes(stub_mps_only):
    assert d.is_available("mps") is True


def test_is_available_mps_when_torch_says_no(stub_neither_accel):
    assert d.is_available("mps") is False


def test_is_available_returns_false_if_torch_missing(stub_no_torch):
    assert d.is_available("cuda") is False
    assert d.is_available("mps") is False


def test_is_available_rejects_unknown_backend():
    with pytest.raises(ValueError) as excinfo:
        d.is_available("tpu")
    assert "tpu" in str(excinfo.value)


# ----------------------------------------------------------------------------
# default_preference()
# ----------------------------------------------------------------------------

def test_default_preference_parses_canonical_env(monkeypatch):
    monkeypatch.setenv(d.DEVICE_PREFERENCE_ENV, "cuda,mps,cpu")
    assert d.default_preference() == ("cuda", "mps", "cpu")


def test_default_preference_accepts_reordering(monkeypatch):
    monkeypatch.setenv(d.DEVICE_PREFERENCE_ENV, "mps,cpu,cuda")
    assert d.default_preference() == ("mps", "cpu", "cuda")


def test_default_preference_filters_out_unknown_tokens(monkeypatch):
    monkeypatch.setenv(d.DEVICE_PREFERENCE_ENV, "cuda,tpu,mps")
    assert d.default_preference() == ("cuda", "mps")


def test_default_preference_falls_back_when_blank(monkeypatch):
    monkeypatch.setenv(d.DEVICE_PREFERENCE_ENV, "")
    assert d.default_preference() == ("cuda", "mps", "cpu")


def test_default_preference_falls_back_when_only_commas(monkeypatch):
    monkeypatch.setenv(d.DEVICE_PREFERENCE_ENV, ",,,")
    assert d.default_preference() == ("cuda", "mps", "cpu")


def test_default_preference_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv(d.DEVICE_PREFERENCE_ENV, raising=False)
    assert d.default_preference() == ("cuda", "mps", "cpu")


# ----------------------------------------------------------------------------
# torch_dtype()
# ----------------------------------------------------------------------------

def test_torch_dtype_cuda_is_float16(stub_cuda_and_mps):
    import torch as real_torch
    assert d.torch_dtype("cuda") is real_torch.float16


def test_torch_dtype_mps_is_float32(stub_mps_only):
    import torch as real_torch
    assert d.torch_dtype("mps") is real_torch.float32


def test_torch_dtype_cpu_is_float32(stub_neither_accel):
    import torch as real_torch
    assert d.torch_dtype("cpu") is real_torch.float32


def test_torch_dtype_rocm_alias_is_float16(stub_cuda_and_mps):
    """``"rocm"`` is an alias for ``"cuda"`` -- AMD ROCm uses CUDA API."""
    import torch as real_torch
    assert d.torch_dtype("rocm") is real_torch.float16


def test_torch_dtype_raises_when_no_torch(stub_no_torch):
    with pytest.raises(RuntimeError):
        d.torch_dtype("mps")
