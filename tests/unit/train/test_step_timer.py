from unittest.mock import patch

import pytest

from speculators.train.trainer import _StepTimer


def test_disabled_timer_returns_none():
    timer = _StepTimer(enabled=False)
    timer.mark_value("start", 0.0)
    timer.mark("queue")
    timer.mark("pre_h2d")
    timer.mark("fetch")
    timer.mark("fwd")
    timer.mark("pre_clip")
    timer.mark("bwd")
    timer.mark("opt")
    assert timer.now() is None
    assert timer.profile(num_tokens=1024) is None


@patch("speculators.train.trainer.torch.accelerator.synchronize")
def test_enabled_timer_returns_profile(mock_sync):
    timer = _StepTimer(enabled=True)

    with patch(
        "speculators.train.trainer.time.perf_counter",
        side_effect=[
            0.05,
            0.08,
            0.1,
            0.3,
            0.45,
            0.5,
            0.6,
            0.6,
        ],
    ):
        timer.mark_value("start", 0.0)
        timer.mark("queue")
        timer.mark("pre_h2d")
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("pre_clip")
        timer.mark("bwd")
        timer.mark("opt")
        t_next = timer.now()

    assert mock_sync.call_count == 8
    assert t_next == 0.6

    profile = timer.profile(num_tokens=4096)
    assert profile is not None
    assert profile["queue_ms"] == (0.05 - 0.0) * 1000
    assert profile["h2d_ms"] == (0.1 - 0.08) * 1000
    assert profile["fetch_ms"] == (0.1 - 0.0) * 1000
    assert profile["fwd_ms"] == (0.3 - 0.1) * 1000
    assert profile["clip_ms"] == (0.5 - 0.45) * 1000
    assert profile["bwd_ms"] == (0.5 - 0.3) * 1000
    assert profile["opt_ms"] == (0.6 - 0.5) * 1000
    assert profile["step_ms"] == (0.6 - 0.0) * 1000
    assert profile["tokens_per_s"] == 4096 / 0.6
    assert profile["fetch_frac"] == 100 / 600


def test_disabled_to_enabled_transition():
    """Simulate log_freq=2: a disabled step followed by an enabled step.

    The disabled step's ``timer.now()`` returns None so the training loop
    falls back to ``time.perf_counter()`` for ``t_before_fetch``.  The
    subsequent enabled step feeds that value into ``mark_value("start", ...)``.
    Verify the profile is valid with a realistic ``start`` mark.
    """
    timer = _StepTimer()

    # --- disabled step (global_step=1, log_freq=2 → 1%2 != 0) ---
    timer.reset(enabled=False)
    timer.mark_value("start", 1.0)
    timer.mark("queue")
    timer.mark("pre_h2d")
    timer.mark("fetch")
    timer.mark("fwd")
    timer.mark("pre_clip")
    timer.mark("bwd")
    timer.mark("opt")
    assert timer.now() is None
    assert timer.profile(num_tokens=512) is None

    # Simulate the fallback: t_before_fetch = timer.now() or time.perf_counter()
    t_before_fetch = 2.0  # stands in for the perf_counter() fallback

    # --- enabled step (global_step=2, log_freq=2 → 2%2 == 0) ---
    timer.reset(enabled=True)
    timer.mark_value("start", t_before_fetch)

    with (
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch(
            "speculators.train.trainer.time.perf_counter",
            side_effect=[2.05, 2.08, 2.1, 2.4, 2.45, 2.5, 2.6, 2.6],
        ),
    ):
        timer.mark("queue")
        timer.mark("pre_h2d")
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("pre_clip")
        timer.mark("bwd")
        timer.mark("opt")
        t_next = timer.now()

    assert t_next == 2.6
    profile = timer.profile(num_tokens=2048)
    assert profile is not None
    assert profile["queue_ms"] == (2.05 - 2.0) * 1000
    assert profile["h2d_ms"] == (2.1 - 2.08) * 1000
    assert profile["fetch_ms"] == (2.1 - 2.0) * 1000
    assert profile["fwd_ms"] == (2.4 - 2.1) * 1000
    assert profile["clip_ms"] == (2.5 - 2.45) * 1000
    assert profile["bwd_ms"] == (2.5 - 2.4) * 1000
    assert profile["opt_ms"] == (2.6 - 2.5) * 1000
    assert profile["step_ms"] == (2.6 - 2.0) * 1000
    assert profile["tokens_per_s"] == pytest.approx(2048 / 0.6)


def test_zero_step_ms_returns_zero_throughput():
    timer = _StepTimer(enabled=True)
    timer.mark_value("start", 1.0)
    with (
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch("speculators.train.trainer.time.perf_counter", return_value=1.0),
    ):
        timer.mark("queue")
        timer.mark("pre_h2d")
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("pre_clip")
        timer.mark("bwd")
        timer.mark("opt")
    profile = timer.profile(num_tokens=4096)
    assert profile is not None
    assert profile["tokens_per_s"] == 0.0
    assert profile["fetch_frac"] == 0.0


class _FakeDevice:
    def __init__(self, device_type: str):
        self.type = device_type


class _StubBackend:
    """Minimal stub backend module exposing ``memory_allocated``."""

    def __init__(self, mem_bytes: int) -> None:
        self._mem_bytes = mem_bytes

    def memory_allocated(self) -> int:
        return self._mem_bytes


class _EmptyBackend:
    """Stub backend with no ``memory_allocated`` attribute."""


def test_memory_fallback_uses_backend_native_api():
    """Generic accelerator API raises RuntimeError; falls back to backend-native.

    This is a backend-agnostic fallback: when torch.accelerator.memory_allocated()
    fails on any device type, the code tries torch.<device_type>.memory_allocated()
    instead.  We simulate this on a CPU-only CI by patching the accelerator
    APIs and injecting a stub backend module.
    """
    stub_backend = _StubBackend(mem_bytes=42 * 1024**2)  # 42 MiB

    fake_device = _FakeDevice("fake_device")

    with (
        patch(
            "speculators.train.trainer.torch.accelerator.is_available",
            return_value=True,
        ),
        patch(
            "speculators.train.trainer.torch.accelerator.current_accelerator",
            return_value=fake_device,
        ),
        patch(
            "speculators.train.trainer.torch.accelerator.memory_allocated",
            side_effect=RuntimeError("Allocator is not a DeviceAllocator"),
        ),
        patch(
            "speculators.train.trainer.torch.fake_device",
            stub_backend,
            create=True,
        ),
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch(
            "speculators.train.trainer.time.perf_counter",
            return_value=1.0,
        ),
    ):
        timer = _StepTimer(enabled=True)
        timer.mark("fwd")

    assert timer._memory["fwd"] == 42.0


def test_memory_fallback_skips_when_backend_missing():
    """Generic API raises RuntimeError and no backend module exists -> skip."""
    fake_device = _FakeDevice("privateuse1")

    with (
        patch(
            "speculators.train.trainer.torch.accelerator.is_available",
            return_value=True,
        ),
        patch(
            "speculators.train.trainer.torch.accelerator.current_accelerator",
            return_value=fake_device,
        ),
        patch(
            "speculators.train.trainer.torch.accelerator.memory_allocated",
            side_effect=RuntimeError("not a DeviceAllocator"),
        ),
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch(
            "speculators.train.trainer.time.perf_counter",
            return_value=1.0,
        ),
    ):
        timer = _StepTimer(enabled=True)
        timer.mark("fwd")

    assert "fwd" not in timer._memory


def test_memory_fallback_skips_when_backend_lacks_memory_allocated():
    """Backend module exists but has no memory_allocated -> skip silently."""
    # _EmptyBackend has no memory_allocated attribute, so hasattr returns False
    # and the trainer skips memory recording for this step.
    stub_backend = _EmptyBackend()

    fake_device = _FakeDevice("fake_device")

    with (
        patch(
            "speculators.train.trainer.torch.accelerator.is_available",
            return_value=True,
        ),
        patch(
            "speculators.train.trainer.torch.accelerator.current_accelerator",
            return_value=fake_device,
        ),
        patch(
            "speculators.train.trainer.torch.accelerator.memory_allocated",
            side_effect=RuntimeError("not a DeviceAllocator"),
        ),
        patch(
            "speculators.train.trainer.torch.fake_device",
            stub_backend,
            create=True,
        ),
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch(
            "speculators.train.trainer.time.perf_counter",
            return_value=1.0,
        ),
    ):
        timer = _StepTimer(enabled=True)
        timer.mark("fwd")

    assert "fwd" not in timer._memory


def test_memory_happy_path_still_works():
    """Generic accelerator API succeeds (CUDA-like) - memory recorded normally."""
    fake_device = _FakeDevice("cuda")

    with (
        patch(
            "speculators.train.trainer.torch.accelerator.is_available",
            return_value=True,
        ),
        patch(
            "speculators.train.trainer.torch.accelerator.current_accelerator",
            return_value=fake_device,
        ),
        patch(
            "speculators.train.trainer.torch.accelerator.memory_allocated",
            return_value=100 * 1024**2,
        ),
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch(
            "speculators.train.trainer.time.perf_counter",
            return_value=1.0,
        ),
    ):
        timer = _StepTimer(enabled=True)
        timer.mark("fwd")

    assert timer._memory["fwd"] == 100.0
