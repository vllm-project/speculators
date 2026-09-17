"""Accelerator/device utilities shared by the hidden-states connectors.

Mooncake's transfer engine ships CUDA and Ascend builds, but the connector and
store code should not hardcode either. Resolve the active accelerator (CUDA,
XPU, NPU, MUSA, MTIA, ...) via ``torch.accelerator`` when available, falling
back to probing the known ``torch.*`` device modules.
"""

from __future__ import annotations

import contextlib
import os
from functools import lru_cache
from typing import Any

import torch

# Device modules that expose a CUDA-like stream API. Order matters only for the
# fallback probe; ``torch.accelerator`` is authoritative when it is available.
_DEVICE_MODULES = ("cuda", "xpu", "npu", "musa", "mtia")
_STREAM_ATTRS = ("Stream", "Event", "stream")


def _is_usable(module: Any, *, need_stream: bool) -> bool:
    if module is None:
        return False
    try:
        if not module.is_available():
            return False
    except Exception:  # noqa: BLE001 - probing untrusted accelerator modules
        return False
    if need_stream:
        return all(hasattr(module, attr) for attr in _STREAM_ATTRS)
    return True


def _module_for_type(name: str) -> Any:
    module = getattr(torch, name, None)
    if module is None and name == "npu":
        try:
            import torch_npu  # noqa: F401, PLC0415
        except Exception:  # noqa: BLE001 - optional accelerator package
            return None
        module = getattr(torch, "npu", None)
    return module


def _device_type(module: Any) -> str | None:
    for name in _DEVICE_MODULES:
        if module is getattr(torch, name, None):
            return name
    return None


@lru_cache(maxsize=2)
def accelerator_module(*, need_stream: bool = True) -> Any:
    """Return the active accelerator module.

    Prefers ``torch.accelerator`` (torch>=2.6) to detect the active device type
    and probes the known ``torch.*`` device modules otherwise. With
    ``need_stream=True`` the module must expose ``Stream``/``Event``/``stream``
    (Apple MPS does not, so it is only usable for context setup).
    """
    acc = getattr(torch, "accelerator", None)
    if acc is not None and getattr(acc, "is_available", lambda: False)():
        try:
            name: str | None = acc.current_accelerator().type
        except Exception:  # noqa: BLE001 - fall back to the module probe
            name = None
        if name:
            module = _module_for_type(str(name))
            if _is_usable(module, need_stream=need_stream):
                return module

    for name in _DEVICE_MODULES:
        module = _module_for_type(name)
        if _is_usable(module, need_stream=need_stream):
            return module

    raise RuntimeError(
        "No usable accelerator found (looked for CUDA/XPU/NPU/MUSA/MTIA)."
    )


_initialized_pid: int | None = None


def ensure_accelerator_context() -> None:
    """Initialize the accelerator context once per process.

    Some backends (notably Mooncake's Ascend transport) cannot allocate their
    local segment without an active device context, which dataloader/storage
    worker processes lack after a ``fork``. The PID guard re-runs the setup in
    forked children so each worker gets its own context.
    """
    global _initialized_pid  # noqa: PLW0603 - process-local init guard
    pid = os.getpid()
    if _initialized_pid == pid:
        return

    try:
        module = accelerator_module(need_stream=False)
    except RuntimeError:
        # CPU-only process: nothing to initialize (a plain TCP store may still
        # work, so this is not fatal).
        return

    device = 0
    acc = getattr(torch, "accelerator", None)
    if acc is not None and getattr(acc, "is_available", lambda: False)():
        with contextlib.suppress(Exception):
            device = acc.current_device_index()

    if hasattr(module, "set_device"):
        with contextlib.suppress(Exception):
            module.set_device(device)

    # Touching a device tensor is what actually creates the context.
    dev_type = _device_type(module)
    if dev_type is not None:
        with contextlib.suppress(Exception):
            torch.zeros(1, device=dev_type)

    _initialized_pid = pid
