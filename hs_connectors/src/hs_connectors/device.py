"""Accelerator/device utilities shared by the hidden-states connectors.

Mooncake's transfer engine ships CUDA and Ascend builds, but the connector and
store code should not hardcode either. The active accelerator is resolved
through ``torch.accelerator`` (torch>=2.6), which reports the device type
(CUDA, XPU, NPU, MUSA, MTIA, ...), and ``torch.get_device_module``, which maps
it to the ``torch`` module exposing the CUDA-like stream API.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import torch


def accelerator_module() -> Any:
    """Return the ``torch`` module backing the active accelerator.

    ``torch.accelerator`` identifies the active device type and
    ``torch.get_device_module`` resolves the module exposing the CUDA-like
    stream API (``Stream``/``Event``/``stream``).
    """
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise RuntimeError("No accelerator is available.")
    return torch.get_device_module(accelerator)


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

    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or not torch.accelerator.is_available():
        # CPU-only process: nothing to initialize (a plain TCP store may still
        # work, so this is not fatal).
        return

    # Re-pin the device inherited across ``fork``. Don't default to 0: that
    # would hijack a worker's selected device and desynchronize the connector's
    # copy stream / ready event from the KV cache.
    with contextlib.suppress(Exception):
        torch.accelerator.set_device_index(torch.accelerator.current_device_index())

    # Touching a device tensor is what actually creates the context.
    with contextlib.suppress(Exception):
        torch.zeros(1, device=accelerator)

    _initialized_pid = pid
