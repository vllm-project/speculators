"""Flex attention kernel_options selection for low shared-memory GPUs."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from speculators.models import attention

CUDA = torch.device("cuda")


def _set_shared_memory(monkeypatch, nbytes):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(shared_memory_per_block_optin=nbytes),
    )


def test_low_shared_memory_uses_small_backward_tiles(monkeypatch):
    _set_shared_memory(monkeypatch, 101376)
    assert (
        attention.flex_kernel_options(CUDA, head_dim=256)
        == attention._LOW_SHARED_MEMORY_KERNEL_OPTIONS
    )


def test_large_shared_memory_keeps_defaults(monkeypatch):
    _set_shared_memory(monkeypatch, 232448)
    assert attention.flex_kernel_options(CUDA, head_dim=256) is None


def test_small_head_dim_keeps_defaults(monkeypatch):
    _set_shared_memory(monkeypatch, 101376)
    assert attention.flex_kernel_options(CUDA, head_dim=128) is None


def test_cpu_keeps_defaults():
    assert attention.flex_kernel_options(torch.device("cpu"), head_dim=256) is None


def test_env_override_wins(monkeypatch):
    monkeypatch.setattr(attention, "_ENV_KERNEL_OPTIONS", {"bwd_BLOCK_M1": 16})
    assert attention.flex_kernel_options(CUDA, head_dim=256) == {"bwd_BLOCK_M1": 16}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_compiled_backward_head_dim_256():
    seq_len = 1024
    kw: dict[str, Any] = {
        "device": "cuda",
        "dtype": torch.bfloat16,
        "requires_grad": True,
    }
    q = torch.randn(1, 16, seq_len, 256, **kw)
    k = torch.randn(1, 4, seq_len, 256, **kw)
    v = torch.randn(1, 4, seq_len, 256, **kw)
    mask = create_block_mask(
        lambda _b, _h, q_idx, kv_idx: q_idx >= kv_idx,
        None,
        None,
        seq_len,
        seq_len,
        device="cuda",
    )
    out, _ = torch.compile(attention.flex_attention_forward)(
        torch.nn.Identity(), q, k, v, mask
    )
    out.sum().backward()
    assert all(t.grad is not None for t in (q, k, v))
