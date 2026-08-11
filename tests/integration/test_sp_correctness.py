# ruff: noqa: T201, PLC0415
"""Verify Ulysses sequence parallelism produces identical results to non-SP.

Run with:
    torchrun --standalone --nproc_per_node=2 tests/integration/test_sp_correctness.py

Tests:
  1. All-to-all round-trip: scatter then gather recovers the original tensor.
  2. Attention equivalence: SP attention output matches direct full attention.
  3. Gradient equivalence: gradients through SP match gradients without SP.
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def _causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _a2a(x, scatter_dim, gather_dim):
    import speculators.train.sequence_parallel  # noqa: F401

    return torch.ops.speculators.all_to_all_sp(x, scatter_dim, gather_dim)


def test_round_trip(rank, device):
    """Scatter then gather should recover the original tensor."""
    torch.manual_seed(42)
    x = torch.randn(1, 8, 256, 64, device=device, dtype=torch.bfloat16)

    scattered = _a2a(x, 1, 2)
    recovered = _a2a(scattered, 2, 1)

    max_diff = (x - recovered).abs().max().item()
    if rank == 0:
        status = "PASS" if max_diff < 1e-6 else "FAIL"
        print(f"[Round-trip] max diff = {max_diff:.2e}  {status}")
    return max_diff < 1e-6


def test_attention_equivalence(sp_size, rank, device):
    """SP scatter->attention->gather should match direct attention."""
    from speculators.train.sequence_parallel import (
        maybe_replicate_kv_heads,
    )

    torch.manual_seed(42)
    B, H, S_full, D = 1, 8, 256, 64
    S_local = S_full // sp_size

    q_full = torch.randn(B, H, S_full, D, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(B, H, S_full, D, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(B, H, S_full, D, device=device, dtype=torch.bfloat16)

    full_mask = create_block_mask(
        _causal, B=None, H=None, Q_LEN=S_full, KV_LEN=S_full, device=device
    )
    ref = flex_attention(q_full, k_full, v_full, block_mask=full_mask)

    q_local = q_full[:, :, rank * S_local : (rank + 1) * S_local, :]
    k_local = k_full[:, :, rank * S_local : (rank + 1) * S_local, :]
    v_local = v_full[:, :, rank * S_local : (rank + 1) * S_local, :]

    k_rep, v_rep = maybe_replicate_kv_heads(k_local, v_local, sp_size)
    q_sp = _a2a(q_local, 1, 2)
    k_sp = _a2a(k_rep, 1, 2)
    v_sp = _a2a(v_rep, 1, 2)

    sp_out = flex_attention(
        q_sp.contiguous(),
        k_sp.contiguous(),
        v_sp.contiguous(),
        block_mask=full_mask,
        enable_gqa=q_sp.shape[1] != k_sp.shape[1],
    )
    sp_out = _a2a(sp_out, 2, 1)

    ref_local = ref[:, :, rank * S_local : (rank + 1) * S_local, :]

    max_diff = (ref_local - sp_out).abs().max().item()
    mean_diff = (ref_local - sp_out).abs().mean().item()
    if rank == 0:
        status = "PASS" if max_diff < 5e-2 else "FAIL"
        print(
            f"[Attention]  max diff = {max_diff:.2e}, "
            f"mean diff = {mean_diff:.2e}  {status}"
        )
    return max_diff < 5e-2


def test_gradient_equivalence(sp_size, rank, device):
    """Gradients through SP path should match gradients without SP."""
    from speculators.train.sequence_parallel import (
        maybe_replicate_kv_heads,
    )

    torch.manual_seed(42)
    B, H, S_full, D = 1, 8, 256, 64
    S_local = S_full // sp_size

    q_full = torch.randn(B, H, S_full, D, device=device, dtype=torch.float32)
    k_full = torch.randn(B, H, S_full, D, device=device, dtype=torch.float32)
    v_full = torch.randn(B, H, S_full, D, device=device, dtype=torch.float32)

    full_mask = create_block_mask(
        _causal, B=None, H=None, Q_LEN=S_full, KV_LEN=S_full, device=device
    )

    q_ref = q_full.clone().requires_grad_(True)
    k_ref = k_full.clone().requires_grad_(True)
    v_ref = v_full.clone().requires_grad_(True)
    ref_out = flex_attention(q_ref, k_ref, v_ref, block_mask=full_mask)
    ref_out.sum().backward()

    sl = slice(rank * S_local, (rank + 1) * S_local)
    q_local = q_full[:, :, sl, :].clone().requires_grad_(True)
    k_local = k_full[:, :, sl, :].clone().requires_grad_(True)
    v_local = v_full[:, :, sl, :].clone().requires_grad_(True)

    k_rep, v_rep = maybe_replicate_kv_heads(k_local, v_local, sp_size)
    q2 = _a2a(q_local, 1, 2)
    k2 = _a2a(k_rep, 1, 2)
    v2 = _a2a(v_rep, 1, 2)

    sp_out = flex_attention(
        q2.contiguous(),
        k2.contiguous(),
        v2.contiguous(),
        block_mask=full_mask,
        enable_gqa=q2.shape[1] != k2.shape[1],
    )
    sp_out = _a2a(sp_out, 2, 1)
    sp_out.sum().backward()

    assert q_ref.grad is not None
    assert q_local.grad is not None
    q_ref_local = q_ref.grad[:, :, sl, :]
    q_max = (q_ref_local - q_local.grad).abs().max().item()
    if rank == 0:
        status = "PASS" if q_max < 1e-4 else "FAIL"
        print(f"[Gradients]  Q grad max diff = {q_max:.2e}  {status}")
    return q_max < 1e-4


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from speculators.train.distributed import (
        _init_sp_process_groups,
    )

    _init_sp_process_groups(rank, world_size, sp_size=world_size)

    if rank == 0:
        print(f"Testing SP correctness with {world_size} GPUs\n")

    results = [
        test_round_trip(rank, device),
        test_attention_equivalence(world_size, rank, device),
        test_gradient_equivalence(world_size, rank, device),
    ]

    if rank == 0:
        print()
        if all(results):
            print("All tests passed.")
        else:
            print("Some tests FAILED.")
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
