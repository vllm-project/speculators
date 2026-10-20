"""Tests for speculators.utils.argmax."""

import pytest
import torch

from speculators.utils.argmax import argmax_last_dim

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Triton argmax kernel runs on CUDA"
)


def _hard_rows(n_cols: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(64, n_cols, device="cuda").to(dtype)
    x[1] = 0.0  # all tied: first index
    x[2] = float("-inf")  # all -inf: index 0
    x[3, [7 % n_cols, n_cols - 1]] = float("inf")  # tied +inf
    # NaN beats everything; first NaN wins
    x[4, [(n_cols - 3) % n_cols, 11 % n_cols]] = float("nan")
    x[5, -1] = float("nan")  # NaN in the masked-tail block
    x[6, [3 % n_cols, n_cols // 2, n_cols - 1]] = 100.0  # ties across blocks
    x[8:] = x[8:].round().clamp(-2, 2)  # dense ties
    return x


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n_cols", [1, 100, 4096, 151936])
def test_matches_torch_argmax(dtype, n_cols):
    x = _hard_rows(n_cols, dtype).view(2, 32, n_cols)
    assert torch.equal(argmax_last_dim(x), torch.argmax(x, dim=-1))


@requires_cuda
def test_non_contiguous_falls_back():
    x = _hard_rows(4096, torch.bfloat16)[:, ::2]
    assert torch.equal(argmax_last_dim(x), torch.argmax(x, dim=-1))


@requires_cuda
def test_compiles_fullgraph():
    x = _hard_rows(4096, torch.bfloat16)
    compiled = torch.compile(argmax_last_dim, fullgraph=True)
    assert torch.equal(compiled(x), torch.argmax(x, dim=-1))


def test_cpu_falls_back():
    x = torch.randn(3, 17)
    assert torch.equal(argmax_last_dim(x), torch.argmax(x, dim=-1))
