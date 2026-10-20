import pytest
import torch

from speculators.models.argmax import argmax_last_dim

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_cpu_falls_back_to_torch():
    x = torch.randn(3, 5, 17)
    assert torch.equal(argmax_last_dim(x), torch.argmax(x, dim=-1))


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("n_cols", [1, 7, 1000, 32000, 151936])
def test_matches_torch_argmax(dtype, n_cols):
    torch.manual_seed(0)
    x = torch.randn(1, 33, n_cols, device="cuda").to(dtype)
    assert torch.equal(argmax_last_dim(x), torch.argmax(x, dim=-1))


@requires_cuda
def test_ties_nan_and_inf_follow_torch():
    torch.manual_seed(0)
    # Coarse quantization makes ties across and within tiles common.
    x = (torch.randn(64, 20000, device="cuda") * 2).round().to(torch.bfloat16)
    x[0] = 1.0
    x[1] = float("-inf")
    x[2, 100] = float("nan")
    x[3, 10] = float("inf")
    x[3, 900] = float("nan")
    x[4, 5] = float("inf")
    x[4, 19999] = float("inf")
    assert torch.equal(argmax_last_dim(x), torch.argmax(x, dim=-1))


@requires_cuda
def test_non_contiguous_input():
    x = torch.randn(40, 64, device="cuda").t()
    assert torch.equal(argmax_last_dim(x), torch.argmax(x, dim=-1))
