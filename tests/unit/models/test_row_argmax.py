"""row_argmax must match torch.argmax(x, dim=-1) bit for bit."""

import pytest
import torch

from speculators.models.argmax import row_argmax

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton path")


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [1000, 32000, 151936])
def test_matches_torch_random(dtype, vocab):
    x = torch.randn(1, 257, vocab, device="cuda", dtype=dtype)
    assert torch.equal(row_argmax(x), torch.argmax(x, dim=-1))


@cuda
def test_matches_torch_ties_nan_inf():
    vocab = 151936
    x = torch.randint(-3, 3, (16, vocab), device="cuda").to(torch.bfloat16)
    x[0] = float("-inf")  # all -inf -> index 0
    x[1] = 0.0  # all tied -> first index
    x[2, 100] = float("nan")  # first NaN wins
    x[2, 50000] = float("nan")
    x[3, 9] = float("inf")  # first +inf wins
    x[3, 9000] = float("inf")
    x[4, 140000] = float("nan")  # NaN beats an earlier +inf
    x[4, 3] = float("inf")
    x[5, vocab - 1] = 100.0  # max in the ragged last tile
    assert torch.equal(row_argmax(x), torch.argmax(x, dim=-1))


@cuda
def test_row_sliced_view_with_grad():
    x = torch.randn(1, 64, 4096, device="cuda", requires_grad=True)
    view = x[:, :-3]
    assert torch.equal(row_argmax(view), torch.argmax(view, dim=-1))


def test_cpu_falls_back_to_torch():
    x = torch.randn(2, 5, 33)
    assert torch.equal(row_argmax(x), torch.argmax(x, dim=-1))
