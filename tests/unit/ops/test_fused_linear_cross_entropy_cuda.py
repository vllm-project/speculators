"""Opt-in correctness test for the real Liger Triton CE kernel."""

import os

import pytest
import torch
from torch.nn.functional import cosine_similarity, cross_entropy

from speculators.ops.fused_linear_cross_entropy import frozen_linear_cross_entropy

pytestmark = pytest.mark.skipif(
    os.environ.get("SPECULATORS_RUN_LIGER_TESTS") != "1"
    or not torch.cuda.is_available(),
    reason="set SPECULATORS_RUN_LIGER_TESTS=1 on an exclusive CUDA GPU",
)


def test_bf16_loss_accuracy_and_weighted_hidden_gradient():
    torch.manual_seed(123)
    token_count, hidden_size, vocab_size = 67, 128, 1024
    hidden = torch.randn(
        token_count,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (token_count,), device="cuda")
    token_weights = torch.linspace(0.0, 1.75, token_count, device="cuda")

    loss, accuracy = frozen_linear_cross_entropy(hidden, weight, target)
    (loss * token_weights).sum().backward()
    assert hidden.grad is not None
    fused_grad = hidden.grad.float().clone()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_logits = reference_hidden @ weight.t()
    reference_loss = cross_entropy(reference_logits, target, reduction="none")
    (reference_loss * token_weights).sum().backward()
    assert reference_hidden.grad is not None
    reference_grad = reference_hidden.grad.float()

    assert (loss - reference_loss.float()).abs().mean().item() < 0.1
    assert torch.equal(accuracy, (reference_logits.argmax(dim=-1) == target).float())
    cosine = cosine_similarity(fused_grad.flatten(), reference_grad.flatten(), dim=0)
    assert cosine.item() > 0.999
    assert torch.isfinite(fused_grad).all()
