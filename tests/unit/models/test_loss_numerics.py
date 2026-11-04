# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused loss regressions from TRL's TestHeadLossNumerics, adapted for Speculators.

Paper: https://arxiv.org/abs/2610.31642 (softmax collapse and cancellation).
Source: https://github.com/huggingface/trl/blob/3885117d6a1dd647d43befeddf0da482c505cb47/tests/test_kernels.py#L683
Source file blob: eb36e0d05679fc8944c4a272138e47c9507730cc (2026-11-04).

The logit values, reference calculations, and tolerances are preserved.
TRL's identity output projection becomes a transpose and a batch dimension;
CE labels become teacher logits whose argmax is the same label. The analytical
and FP64 references are retained because eager FP32 can share the kernels'
numerical errors.

API differences:
- Projection chunk sizes, score IDs, and restricted supports have no equivalent.
  The CE tests retain the full-vocabulary cases and the original loss weights.
- The 2053-token CE case preserves TRL's target positions, but these are not
  necessarily block boundaries in Speculators' autotuned kernel.
- Speculators' JSD is fixed at beta=0.5, replacing TRL's beta=0.7 parameter.
  Generalized-JSD endpoint cases (beta=1e-6 and 1-1e-6) are not applicable.

Known failures use strict xfails; use pytest --runxfail to see the raw assertions.
Loss and gradient assertions run as separate cases so neither failure hides the other.
"""

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 -- retain TRL's reference notation

from speculators.losses import resolve_loss_config


def _grad(tensor: torch.Tensor) -> torch.Tensor:
    # Missing backward plumbing must not be hidden by a numerical xfail.
    if tensor.grad is None:
        raise RuntimeError("The loss did not produce an expected gradient")
    return tensor.grad


@pytest.fixture
def device():
    # Training defaults to fused losses. CPU/eager compatibility coverage does
    # not exercise these Triton kernels, so this suite targets fused CUDA only.
    if not torch.cuda.is_available():
        pytest.skip("Fused loss numerics require CUDA")
    pytest.importorskip("triton")
    return "cuda"


@pytest.fixture
def loss_fn():
    """Adapt TRL's [vocab, rows] identity heads to [batch, rows, vocab]."""

    def compute(student, teacher, *, beta=None):
        name = "ce" if beta is None else {0.0: "kl_div", 0.5: "jsd", 1.0: "rkl"}[beta]
        fn = resolve_loss_config(name, "fused")[name][0]
        if beta is None:
            # CE accepts teacher logits, but only their argmax is used.
            targets = F.one_hot(teacher, num_classes=student.shape[0])
            targets = targets.to(dtype=student.dtype)
        else:
            targets = teacher.t()
        return fn(student.t().unsqueeze(0), targets.unsqueeze(0)).squeeze(0)

    return compute


class TestLossNumerics:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Softmax collapse: confident CE loses the small loss/target gradient",
    )
    @pytest.mark.parametrize("check", ["loss", "gradient"])
    def test_confident_target_large_vocabulary(self, device, loss_fn, check):
        """TRL's across_triton_blocks case, with the same 2053 logits."""
        targets = torch.tensor([0, 1024, 2052], device=device)
        weight = torch.full((2053, 3), -9.0, device=device)
        weight[targets, torch.arange(3, device=device)] = 20.0
        weight.requires_grad_()
        loss = loss_fn(weight, targets)
        loss.sum().backward()
        grad = _grad(weight)
        # Even fp64 CE loses relative precision here; retain TRL's analytic oracle.
        tail_mass = 2052 * math.exp(-29.0)
        expected_loss = torch.tensor(
            3 * math.log1p(tail_mass), device=device, dtype=torch.float64
        )
        expected_grad = torch.full(
            (2053, 3),
            math.exp(-29.0) / (1 + tail_mass),
            device=device,
            dtype=torch.float64,
        )
        expected_grad[targets, torch.arange(3, device=device)] = -tail_mass / (
            1 + tail_mass
        )
        if check == "loss":
            torch.testing.assert_close(
                loss.double().sum(), expected_loss, atol=0, rtol=1e-5
            )
        else:
            torch.testing.assert_close(grad.double(), expected_grad, atol=0, rtol=1e-5)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Softmax collapse: confident CE loses the small loss/target gradient",
    )
    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
    )
    @pytest.mark.parametrize("check", ["loss", "gradient"])
    def test_confident_target(self, device, loss_fn, dtype, check):
        # BF16 inputs retain BF16 leaf gradients; fused loss math/output are FP32.
        weight = torch.tensor(
            [[9.0, -9.0, -9.0], [-9.0, 9.0, -9.0], [-9.0, -9.0, 9.0]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        targets = torch.arange(3, device=device)
        losses = loss_fn(weight, targets)
        advantages = torch.tensor([0.5, -1.0, 2.0], device=device)
        # TRL sums target and score log-probs; both select the same target.
        loss = (2 * losses * advantages).sum()
        loss.backward()
        grad = _grad(weight)
        logits = weight.detach().double().t().requires_grad_()
        ref_logps = logits.log_softmax(-1)
        ref_loss = -2 * (ref_logps.diagonal() * advantages.double()).sum()
        ref_loss.backward()
        if check == "loss":
            torch.testing.assert_close(loss.double(), ref_loss, atol=0, rtol=1e-5)
        else:
            torch.testing.assert_close(
                grad.double(),
                _grad(logits).t(),
                atol=0,
                rtol=1e-2 if dtype == torch.bfloat16 else 1e-5,
            )

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0], ids=["kl", "jsd", "rkl"])
    @pytest.mark.parametrize(
        "perturbation",
        [
            pytest.param(0.0, id="equal"),
            pytest.param(
                1e-7,
                id="near",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="Softmax cancellation: nearby teacher gradient",
                ),
            ),
        ],
    )
    def test_near_teacher(self, device, loss_fn, beta, perturbation):
        teacher = torch.tensor([[2.0, 1.0, 0.0], [2.0, 0.01, 0.0]], device=device).t()
        delta = torch.tensor(
            [[-perturbation, perturbation, 0.0], [0.0, perturbation / 10, 0.0]],
            device=device,
        ).t()
        student = (teacher + delta).requires_grad_()
        divergence = loss_fn(student, teacher, beta=beta)
        divergence.sum().backward()
        logits = student.detach().double().t().requires_grad_()
        log_q = logits.log_softmax(-1)
        log_p = teacher.double().t().log_softmax(-1)
        if beta == 0:
            ref = (log_p.exp() * (log_p - log_q)).sum()
        elif beta == 1:
            ref = (log_q.exp() * (log_q - log_p)).sum()
        else:
            log_m = torch.logaddexp(log_q + math.log(1 - beta), log_p + math.log(beta))
            ref = (
                (1 - beta) * log_q.exp() * (log_q - log_m)
                + beta * log_p.exp() * (log_p - log_m)
            ).sum()
        ref.backward()
        if perturbation == 0:
            assert torch.count_nonzero(_grad(student)) == 0
        else:
            torch.testing.assert_close(
                _grad(student).double().t(), _grad(logits), atol=1e-15, rtol=1e-5
            )

    @pytest.mark.parametrize(
        "check",
        [
            "loss",
            pytest.param(
                "gradient",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="Concentrated distributions lose small divergence gradients",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0], ids=["kl", "jsd", "rkl"])
    def test_concentrated_teacher(self, device, loss_fn, beta, check):
        student = (
            torch.tensor(
                [[9.0, -9.0, -9.0], [9.0, -9.0, -9.0], [20.0, -20.0, -20.0]],
                device=device,
            )
            .t()
            .requires_grad_()
        )
        teacher = torch.tensor(
            [[40.0, -40.0, -40.0], [-9.0, 9.0, -9.0], [-20.0, 20.0, -20.0]],
            device=device,
        ).t()
        divergence = loss_fn(student, teacher, beta=beta)
        divergence.sum().backward()
        # Use the paper's normalization even for the float64 reference.
        logits = student.detach().double().t().requires_grad_()
        is_max = torch.arange(3, device=device) == logits.argmax(-1, keepdim=True)
        shifted = (logits - logits.max(-1, keepdim=True).values).masked_fill(
            is_max, 0.0
        )
        log_q = shifted - F.softplus(
            shifted.masked_fill(is_max, -torch.inf).logsumexp(-1, keepdim=True)
        )
        log_p = teacher.double().t().log_softmax(-1)
        if beta == 0:
            ref = (log_p.exp() * (log_p - log_q)).sum()
        elif beta == 1:
            ref = (log_q.exp() * (log_q - log_p)).sum()
        else:
            log_m = torch.logaddexp(log_q + math.log(1 - beta), log_p + math.log(beta))
            ref = (
                (1 - beta) * log_q.exp() * (log_q - log_m)
                + beta * log_p.exp() * (log_p - log_m)
            ).sum()
        ref.backward()
        if check == "loss":
            torch.testing.assert_close(
                divergence.double().sum(), ref, atol=0, rtol=1e-5
            )
        else:
            torch.testing.assert_close(
                _grad(student).double().t(), _grad(logits), atol=1e-22, rtol=1e-5
            )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Independent logit offsets lose small divergence gradients",
    )
    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0], ids=["kl", "jsd", "rkl"])
    def test_independent_logit_shifts(self, device, loss_fn, beta):
        epsilon = 2**-20
        student = torch.tensor(
            [[32.0, -epsilon], [32.0, 0.0], [32.0, epsilon]],
            device=device,
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[-epsilon, 32.0], [0.0, 32.0], [epsilon, 32.0]], device=device
        )
        loss = loss_fn(student, teacher, beta=beta)
        loss.sum().backward()
        logits = student.detach().double().t().requires_grad_()
        log_q = logits.log_softmax(-1)
        log_p = teacher.double().t().log_softmax(-1)
        if beta == 0:
            reference = (log_p.exp() * (log_p - log_q)).sum()
        elif beta == 1:
            reference = (log_q.exp() * (log_q - log_p)).sum()
        else:
            log_m = torch.logaddexp(log_q + math.log1p(-beta), log_p + math.log(beta))
            reference = (
                (1 - beta) * log_q.exp() * (log_q - log_m)
                + beta * log_p.exp() * (log_p - log_m)
            ).sum()
        reference.backward()
        torch.testing.assert_close(
            _grad(student).double().t(), _grad(logits), atol=1e-13, rtol=1e-5
        )
