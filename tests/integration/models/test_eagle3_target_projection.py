"""Teacher row selection must preserve loss and cross-depth accuracy history."""

from types import MethodType
from unittest.mock import patch

import pytest
import torch

from speculators.losses import eager
from speculators.models.eagle3.metrics import compute_metrics
from tests.integration.conftest import make_eagle3_model


@pytest.mark.parametrize(
    "mask_bits",
    [[0, 1, 0, 0, 0, 1, 0, 0, 0, 1], [0] * 6 + [1] * 4, [1] * 10, [0] * 10, None],
    ids=["sparse", "suffix", "all", "empty", "none"],
)
@pytest.mark.parametrize("depth", [1, 3, 4])
@pytest.mark.parametrize(
    "loss_name",
    [
        "kl_div_loss",
        "reverse_kl_div_loss",
        "js_div_loss",
        "ce_loss",
        "tv_loss",
        "neg_log_acceptance_loss",
        "lk_hybrid_loss",
    ],
)
def test_teacher_projection_preserves_training(mask_bits, depth, loss_name):
    torch.manual_seed(17)
    model = make_eagle3_model(
        device="cpu", dtype=torch.float32, draft_attn_impl="eager"
    )
    seq = 10
    mask = (
        torch.tensor([mask_bits], dtype=torch.bool) if mask_bits is not None else None
    )
    teacher_hs = torch.randn(1, seq, 64)
    # Independent oracle: enumerate every teacher row inspected by a chain
    # whose initial prev_correct is true, across all requested depths.
    required = set()
    for row in range(seq):
        if mask is None or mask[0, row]:
            required.update(range(row, min(seq, row + depth)))
    with patch.object(
        model.verifier_lm_head, "forward", wraps=model.verifier_lm_head.forward
    ) as head:
        selected = model._compute_training_targets(teacher_hs, mask, depth)
    assert head.call_args.args[0].shape[1] == len(required)
    dense = model.verifier_lm_head(model.verifier_norm(teacher_hs))
    torch.testing.assert_close(
        selected[:, sorted(required)], dense[:, sorted(required)]
    )

    inputs = {
        "hidden_states": torch.randn(1, seq, 192),
        "input_ids": torch.randint(0, 128, (1, seq)),
        "document_ids": torch.tensor([[0] * 5 + [1] * 5]),
        "verifier_last_hidden_states": teacher_hs,
        "loss_mask": mask,
        "ttt_steps": depth,
        "loss_config": {loss_name: (getattr(eager, loss_name), 1.0)},
    }

    def dense_targets(self, hidden, _mask, _depth):
        return self.verifier_lm_head(self.verifier_norm(hidden))

    outputs, grads = [], []
    with torch.compiler.set_stance("force_eager"):
        for use_dense in [True, False]:
            model.zero_grad(set_to_none=True)
            if use_dense:
                with patch.object(
                    model, "_compute_training_targets", MethodType(dense_targets, model)
                ):
                    output = model(**inputs)
            else:
                output = model(**inputs)
            output[1].backward()
            outputs.append(output)
            grads.append(
                {
                    n: p.grad.clone()
                    for n, p in model.named_parameters()
                    if p.grad is not None
                }
            )
    torch.testing.assert_close(outputs[0], outputs[1])
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name])


def test_mask_only_projection_changes_accuracy_history():
    # Row 1 has no loss, but chain 0 reads it at depth 1. Replacing its
    # argmax changes the metrics when that chain reaches supervised row 2.
    mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[[0.0, 2.0]] * 3], requires_grad=True)
    targets = logits.detach().clone()
    masked_targets = targets.clone()
    masked_targets[:, 1] = 0
    runs = []
    for teacher in [targets, masked_targets]:
        previous = mask.clone()
        runs.append(
            [
                compute_metrics(
                    logits,
                    teacher,
                    mask,
                    previous,
                    step,
                    1.0,
                    loss_config={"kl_div": (eager.kl_div_loss, 1.0)},
                )
                for step in range(3)
            ]
        )
    for original, masked in zip(*runs, strict=True):
        torch.testing.assert_close(original[0], masked[0])
    assert runs[0][2][1]["cond_acc_2_sum"] != runs[1][2][1]["cond_acc_2_sum"]
