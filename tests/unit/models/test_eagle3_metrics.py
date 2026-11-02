"""Unit tests for the Eagle3 loss and metrics."""

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.models.eagle3.metrics import align_for_step, compute_metrics
from speculators.models.metrics import compute_accuracy_single_step

_EAGER_LOSS = resolve_loss_config("kl_div", "eager")

# In the order compute_accuracy_single_step returns them.
_ACC_KEYS = (
    "full_acc_{}_sum",
    "full_acc_{}_total",
    "cond_acc_{}_sum",
    "cond_acc_{}_total",
)


class TestComputeMetricsIdAlignment:
    """``pred_ids``/``target_ids`` arrive unaligned and must be sliced per step."""

    @staticmethod
    def _inputs(seed: int = 0):
        torch.manual_seed(seed)
        logits = torch.randn(1, 12, 16)
        targets = torch.randn(1, 12, 16)
        loss_mask = torch.randint(0, 2, (1, 12), dtype=torch.bool)
        return logits, targets, loss_mask

    @pytest.mark.parametrize("ttt_step", [0, 1, 2])
    @pytest.mark.parametrize("with_loss_mask", [False, True])
    def test_accuracy_matches_argmax_of_aligned_tensors(self, ttt_step, with_loss_mask):
        """The ids give the accuracy the aligned logits and targets would give."""
        logits, targets, mask = self._inputs()
        loss_mask = mask if with_loss_mask else None
        prev_correct = mask if with_loss_mask else torch.ones(1, 12, dtype=torch.bool)

        # compute_accuracy_single_step writes into prev_correct, so the
        # reference and the call under test each need their own copy.
        s_logits, s_targets, s_loss_mask, s_prev_correct = align_for_step(
            logits, targets, loss_mask, prev_correct.clone(), ttt_step
        )
        if s_loss_mask is None:
            s_loss_mask = torch.ones_like(s_prev_correct)
        expected = compute_accuracy_single_step(
            torch.argmax(s_logits, dim=-1),
            torch.argmax(s_targets, dim=-1),
            s_loss_mask,
            s_prev_correct,
        )

        _, metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            prev_correct.clone(),
            ttt_step,
            1.0,
            torch.argmax(logits, dim=-1),
            torch.argmax(targets, dim=-1),
            loss_config=_EAGER_LOSS,
        )

        for name, value in zip(_ACC_KEYS, expected, strict=True):
            key = name.format(ttt_step)
            assert torch.equal(metrics[key], value), key

    def test_ids_are_shifted_against_each_other(self):
        """A draft that is right at step k only scores if the shift is applied.

        The caller's ids cover all positions, so compute_metrics has to apply
        the same slicing align_for_step applies to the tensors they came from.
        An off-by-one here silently reports the wrong step's accuracy.
        """
        seq_len, vocab_size, ttt_step = 12, 16, 2
        target_ids = torch.arange(seq_len).unsqueeze(0)
        # At step k the draft predicts the token k positions further along.
        pred_ids = target_ids + ttt_step
        loss_mask = torch.ones(1, seq_len, dtype=torch.bool)

        _, metrics = compute_metrics(
            torch.nn.functional.one_hot(pred_ids, vocab_size).float(),
            torch.nn.functional.one_hot(target_ids, vocab_size).float(),
            loss_mask,
            loss_mask.clone(),
            ttt_step,
            1.0,
            pred_ids,
            target_ids,
            loss_config=_EAGER_LOSS,
        )

        aligned_len = seq_len - ttt_step
        for name in _ACC_KEYS:
            key = name.format(ttt_step)
            assert metrics[key].item() == aligned_len, key
