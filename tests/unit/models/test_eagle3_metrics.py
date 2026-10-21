"""Unit tests for the Eagle3 loss and metrics."""

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.models.eagle3.metrics import compute_metrics

_EAGER_LOSS = resolve_loss_config("kl_div", "eager")


class TestComputeMetricsPrecomputedIds:
    """``pred_ids``/``target_ids`` are an optimisation, not a behaviour change."""

    @staticmethod
    def _inputs(seed: int = 0):
        torch.manual_seed(seed)
        logits = torch.randn(1, 12, 16)
        targets = torch.randn(1, 12, 16)
        loss_mask = torch.randint(0, 2, (1, 12), dtype=torch.bool)
        return logits, targets, loss_mask

    @pytest.mark.parametrize("ttt_step", [0, 1, 2])
    @pytest.mark.parametrize("with_loss_mask", [False, True])
    def test_matches_internal_argmax(self, ttt_step, with_loss_mask):
        """Passing unaligned ids gives exactly the metrics computed internally.

        The caller's ids cover all positions; compute_metrics has to apply the
        same slicing align_for_step applies to the tensors they came from. An
        off-by-one here silently reports the wrong step's accuracy.
        """
        logits, targets, mask = self._inputs()
        loss_mask = mask if with_loss_mask else None

        def run(**ids):
            prev_correct = (
                mask.clone() if with_loss_mask else torch.ones(1, 12, dtype=torch.bool)
            )
            return compute_metrics(
                logits,
                targets,
                loss_mask,
                prev_correct,
                ttt_step,
                1.0,
                loss_config=_EAGER_LOSS,
                **ids,
            )

        expected_loss, expected = run()
        actual_loss, actual = run(
            pred_ids=torch.argmax(logits, dim=-1),
            target_ids=torch.argmax(targets, dim=-1),
        )

        assert torch.equal(actual_loss, expected_loss)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert torch.equal(actual[key], value), key

    def test_only_target_ids_precomputed(self):
        """The two ids are independent: supplying one must not shift the other."""
        logits, targets, mask = self._inputs(seed=3)

        def run(**ids):
            return compute_metrics(
                logits,
                targets,
                mask,
                mask.clone(),
                2,
                1.0,
                loss_config=_EAGER_LOSS,
                **ids,
            )

        _, expected = run()
        _, actual = run(target_ids=torch.argmax(targets, dim=-1))
        for key, value in expected.items():
            assert torch.equal(actual[key], value), key


class TestAlignedLogits:
    """``logits_aligned`` lets the caller project only the useful positions."""

    @staticmethod
    def _inputs(seed: int = 0):
        torch.manual_seed(seed)
        logits = torch.randn(1, 12, 16)
        targets = torch.randn(1, 12, 16)
        loss_mask = torch.randint(0, 2, (1, 12), dtype=torch.bool)
        return logits, targets, loss_mask

    @staticmethod
    def _run(logits, targets, mask, ttt_step, **kwargs):
        return compute_metrics(
            logits,
            targets,
            mask,
            mask.clone(),
            ttt_step,
            1.0,
            loss_config=_EAGER_LOSS,
            pred_ids=torch.argmax(logits, dim=-1),
            target_ids=torch.argmax(targets, dim=-1),
            **kwargs,
        )

    @pytest.mark.parametrize("ttt_step", [0, 1, 2])
    def test_pre_sliced_logits_give_the_same_result(self, ttt_step):
        """Slicing outside compute_metrics must not change anything it returns.

        ``pred_ids`` stays unaligned in both cases -- it is the argmax over
        every position, including the ones the loss drops.
        """
        logits, targets, mask = self._inputs()
        expected_loss, expected = self._run(logits, targets, mask, ttt_step)

        sliced = logits[:, :-ttt_step] if ttt_step > 0 else logits
        actual_loss, actual = compute_metrics(
            sliced,
            targets,
            mask,
            mask.clone(),
            ttt_step,
            1.0,
            loss_config=_EAGER_LOSS,
            pred_ids=torch.argmax(logits, dim=-1),
            target_ids=torch.argmax(targets, dim=-1),
            logits_aligned=True,
        )

        assert torch.equal(actual_loss, expected_loss)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert torch.equal(actual[key], value), key

    def test_the_dropped_positions_never_had_a_gradient(self):
        """The trailing positions only ever received a zero gradient.

        Which is the point of projecting them outside the graph: the backward
        of the slice has to allocate a zero tensor the full width of the
        vocabulary to put that nothing into.
        """
        logits, targets, mask = self._inputs(seed=5)
        logits.requires_grad_(True)

        loss, _ = self._run(logits, targets, mask, 2)
        loss.backward()

        assert logits.grad is not None
        assert torch.equal(logits.grad[:, -2:], torch.zeros(1, 2, 16))
        assert logits.grad[:, :-2].abs().sum() > 0
