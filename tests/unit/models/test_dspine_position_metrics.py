"""Prefix acceptance diagnostics must agree with the training EAL proxy."""

import pytest
import torch

from speculators.models.dspine.metrics import compute_dspine_loss
from speculators.train.utils import normalize_counted_metrics


@pytest.mark.parametrize(
    ("mask", "expected_counts", "expected_blocks"),
    [
        ([0, 1, 1, 1] * 4, [3, 2, 1], 4),
        # Exclude the last block entirely; stop the first at a masked suffix.
        ([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0], [3, 1, 0], 3),
        ([0] * 16, [0, 0, 0], 0),
    ],
)
def test_prefix_acceptance_counts_match_eal(mask, expected_counts, expected_blocks):
    # Selected token is 0 throughout. Correctness for the four draft blocks:
    # TTT, TFF, TTF, FTT. Later correct tokens cannot revive a rejected prefix.
    labels = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]])
    candidates = torch.tensor([0, 1]).view(1, 1, 2).expand(1, 16, 2)
    refined = torch.tensor([2.0, 0.0]).view(1, 1, 2).expand(1, 16, 2)
    _, metrics = compute_dspine_loss(
        unary_logits=torch.zeros(1, 16, 3),
        refined_candidate_logits=refined,
        candidate_ids=candidates,
        targets=torch.zeros(1, 16, 3),
        target_ids=labels,
        layer_features=[],
        target_features=torch.zeros(1, 16, 2),
        loss_mask=torch.tensor([mask], dtype=torch.float32),
        block_size=4,
        gamma=7.0,
        alpha=0.1,
        embedding_beta=0.0,
        embedding_warmup_steps=0,
        global_step=0,
        transition_min_target_mass=1e-4,
    )

    for position, expected in enumerate(expected_counts, start=1):
        count = metrics[f"acceptance_rate_pos_{position}_sum"]
        total = metrics[f"acceptance_rate_pos_{position}_total"]
        assert count.item() == expected
        assert total.item() == expected_blocks
        assert count.dtype == torch.float32
        assert not count.requires_grad
        assert torch.isfinite(count)
        assert torch.isfinite(total)
        # Token accuracy is independent of earlier rejections, unlike prefix acceptance.
        expected_accuracy = sum(
            mask[index] and labels[0, index].item() == 0
            for index in range(position, 16, 4)
        )
        assert metrics[f"position_{position}_acc_sum"].item() == expected_accuracy
        assert metrics[f"position_{position}_acc_total"].item() == sum(
            mask[position::4]
        )
    assert "acceptance_rate_pos_0_sum" not in metrics
    assert metrics["eal_sum"].item() == expected_blocks + sum(expected_counts)
    assert metrics["eal_total"].item() == expected_blocks


@pytest.mark.parametrize("block_size", [2, 4, 8, 16])
def test_position_metrics_follow_configured_block_size(block_size):
    _, metrics = compute_dspine_loss(
        unary_logits=torch.zeros(1, block_size, 3),
        refined_candidate_logits=torch.tensor([2.0, 0.0])
        .view(1, 1, 2)
        .expand(1, block_size, 2),
        candidate_ids=torch.tensor([0, 1]).view(1, 1, 2).expand(1, block_size, 2),
        targets=torch.zeros(1, block_size, 3),
        target_ids=torch.zeros(1, block_size, dtype=torch.long),
        layer_features=[],
        target_features=torch.zeros(1, block_size, 2),
        loss_mask=torch.tensor([[0.0] + [1.0] * (block_size - 1)]),
        block_size=block_size,
        gamma=7.0,
        alpha=0.1,
        embedding_beta=0.0,
        embedding_warmup_steps=0,
        global_step=0,
        transition_min_target_mass=1e-4,
    )
    for pos in range(1, block_size):
        assert metrics[f"position_{pos}_acc_sum"].item() == 1
        assert metrics[f"position_{pos}_acc_total"].item() == 1
        assert metrics[f"acceptance_rate_pos_{pos}_sum"].item() == 1
        assert metrics[f"acceptance_rate_pos_{pos}_total"].item() == 1
    assert metrics["full_acc_sum"].item() == block_size - 1
    assert metrics["full_acc_total"].item() == block_size - 1
    assert {key for key in metrics if key.startswith("position_")} == {
        f"position_{pos}_acc_{suffix}"
        for pos in range(1, block_size)
        for suffix in ("sum", "total")
    }
    assert {key for key in metrics if key.startswith("acceptance_rate_pos_")} == {
        f"acceptance_rate_pos_{pos}_{suffix}"
        for pos in range(1, block_size)
        for suffix in ("sum", "total")
    }


def test_raw_counts_pool_across_workers():
    # Workers/batches contribute unequal numbers of blocks; reduce counts first.
    reduced = {
        "acceptance_rate_pos_1_sum": 1.0 + 2.0,
        "acceptance_rate_pos_1_total": 1.0 + 3.0,
    }
    assert normalize_counted_metrics(reduced)["acceptance_rate_pos_1"] == pytest.approx(
        0.75
    )
