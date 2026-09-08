"""Per-position reference-prefix counts."""

import pytest
import torch

from speculators.models.metrics import (
    compute_block_reference_metrics,
    compute_reference_prefix_metrics,
    compute_sampled_reference_metrics,
)
from speculators.train.utils import normalize_counted_metrics


def _assert_counts(metrics, expected):
    assert len(metrics) == 2 * len(expected)
    for position, (correct, total) in enumerate(expected):
        assert metrics[f"reference_acc_at_pos_{position}_sum"] == correct
        assert metrics[f"reference_acc_at_pos_{position}_total"] == total


@pytest.mark.parametrize(
    ("predictions", "second_correct"),
    [([[1, 2], [0, 0]], 1), ([[1, 0], [0, 2]], 0)],
)
def test_prefix_correlation_and_batch_rank_pooling(predictions, second_correct):
    # Both cases have 50% independent accuracy at each position, but different
    # prefix agreement. Splitting counts cannot change either result.
    inputs = torch.tensor([[0, 1, 2]]).expand(2, -1)
    preds = torch.tensor(predictions).unsqueeze(1)
    parts = [
        compute_reference_prefix_metrics(
            preds[a:b], inputs[a:b], torch.tensor([1]), None, None
        )
        for a, b in [(0, 1), (1, 2), (2, 2)]
    ]
    combined = {key: sum(part[key].item() for part in parts) for key in parts[0]}
    _assert_counts(combined, [(1, 2), (second_correct, 2)])
    assert normalize_counted_metrics(combined, world_size=3) == {
        "reference_acc_at_pos_0": 0.5,
        "reference_acc_at_pos_1": second_correct / 2,
    }


@pytest.mark.parametrize(
    ("invalid", "totals"),
    [
        ("mask", [2, 1, 1]),
        ("missing", [2, 1, 1]),
        ("document", [2, 1, 1]),
        ("padding", [1, 1, 1]),
    ],
)
def test_prefixes_require_all_references_to_be_eligible(invalid, totals):
    inputs = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    preds = torch.tensor([[[1, 2, 3], [5, 6, 7]]])
    mask, docs = torch.ones_like(inputs), torch.zeros_like(inputs)
    observed = torch.ones_like(preds, dtype=torch.bool)
    if invalid == "mask":
        mask[:, 6] = 0
    elif invalid == "document":
        docs[:, 6:] = 1
    elif invalid == "padding":
        docs[:, 4:] = -1
    else:
        observed[:, 1, 1] = False
    metrics = compute_reference_prefix_metrics(
        preds,
        inputs,
        torch.tensor([1, 5]),
        mask,
        docs,
        observed=observed,
    )
    _assert_counts(metrics, [(n, n) for n in totals])


def test_sequence_end_and_missing_anchor():
    metrics = compute_reference_prefix_metrics(
        torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 3]]]),
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([0, 1, 2]),
        None,
        None,
    )
    _assert_counts(metrics, [(2, 2), (2, 2), (1, 1)])


def test_pruned_vocabulary_miss_breaks_the_prefix():
    # Draft IDs 0,1,2 map to verifier IDs 2,4,7. Reference ID 5 is absent.
    metrics = compute_reference_prefix_metrics(
        torch.tensor([[[0, 1, 2], [0, 1, 2]]]),
        torch.tensor([[0, 2, 4, 7, 0, 2, 5, 7]]),
        torch.tensor([1, 5]),
        None,
        None,
        d2t=torch.tensor([2, 3, 5]),
    )
    _assert_counts(metrics, [(2, 2), (1, 2), (1, 2)])


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_block_alignment_reports_all_positions_and_ignores_padding(sample_from_anchor):
    inputs = torch.tensor([[8, 9, 1, 2, 3, 4, 5, 6]])
    preds = torch.tensor(
        [[1, 2, 3, 0, 9, 1, 2, 3]] if sample_from_anchor else [[0, 1, 2, 3, 0, 9, 1, 2]]
    )
    metrics = compute_block_reference_metrics(
        preds,
        inputs,
        torch.tensor([1, 2, 3, 4, 0, 1, 2, 3]),
        torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]]),
        torch.ones_like(inputs),
        torch.zeros_like(inputs),
        4,
        sample_from_anchor,
    )
    _assert_counts(metrics, [(1, 1)] * 3 + ([(0, 1)] if sample_from_anchor else []))


def test_sampled_depths_require_complete_prefixes_even_when_shuffled():
    inputs = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    metrics = compute_sampled_reference_metrics(
        torch.tensor([[3, 4, 1, 2, 1, 4, 6]]),
        inputs,
        torch.tensor([0, 3, 0, 0, -1, 0, 3]),
        torch.tensor([2, 0, 0, 1, 1, 3, 2]),
        4,
        torch.ones_like(inputs),
        torch.zeros_like(inputs),
    )
    _assert_counts(metrics, [(2, 2), (1, 1), (1, 1), (1, 1)])


@pytest.mark.parametrize("horizon", [0, 1, 2, 5])
def test_empty_sequence_keeps_configured_count_keys(horizon):
    metrics = compute_reference_prefix_metrics(
        torch.zeros(1, 0, horizon, dtype=torch.long),
        torch.zeros(1, 0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        None,
        None,
    )
    _assert_counts(metrics, [(0, 0)] * horizon)
    assert len(metrics) == 2 * horizon
