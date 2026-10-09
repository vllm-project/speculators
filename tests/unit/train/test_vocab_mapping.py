"""Tests for speculators.train.vocab_mapping."""

import torch

from speculators.train.vocab_mapping import (
    build_vocab_mappings_from_distribution,
    combine_token_frequency_distributions,
    save_token_frequency_distribution,
)


class _FakeDataset:
    """Minimal iterable stub exposing input_ids/loss_mask items."""

    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


def test_build_vocab_mappings_ranks_by_frequency():
    token_freq = {5: 10, 3: 20, 7: 5}

    draft_to_target, target_to_draft = build_vocab_mappings_from_distribution(
        token_freq, draft_vocab_size=2, target_vocab_size=10
    )

    # top-2 by frequency are tokens 3 and 5; offsets encode
    # target_id = draft_idx + draft_to_target[draft_idx]
    assert draft_to_target.tolist() == [3, 4]
    for draft_idx, offset in enumerate(draft_to_target.tolist()):
        assert draft_idx + offset in (3, 5)
    assert target_to_draft.dtype == torch.bool
    assert target_to_draft.tolist() == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_build_vocab_mappings_pads_with_smallest_missing_ids():
    token_freq = {2: 5}

    draft_to_target, target_to_draft = build_vocab_mappings_from_distribution(
        token_freq, draft_vocab_size=4, target_vocab_size=8
    )

    # only token 2 observed: pad with 0, 1, 3 -> sorted [0, 1, 2, 3]
    assert draft_to_target.tolist() == [0, 0, 0, 0]
    for draft_idx, offset in enumerate(draft_to_target.tolist()):
        assert draft_idx + offset in (0, 1, 2, 3)
    assert target_to_draft.sum().item() == 4


def test_combine_token_frequency_distributions(tmp_path):
    first = tmp_path / "freq1.pt"
    second = tmp_path / "freq2.pt"
    out = tmp_path / "combined.pt"
    torch.save({1: 2, 2: 3}, first)
    torch.save({2: 4, 3: 5}, second)

    combine_token_frequency_distributions([first, second], out)

    assert torch.load(out, weights_only=True) == {1: 2, 2: 7, 3: 5}


def test_save_token_frequency_distribution_returns_path(tmp_path):
    dataset = _FakeDataset(
        [
            {
                "input_ids": torch.tensor([1, 2, 2, 3]),
                "loss_mask": torch.tensor([1, 1, 0, 1]),
            },
            {
                "input_ids": torch.tensor([2, 3, 3]),
                "loss_mask": torch.tensor([0, 1, 1]),
            },
        ]
    )
    out = tmp_path / "freq.pt"

    result = save_token_frequency_distribution(dataset, out)

    assert result == out
    assert torch.load(out, weights_only=True) == {1: 1, 2: 1, 3: 3}


def test_save_token_frequency_distribution_reuses_existing_file(tmp_path):
    out = tmp_path / "freq.pt"
    torch.save({9: 9}, out)

    result = save_token_frequency_distribution(_FakeDataset([]), out)

    assert result == out
    assert torch.load(out, weights_only=True) == {9: 9}
