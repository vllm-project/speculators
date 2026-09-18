"""Unit tests for the vocab mapping utilities used during draft model training."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset as HFDataset

from speculators.train import vocab_mapping
from speculators.train.vocab_mapping import (
    build_vocab_mappings_from_distribution,
    combine_token_frequency_distributions,
    get_target_vocab_size,
    save_token_frequency_distribution,
)


def _frequency_dataset() -> HFDataset:
    return HFDataset.from_dict(
        {
            "input_ids": [[1, 2, 2, 3], [3, 3, 4]],
            "loss_mask": [[1, 1, 0, 1], [0, 1, 1]],
        }
    ).with_format("torch")


def test_build_vocab_mappings_ranks_tokens_by_frequency():
    token_freq = {5: 10, 3: 30, 9: 20, 1: 5, 7: 30}

    # Ranked by (-frequency, token id): 3, 7 (tied at 30), 9, 5, 1; top 3 kept.
    draft_to_target, target_to_draft = build_vocab_mappings_from_distribution(
        token_freq, draft_vocab_size=3, target_vocab_size=16
    )

    assert draft_to_target.dtype == torch.long
    assert draft_to_target.shape == (3,)
    draft_idx = torch.arange(3)
    assert (draft_idx + draft_to_target).tolist() == [3, 7, 9]

    expected_target_to_draft = torch.zeros(16, dtype=torch.bool)
    expected_target_to_draft[[3, 7, 9]] = True
    assert torch.equal(target_to_draft, expected_target_to_draft)


def test_build_vocab_mappings_pads_with_unused_token_ids():
    token_freq = {2: 7, 10: 3}

    # Ranked: 2, 10; padded with the smallest unused ids < draft_vocab_size.
    draft_to_target, target_to_draft = build_vocab_mappings_from_distribution(
        token_freq, draft_vocab_size=4, target_vocab_size=16
    )

    draft_idx = torch.arange(4)
    assert (draft_idx + draft_to_target).tolist() == [0, 1, 2, 10]

    expected_target_to_draft = torch.zeros(16, dtype=torch.bool)
    expected_target_to_draft[[0, 1, 2, 10]] = True
    assert torch.equal(target_to_draft, expected_target_to_draft)


def test_build_vocab_mappings_padding_skips_already_selected_ids():
    token_freq = {0: 5, 9: 2}

    # Ranked: 0, 9; padding must skip 0 (already selected) and pick 1 instead.
    draft_to_target, _ = build_vocab_mappings_from_distribution(
        token_freq, draft_vocab_size=3, target_vocab_size=16
    )

    draft_idx = torch.arange(3)
    assert (draft_idx + draft_to_target).tolist() == [0, 1, 9]


def test_combine_token_frequency_distributions_merges_files(tmp_path: Path):
    torch.save({1: 2, 2: 3}, tmp_path / "freq_a.pt")
    torch.save({2: 4, 5: 1}, tmp_path / "freq_b.pt")
    output_path = tmp_path / "combined.pt"

    combine_token_frequency_distributions(
        [tmp_path / "freq_a.pt", tmp_path / "freq_b.pt"], output_path
    )

    assert output_path.exists()
    assert torch.load(output_path, weights_only=True) == {1: 2, 2: 7, 5: 1}


def test_save_token_frequency_distribution_counts_masked_tokens(tmp_path: Path):
    output_path = tmp_path / "nested" / "token_freq.pt"

    save_token_frequency_distribution(_frequency_dataset(), output_path)

    # Only tokens with loss_mask == 1 are counted: [1, 2, 3] and [3, 4].
    assert output_path.exists()
    assert torch.load(output_path, weights_only=True) == {1: 1, 2: 1, 3: 2, 4: 1}


def test_save_token_frequency_distribution_skips_existing_file(tmp_path: Path):
    output_path = tmp_path / "token_freq.pt"
    output_path.write_text("sentinel")

    save_token_frequency_distribution(_frequency_dataset(), output_path)

    assert output_path.read_text() == "sentinel"


def test_get_target_vocab_size_accepts_explicit_value():
    assert get_target_vocab_size(151936, None) == 151936


def test_get_target_vocab_size_rejects_both_options():
    with pytest.raises(ValueError, match="both"):
        get_target_vocab_size(100, "some/model")


def test_get_target_vocab_size_rejects_neither_option():
    with pytest.raises(ValueError, match="either"):
        get_target_vocab_size(None, None)


def test_get_target_vocab_size_loads_from_model_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        vocab_mapping.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(vocab_size=42),
    )

    assert get_target_vocab_size(None, "model-path") == 42


def test_get_target_vocab_size_unwraps_text_config(monkeypatch: pytest.MonkeyPatch):
    config = SimpleNamespace(vocab_size=0, text_config=SimpleNamespace(vocab_size=7))
    monkeypatch.setattr(
        vocab_mapping.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config,
    )

    assert get_target_vocab_size(None, "model-path", trust_remote_code=True) == 7
