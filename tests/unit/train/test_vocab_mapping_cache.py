"""Unit test: reusing a cached token_freq.pt must warn, not silently return."""

import logging

import torch
from datasets import Dataset as HFDataset

from speculators.train.vocab_mapping import save_token_frequency_distribution


def _tiny_dataset():
    data = HFDataset.from_dict(
        {"input_ids": [[1, 2, 2, 3]], "loss_mask": [[1, 1, 1, 0]]}
    )
    return data.with_format("torch")


def test_fresh_run_counts_and_saves(tmp_path):
    path = tmp_path / "token_freq.pt"
    save_token_frequency_distribution(_tiny_dataset(), output_path=path)
    freq = torch.load(path, weights_only=True)
    assert freq == {1: 1, 2: 2}


def test_existing_file_is_reused_with_warning(tmp_path, caplog):
    path = tmp_path / "token_freq.pt"
    torch.save({99: 5}, path)
    with caplog.at_level(logging.WARNING, logger="speculators"):
        save_token_frequency_distribution(_tiny_dataset(), output_path=path)
    assert torch.load(path, weights_only=True) == {99: 5}  # cache kept
    assert any("Reusing existing token frequency" in r.message for r in caplog.records)
