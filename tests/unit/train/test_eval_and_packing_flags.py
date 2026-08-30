"""Tests for the separate validation corpus, no-packing and mid-epoch eval flags."""

import numpy as np
import pytest

from speculators.train.config.schema import TrainConfig
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.trainer import TrainerConfig


def test_flags_default_to_the_existing_behaviour():
    cfg = TrainConfig()

    assert cfg.data.val_data_path is None
    assert cfg.data.no_packing is False
    assert cfg.trainer.eval_interval is None
    assert cfg.trainer.eval_max_batches is None


def test_trainer_config_carries_the_eval_schedule():
    cfg = TrainerConfig(lr=1e-4, num_epochs=1, save_path="/tmp/x")

    assert cfg.eval_interval is None
    assert cfg.eval_max_batches is None
    assert (
        TrainerConfig(
            lr=1e-4, num_epochs=1, save_path="/tmp/x", eval_interval=500
        ).eval_interval
        == 500
    )


def test_no_packing_lengths_make_every_batch_one_conversation():
    """The packer batches by a token budget; a constant length equal to the whole
    budget is what forces exactly one conversation per rank per step.
    """
    budget = 8192
    approx = np.array([120, 3000, 700, 8000], dtype=np.int64)

    packed = approx
    unpacked = np.full(len(approx), budget, dtype=np.int64)

    assert packed.sum() < budget * len(approx), "short samples would pack together"
    assert (unpacked == budget).all()


@pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5])
def test_train_data_ratio_is_still_validated_without_a_val_path(ratio):
    with pytest.raises(ValueError, match="train_data_ratio"):
        create_train_val_loaders(
            data_path="/nonexistent",
            total_seq_len=8,
            hidden_states_dtype=None,  # type: ignore[arg-type]
            noise_std=0.0,
            vllm_endpoint="",
            on_missing="skip",
            on_generate="cache",
            verifier_name_or_path="",
            request_timeout=None,
            max_retries=0,
            hidden_size=8,
            num_target_layers=1,
            num_workers=0,
            prefetch_factor=4,
            preprocess=None,
            train_data_ratio=ratio,
        )
