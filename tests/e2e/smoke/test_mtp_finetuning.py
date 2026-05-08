"""E2E smoke test: MTP finetuning loop on synthetic data.

Verifies frozen weights (embed_tokens, lm_head) stay bit-identical after
training, trainable MTP layer weights change, and changes are bounded.
"""

import logging
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen3Config

from speculators.models.mtp.core import MTPDraftModel, compute_step_weights
from speculators.models.mtp.data import shift_batch_mtp
from speculators.train.data import create_collate_fn
from speculators.train.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)

HIDDEN_SIZE = 64
VOCAB_SIZE = 128
SEQ_LEN = 32
NUM_SAMPLES = 8
NUM_STEPS = 3


class SyntheticMTPDataset(Dataset):
    """Yields samples in the standardized format expected by the data pipeline."""

    def __init__(self, num_samples: int, seq_len: int, hidden_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        seq_len = self.seq_len
        return {
            "hidden_states": torch.randn(
                seq_len, 3 * self.hidden_size, dtype=torch.bfloat16
            ),
            "input_ids": torch.randint(0, VOCAB_SIZE, (seq_len,)),
            "verifier_last_hidden_states": torch.randn(
                seq_len, self.hidden_size, dtype=torch.bfloat16
            ),
            "loss_mask": torch.ones(seq_len),
            "lengths": torch.tensor([seq_len], dtype=torch.long),
            "position_ids": torch.arange(seq_len, dtype=torch.long),
        }


def _make_model() -> MTPDraftModel:
    verifier_config = Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQ_LEN,
    )
    return MTPDraftModel.from_training_args(
        verifier_config=verifier_config,
        num_speculative_steps=NUM_STEPS,
    )


@pytest.mark.e2e
def test_mtp_finetuning_smoke(tmp_path: Path):
    """Short MTP training run: frozen weights unchanged, trainable weights move."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    LR = 1e-3
    NUM_EPOCHS = 2
    REL_L1_MAX = 0.5
    REL_L1_MIN = 1e-6
    MIN_CHANGED = 1
    EPS = 1e-12

    FROZEN_PATTERNS = ("embed_tokens", "lm_head")

    model = _make_model()
    model.to(dtype=torch.bfloat16)
    initial_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    step_weights = compute_step_weights(beta=0.6, num_steps=NUM_STEPS)
    train_kwargs: dict = {"step_weights": step_weights}

    dataset = SyntheticMTPDataset(NUM_SAMPLES, SEQ_LEN, HIDDEN_SIZE)
    collate_fn = create_collate_fn(
        max_len=SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        preprocess=shift_batch_mtp,
    )
    train_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    save_path = str(tmp_path / "ckpt")
    trainer_config = TrainerConfig(
        lr=LR,
        num_epochs=NUM_EPOCHS,
        save_path=save_path,
        train_call_kwargs=train_kwargs,
        scheduler_type="none",
    )

    trainer = Trainer(model, trainer_config, train_loader)
    trainer.run_training()

    finetuned_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert set(initial_sd.keys()) == set(finetuned_sd.keys())

    num_changed = 0
    for key in sorted(initial_sd.keys()):
        assert initial_sd[key].shape == finetuned_sd[key].shape, (
            f"Shape mismatch for {key}"
        )

        if any(pat in key for pat in FROZEN_PATTERNS):
            assert torch.equal(initial_sd[key], finetuned_sd[key]), (
                f"Frozen tensor {key} changed during training"
            )
            logger.info("  [frozen] %s: identical", key)
        else:
            diff = initial_sd[key] - finetuned_sd[key]
            l1_norm = finetuned_sd[key].abs().sum() + EPS
            rel_l1 = (diff.abs().sum() / l1_norm).item()
            logger.info(
                "  %s: rel_l1=%.3e  max|d|=%.3e",
                key,
                rel_l1,
                diff.abs().max().item(),
            )
            assert rel_l1 <= REL_L1_MAX, (
                f"Tensor {key} rel_l1={rel_l1:.4e} exceeds {REL_L1_MAX}"
            )
            if rel_l1 >= REL_L1_MIN:
                num_changed += 1

    assert num_changed >= MIN_CHANGED, (
        f"Expected >= {MIN_CHANGED} tensors with rel_l1 > {REL_L1_MIN}, "
        f"got {num_changed}"
    )
