"""Unit tests for fast_mtp_trainer_utils — build_fast_mtp_trainer."""

from pathlib import Path

import pytest
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp import FastMTPConfig, FastMTPSpeculator
from speculators.proposals import GreedyTokenProposalConfig
from speculators.train.fast_mtp_trainer_utils import build_fast_mtp_trainer
from speculators.train.trainer import Trainer

H = 8
VOCAB = 64
SEQ_LEN = 10
MAX_LEN = 8


def _tiny_config() -> FastMTPConfig:
    tc = Qwen2Config(
        hidden_size=H,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        vocab_size=VOCAB,
        max_position_embeddings=32,
    )
    return FastMTPConfig(
        transformer_layer_config=tc,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
    )


def _save_pt(path: Path, n: int = 6) -> None:
    for i in range(n):
        torch.save(
            {
                "input_ids": torch.randint(0, VOCAB, (SEQ_LEN,)),
                "hidden_states": torch.randn(SEQ_LEN, H),
                "loss_mask": torch.ones(SEQ_LEN, dtype=torch.long),
            },
            str(path / f"data_{i}.pt"),
        )


@pytest.fixture
def model() -> FastMTPSpeculator:
    return FastMTPSpeculator(_tiny_config())


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    _save_pt(tmp_path)
    return tmp_path


class TestBuildFastMtpTrainer:
    def test_returns_trainer(
        self, model: FastMTPSpeculator, data_dir: Path, tmp_path: Path
    ) -> None:
        trainer = build_fast_mtp_trainer(
            model, data_dir, tmp_path / "ckpt", max_len=MAX_LEN, batch_size=2
        )
        assert isinstance(trainer, Trainer)

    def test_train_call_kwargs_has_step_weights(
        self, model: FastMTPSpeculator, data_dir: Path, tmp_path: Path
    ) -> None:
        trainer = build_fast_mtp_trainer(
            model, data_dir, tmp_path / "ckpt", max_len=MAX_LEN, batch_size=2
        )
        assert "step_weights" in trainer.config.train_call_kwargs
        assert "step_weights" in trainer.config.val_call_kwargs

    def test_train_call_kwargs_return_dict_false(
        self, model: FastMTPSpeculator, data_dir: Path, tmp_path: Path
    ) -> None:
        """Trainer unpacks (logits, loss, metrics) — requires return_dict=False."""
        trainer = build_fast_mtp_trainer(
            model, data_dir, tmp_path / "ckpt", max_len=MAX_LEN, batch_size=2
        )
        assert trainer.config.train_call_kwargs.get("return_dict") is False

    def test_train_loader_batch_shapes(
        self, model: FastMTPSpeculator, data_dir: Path, tmp_path: Path
    ) -> None:
        trainer = build_fast_mtp_trainer(
            model, data_dir, tmp_path / "ckpt", max_len=MAX_LEN, batch_size=2
        )
        batch = next(iter(trainer.train_loader))
        assert batch["input_ids"].ndim == 2
        assert batch["hidden_states"].ndim == 3
        assert batch["hidden_states"].shape[-1] == H
        assert batch["labels"].shape == batch["input_ids"].shape

    def test_scheduler_type_flows_to_config(
        self, model: FastMTPSpeculator, data_dir: Path, tmp_path: Path
    ) -> None:
        trainer = build_fast_mtp_trainer(
            model, data_dir, tmp_path / "ckpt", max_len=MAX_LEN, scheduler_type="cosine"
        )
        assert trainer.config.scheduler_type == "cosine"

    def test_model_layers_property(self, model: FastMTPSpeculator) -> None:
        """verify_training_compatible checks for .layers; returns mtp_layers."""
        assert model.layers is model.mtp_layers
