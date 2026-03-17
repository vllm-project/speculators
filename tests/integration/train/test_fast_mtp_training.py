"""Integration test for FastMTP training: tiny synthetic model + data, a few steps."""

from pathlib import Path

import pytest
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp import FastMTPConfig, FastMTPSpeculator
from speculators.proposals import GreedyTokenProposalConfig
from speculators.train.fast_mtp_trainer_utils import build_fast_mtp_trainer

H = 16
VOCAB = 128
SEQ_LEN = 12
MAX_LEN = 10
NUM_SAMPLES = 20


def _tiny_model() -> FastMTPSpeculator:
    tc = Qwen2Config(
        hidden_size=H,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        vocab_size=VOCAB,
        max_position_embeddings=32,
    )
    cfg = FastMTPConfig(
        transformer_layer_config=tc,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
    )
    return FastMTPSpeculator(cfg)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    for i in range(NUM_SAMPLES):
        torch.save(
            {
                "input_ids": torch.randint(0, VOCAB, (SEQ_LEN,)),
                "hidden_states": torch.randn(SEQ_LEN, H),
                "loss_mask": torch.ones(SEQ_LEN, dtype=torch.long),
            },
            str(tmp_path / f"data_{i}.pt"),
        )
    return tmp_path


def test_training_loop_runs_and_checkpoints_saved(
    data_dir: Path, tmp_path: Path
) -> None:
    """Run 2 epochs with a tiny model; verify checkpoints are written."""
    ckpt_dir = tmp_path / "ckpt"
    trainer = build_fast_mtp_trainer(
        _tiny_model(),
        data_dir,
        ckpt_dir,
        max_len=MAX_LEN,
        lr=1e-4,
        num_epochs=2,
        batch_size=2,
        train_ratio=0.8,
        scheduler_type="none",
        hidden_states_dtype=torch.float32,
    )
    trainer.run_training()
    assert any(ckpt_dir.rglob("*.pt"))


def test_forward_with_labels_produces_finite_loss() -> None:
    """forward with return_dict=False produces a finite scalar loss."""
    torch.manual_seed(1)
    model = _tiny_model()
    device = next(model.parameters()).device

    B, T = 1, MAX_LEN
    input_ids = torch.randint(0, VOCAB, (B, T), device=device)
    hidden_states = torch.randn(B, T, H, device=device)
    labels = input_ids.clone()

    _logits, loss, metrics = model(
        input_ids=input_ids,
        hidden_states=hidden_states,
        labels=labels,
        step_weights=[0.51, 0.31, 0.18],
        return_dict=False,
    )

    assert loss is not None
    assert loss.isfinite()
    assert "loss_step_0" in metrics
    assert all(v.isfinite() for v in metrics.values())
