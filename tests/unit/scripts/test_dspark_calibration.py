"""Behavior at the offline cache to calibration batch boundary."""

import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import Qwen3Config

from examples.quantization.dspark_calibration import (
    calibration_batches,
    calibration_forward,
)
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _calibration_batches(tmp_path, token_ids):
    records = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6]],
            "loss_mask": [[0, 1, 1, 1, 1, 1]],
            "seq_len": [6],
        }
    ).with_format("torch")
    records.save_to_disk(tmp_path / "records")
    cache = tmp_path / "cache"
    cache.mkdir()
    hidden = torch.tensor([[[2.0, 2.0], [5.0, 5.0], [99.0, 99.0]]] * 6)
    save_file(
        {"token_ids": torch.tensor(token_ids), "hidden_states": hidden},
        cache / "hs_0.safetensors",
    )
    return calibration_batches(
        tmp_path / "records",
        cache,
        count=1,
        max_length=6,
        max_anchors=10,
        layer_ids=[1, 2],
        width=2,
        block_size=2,
    )


def test_calibration_batches_align_cached_tokens(tmp_path):
    batches = _calibration_batches(tmp_path, [1, 2, 3, 4, 5, 6])

    batch = next(iter(batches))

    assert batch["hidden_states"].shape == (1, 6, 4)
    assert batch["hidden_states"][0, 0].tolist() == [2, 2, 5, 5]
    assert batch["max_anchors"] == 3


def test_calibration_batches_require_matching_cached_tokens(tmp_path):
    batches = _calibration_batches(tmp_path, [1, 2, 3, 4, 5, 7])

    with pytest.raises(ValueError, match="Token ids don't match"):
        next(iter(batches))


@pytest.mark.parametrize(
    ("hidden", "message"),
    [
        (torch.ones(6, 4, 2), "Invalid hidden-state shape"),
        (torch.full((6, 3, 2), float("nan")), "non-finite"),
    ],
)
def test_calibration_rejects_invalid_hidden_states(tmp_path, hidden, message):
    Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4, 5, 6]], "loss_mask": [[0, 1, 1, 1, 1, 1]]}
    ).save_to_disk(tmp_path / "records")
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file(
        {"token_ids": torch.tensor([1, 2, 3, 4, 5, 6]), "hidden_states": hidden},
        cache / "hs_0.safetensors",
    )
    batches = calibration_batches(
        tmp_path / "records",
        cache,
        count=1,
        max_length=6,
        max_anchors=10,
        layer_ids=[1, 2],
        width=2,
        block_size=2,
    )
    with pytest.raises(ValueError, match=message):
        next(iter(batches))


def _dspark_model():
    config = DSparkSpeculatorConfig(
        transformer_layer_config=Qwen3Config(  # type: ignore[call-arg]
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            _attn_implementation="eager",
        ),
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
    )
    model = DSparkDraftModel(config).eval()
    for weight in (
        model.embed_tokens.weight,
        model.lm_head.weight,
        model.verifier_lm_head.weight,
    ):
        torch.nn.init.normal_(weight)
    return model


def _calibration_batch():
    return {
        "hidden_states": torch.randn(1, 12, 32),
        "input_ids": torch.randint(1, 64, (1, 12)),
        "loss_mask": torch.ones(1, 12, dtype=torch.long),
        "document_ids": torch.zeros(1, 12, dtype=torch.long),
        "position_ids": torch.arange(12).unsqueeze(0),
        "max_anchors": 2,
    }


def test_calibration_activations_match_training():
    """Calibration observers see the same backbone activations as training."""
    model = _dspark_model()
    batch = _calibration_batch()
    verifier_states = torch.randn(1, 12, 16)
    with torch.no_grad():
        torch.manual_seed(42)
        expected = model._backbone_forward(
            **batch, verifier_last_hidden_states=verifier_states
        )[0]
        torch.manual_seed(42)
        actual = calibration_forward(model, **batch)
    torch.testing.assert_close(actual, expected)


def test_calibration_forward_skips_output_heads_and_verifier_norm():
    model = _dspark_model()

    def fail(*_args, **_kwargs):
        raise AssertionError("Calibration must not compute verifier outputs")

    model.lm_head.forward = fail
    model.verifier_lm_head.forward = fail
    model.verifier_norm.forward = fail

    with torch.no_grad():
        calibration_forward(model, **_calibration_batch())
