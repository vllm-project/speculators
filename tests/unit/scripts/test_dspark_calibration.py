"""Behavior at the offline cache to calibration batch boundary."""

import json
import subprocess
import sys

import pytest
import torch
from datasets import Dataset, load_from_disk
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config

from examples.quantization._dspark_calibration import (
    calibration_batches,
    calibration_forward,
)
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def test_calibration_batches_require_matching_cached_tokens(tmp_path):
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
    path = cache / "hs_0.safetensors"
    save_file(
        {"token_ids": torch.tensor([1, 2, 3, 4, 5, 6]), "hidden_states": hidden}, path
    )

    batches = calibration_batches(
        tmp_path / "records",
        cache,
        count=1,
        max_length=6,
        max_anchors=10,
        layer_ids=[1, 2],
        captured_layer_ids=[1, 2, 3],
        width=2,
        block_size=2,
    )
    batch = next(iter(batches))
    assert batch["hidden_states"].shape == (1, 6, 4)
    assert batch["hidden_states"][0, 0].tolist() == [2, 2, 5, 5]
    assert batch["max_anchors"] == 3

    save_file(
        {"token_ids": torch.tensor([1, 2, 3, 4, 5, 7]), "hidden_states": hidden}, path
    )
    with pytest.raises(ValueError, match="Token ids don't match"):
        next(iter(batches))


def test_prepared_rows_are_usable_for_offline_calibration(tmp_path, monkeypatch):
    """The documented preparation filter excludes rows without valid anchors."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    processor = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    processor.save_pretrained(tmp_path / "tokenizer")
    rows = [
        {"input_ids": list(range(12)), "loss_mask": [0] * 4 + [1] * 8},
        {"input_ids": list(range(12, 24)), "loss_mask": [0] * 3 + [1] * 9},
        # Nine supervised tokens before clipping, but only eight afterward.
        {"input_ids": list(range(13)), "loss_mask": [0] * 4 + [1] * 9},
    ]
    source = tmp_path / "responses.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    subprocess.run(  # noqa: S603 -- fixed command and locally created inputs
        [
            sys.executable,
            "-m",
            "speculators",
            "prepare-data",
            "--model",
            str(tmp_path / "tokenizer"),
            "--data",
            str(source),
            "--output",
            str(tmp_path / "prepared"),
            "--seq-length",
            "12",
            "--max-samples",
            "1",
            "--num-preprocessing-workers",
            "1",
            "--minimum-valid-tokens",
            "9",
        ],
        check=True,
    )
    prepared = load_from_disk(tmp_path / "prepared")
    assert len(prepared) == 1
    assert prepared[0]["input_ids"].tolist() == list(range(12, 24))
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file(
        {
            "token_ids": torch.tensor(list(range(12, 24))),
            "hidden_states": torch.ones((12, 1, 2)),
        },
        cache / "hs_0.safetensors",
    )
    batch = next(
        iter(
            calibration_batches(
                tmp_path / "prepared",
                cache,
                count=1,
                max_length=12,
                max_anchors=32,
                layer_ids=[1],
                captured_layer_ids=[1],
                width=2,
                block_size=8,
            )
        )
    )
    assert batch["max_anchors"] == 1


def test_calibration_activations_match_training():
    """Calibration observers see the same backbone activations as training."""
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
    batch = {
        "hidden_states": torch.randn(1, 12, 32),
        "input_ids": torch.randint(1, 64, (1, 12)),
        "loss_mask": torch.ones(1, 12, dtype=torch.long),
        "document_ids": torch.zeros(1, 12, dtype=torch.long),
        "position_ids": torch.arange(12).unsqueeze(0),
        "max_anchors": 2,
    }
    verifier_states = torch.randn(1, 12, 16)
    with torch.no_grad():
        torch.manual_seed(42)
        expected = model._backbone_forward(
            **batch, verifier_last_hidden_states=verifier_states
        )[0]
        torch.manual_seed(42)
        actual = calibration_forward(model, **batch)
    torch.testing.assert_close(actual, expected)
