"""Test that MTP verifier weights stay frozen after from_pretrained."""

from pathlib import Path

import torch

from speculators.convert.mtp.converter import MTPConverter
from speculators.models.mtp.core import MTPDraftModel
from tests.conftest import requires_transformers_version


@requires_transformers_version("5.2.0")
def test_mtp_verifier_weights_frozen_after_from_pretrained(tmp_path: Path):
    """HF's from_pretrained resets requires_grad=True on all parameters.
    MTP's load_verifier_weights must re-freeze embed_tokens and lm_head.
    """
    converter = MTPConverter()
    out = tmp_path / "converted"
    converter.convert(
        input_path="Qwen/Qwen3.5-0.8B",
        output_path=str(out),
        base_model="Qwen/Qwen3.5-0.8B",
        num_speculative_steps=3,
        validate=False,
    )

    model = MTPDraftModel.from_pretrained(str(out))
    assert isinstance(model, MTPDraftModel)
    model.train()

    assert not model.embed_tokens.weight.requires_grad
    assert not model.lm_head.weight.requires_grad

    vc = model.config.transformer_layer_config
    input_ids = torch.randint(0, vc.vocab_size, (1, 32))
    hidden_states = torch.randn(1, 32, vc.hidden_size, dtype=torch.bfloat16)
    _, loss, _ = model(input_ids=input_ids, hidden_states=hidden_states)
    loss.backward()

    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
