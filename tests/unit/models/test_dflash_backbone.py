"""Backbone execution and trainer integration independently of targets."""

from unittest.mock import Mock

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash2.model_definitions import Qwen3DFlash2DecoderLayer
from speculators.train import distributed
from tests.unit.models.test_dflash_family_contracts import RECIPES, make_contract_model


@pytest.mark.parametrize("recipe", RECIPES)
def test_target_free_draft_features_and_decoder_access(recipe, monkeypatch):
    model = make_contract_model(recipe, False)
    SpeculatorModel.verify_training_compatible(model)
    # Accessors must not register a second copy under the old flat path.
    assert "layers" not in model._modules
    assert model.get_decoder_layers() is model.backbone.layers
    if recipe == "dflash2":
        assert isinstance(model.get_decoder_layers()[0], Qwen3DFlash2DecoderLayer)
    verifier = Mock(side_effect=AssertionError("verifier head used during drafting"))
    monkeypatch.setattr(model.verifier_lm_head, "forward", verifier)
    features = model.draft_features(
        torch.randn(1, 8, 32),
        torch.arange(8).view(1, -1),
        torch.ones(1, 8),
        torch.zeros(1, 8, dtype=torch.long),
        max_anchors=2,
    )
    assert features.hidden.shape == (1, 8, 16)
    assert features.logits.shape == (1, 8, 32)
    verifier.assert_not_called()
    shard = Mock()
    monkeypatch.setattr(distributed, "fully_shard", shard)
    distributed.apply_fully_sharded(model)
    assert [call.args[0] for call in shard.call_args_list] == [
        *model.get_decoder_layers(),
        model,
    ]


@pytest.mark.parametrize("recipe", ["dflash", "dflash2"])
def test_nested_layers_support_gradient_checkpointing(recipe):
    model = make_contract_model(recipe, False)
    model.gradient_checkpointing_enable()
    assert all(layer.gradient_checkpointing for layer in model.get_decoder_layers())


@pytest.mark.parametrize("recipe", ["dflash", "dflash2"])
def test_compiled_features_match_eager_with_dynamic_anchor_counts(recipe):
    model = make_contract_model(recipe, False)
    compiled = torch.compile(model.draft_features, backend="eager")
    hidden = torch.randn(1, 12, 32)
    tokens = torch.arange(12).view(1, -1)
    documents = torch.zeros(1, 12, dtype=torch.long)
    for valid_count in (8, 1, 0):
        mask = (torch.arange(12) < valid_count).float().unsqueeze(0)
        torch.manual_seed(42)
        expected = model.draft_features(hidden, tokens, mask, documents, max_anchors=3)
        torch.manual_seed(42)
        actual = compiled(hidden, tokens, mask, documents, max_anchors=3)
        torch.testing.assert_close(actual.hidden, expected.hidden)
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(actual.layout.loss_mask, expected.layout.loss_mask)


@pytest.mark.parametrize("seq_len", [1, 2, 4])
def test_short_sequence_padding_has_zero_finite_loss(seq_len):
    model = make_contract_model("dflash", False)
    _, loss, _ = model(
        torch.randn(1, seq_len, 32),
        torch.arange(seq_len).view(1, -1),
        torch.ones(1, seq_len),
        torch.randn(1, seq_len, 16),
        torch.zeros(1, seq_len, dtype=torch.long),
        max_anchors=2,
        loss_config=resolve_loss_config("kl_div", "eager"),
    )
    assert loss.item() == 0
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
