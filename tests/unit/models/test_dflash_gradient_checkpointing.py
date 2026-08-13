import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel


def _tiny_model() -> DFlashDraftModel:
    tl_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
    )
    model = DFlashDraftModel(config)
    # These three are left uninitialized by the constructor (they are normally loaded
    # from the verifier), so they hold garbage that would poison the forward.
    for weight in (
        model.embed_tokens.weight,
        model.lm_head.weight,
        model.verifier_lm_head.weight,
    ):
        torch.nn.init.normal_(weight, std=0.02)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model


def _backbone_grads(model: DFlashDraftModel) -> dict[str, torch.Tensor]:
    """Run one backbone forward/backward and return the draft-layer gradients."""
    torch.manual_seed(0)
    seq_len = 32
    hidden, *_ = model._backbone_forward(
        hidden_states=torch.randn(1, seq_len, 2 * 16),
        input_ids=torch.randint(0, 64, (1, seq_len)),
        loss_mask=torch.ones(1, seq_len),
        verifier_last_hidden_states=torch.randn(1, seq_len, 16),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long),
        max_anchors=8,
    )
    model.zero_grad(set_to_none=True)
    hidden.pow(2).sum().backward()
    return {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if name.startswith("layers.") and param.grad is not None
    }


@pytest.mark.parametrize("model_cls", [DFlashDraftModel, DSparkDraftModel])
def test_model_declares_gradient_checkpointing_support(model_cls):
    assert model_cls.supports_gradient_checkpointing is True


def test_enable_sets_the_flag_on_every_draft_layer():
    model = _tiny_model()
    assert all(not layer.gradient_checkpointing for layer in model.layers)

    model.gradient_checkpointing_enable({"use_reentrant": False})

    assert all(layer.gradient_checkpointing for layer in model.layers)


def test_gradients_match_without_checkpointing():
    """Checkpointing must only change *when* activations exist, not the gradients.

    This is the regression guard for `use_reentrant`: the draft layers take every
    input as a keyword argument, so the reentrant implementation would receive no
    tensor inputs and silently produce no gradients at all.
    """
    model = _tiny_model()
    model.train()

    expected = _backbone_grads(model)
    model.gradient_checkpointing_enable({"use_reentrant": False})
    actual = _backbone_grads(model)

    assert actual.keys() == expected.keys()
    assert expected  # guard against the comparison passing on an empty dict
    for name, grad in expected.items():
        torch.testing.assert_close(actual[name], grad)
