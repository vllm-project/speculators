import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel


def _tiny_model(sample_from_anchor: bool) -> DFlashDraftModel:
    tl_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
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
        sample_from_anchor=sample_from_anchor,
    )
    model = DFlashDraftModel(config)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model.eval()


@pytest.mark.parametrize("max_anchors", [5, 16])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_targets_match_full_sequence_roll(sample_from_anchor, max_anchors):
    torch.manual_seed(0)
    model = _tiny_model(sample_from_anchor)
    seq_len = 32
    verifier_last_hidden_states = torch.randn(1, seq_len, 16)
    anchor_positions = torch.arange(max_anchors)
    offsets = torch.arange(model.block_size)
    block_positions = (anchor_positions[:, None] + offsets[None, :]).reshape(-1)

    with torch.no_grad():
        targets = model._build_targets(
            verifier_last_hidden_states,
            block_positions,
        )

        full_logits = model.verifier_lm_head(
            model.verifier_norm(verifier_last_hidden_states)
        )
        if not sample_from_anchor:
            full_logits = torch.roll(full_logits, 1, dims=1)
        expected = full_logits[:, block_positions]

    torch.testing.assert_close(targets, expected, atol=1e-5, rtol=0)
