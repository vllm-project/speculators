"""Unit tests for the P-EAGLE attention mask."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask
from transformers import LlamaConfig

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.peagle.attention import create_peagle_mask_mod
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.models.peagle.core import PEagleDraftModel
from speculators.proposals import GreedyTokenProposalConfig


def _dense_mask(anchor_pos, depth, document_ids):
    mask_mod = create_peagle_mask_mod(
        anchor_pos=anchor_pos, depth=depth, document_ids=document_ids
    )
    n = anchor_pos.numel()
    return create_mask(mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device="cpu")[0, 0]


def test_padding_rows_attend_only_to_themselves():
    # Docstring example: 6 positions, depths 0-3
    anchor_pos = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 3, 4, 0, 3, 0])
    depth = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3])
    unpadded = _dense_mask(anchor_pos, depth, torch.zeros(6, dtype=torch.long))

    # The same example packed with 2 padding positions, which depth 0 samples
    # (rows 6 and 7) and later depths do not
    padded = _dense_mask(
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 3, 4, 0, 3, 0]),
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3]),
        torch.tensor([0, 0, 0, 0, 0, 0, -1, -1]),
    )
    padding = torch.zeros(15, dtype=torch.bool)
    padding[6:8] = True

    assert padded.any(dim=-1).all()
    assert torch.equal(padded[padding], torch.eye(15, dtype=torch.bool)[padding])
    assert not padded[~padding][:, padding].any()
    assert torch.equal(padded[~padding][:, ~padding], unpadded)


def _build_model(attn_impl):
    layer_config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    layer_config._attn_implementation = attn_impl
    config = PEagleSpeculatorConfig(
        transformer_layer_config=layer_config,
        draft_vocab_size=128,
        mask_token_id=5,
        eagle_aux_hidden_state_layer_ids=[0, 1, 2],
        speculators_config=SpeculatorsConfig(
            algorithm="peagle",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=4)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=["X"]),
        ),
    )
    torch.manual_seed(0)
    model = PEagleDraftModel(config)
    for weight in (
        model.embed_tokens.weight,
        model.lm_head.weight,
        model.verifier_lm_head.weight,
    ):
        torch.nn.init.normal_(weight)
    return model


@pytest.mark.parametrize("num_padding", [0, 4])
def test_eager_loss_matches_sdpa_with_padding(num_padding):
    seq_len = 24
    document_ids = torch.cat(
        [
            torch.zeros(seq_len - num_padding, dtype=torch.long),
            torch.full((num_padding,), -1),
        ]
    ).unsqueeze(0)
    generator = torch.Generator().manual_seed(1)
    inputs = {
        "hidden_states": torch.randn(1, seq_len, 192, generator=generator),
        "input_ids": torch.randint(0, 128, (1, seq_len), generator=generator),
        "verifier_last_hidden_states": torch.randn(1, seq_len, 64, generator=generator),
        "document_ids": document_ids,
        "position_ids": torch.arange(seq_len).unsqueeze(0),
        "loss_mask": (document_ids >= 0) & (torch.arange(seq_len) > 0),
        "loss_config": resolve_loss_config("kl_div", "eager"),
        "num_depths": 4,
    }

    losses = {}
    for attn_impl in ("eager", "sdpa"):
        model = _build_model(attn_impl)
        torch.manual_seed(2)  # the same COD samples for both models
        with torch.no_grad():
            _, losses[attn_impl], _ = model(**inputs)

    assert torch.isfinite(losses["eager"])
    torch.testing.assert_close(losses["eager"], losses["sdpa"])
