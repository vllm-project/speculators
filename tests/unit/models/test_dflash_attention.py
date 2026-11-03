"""Unit tests for the DFlash anchor-block attention mask."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask
from transformers import Qwen3Config

from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel


def _lengths_to_document_ids(lengths, total_seq_len):
    document_ids = torch.full((total_seq_len,), -1, dtype=torch.long)
    document_ids[: lengths.sum()] = torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.long), lengths
    )
    return document_ids


def _reference_dense_from_mask_mod(
    document_ids,
    total_seq_len,
    anchor_positions,
    block_size,
    is_causal,
    sliding_window=None,
):
    """Ground truth: evaluate the flex mask_mod element-wise over the q x kv grid."""
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        is_causal=is_causal,
        sliding_window=sliding_window,
    )
    zero = torch.zeros((), dtype=torch.long)
    ref = torch.zeros(q_len, kv_len, dtype=torch.bool)
    for q in range(q_len):
        for kv in range(kv_len):
            ref[q, kv] = bool(mask_mod(zero, zero, torch.tensor(q), torch.tensor(kv)))
    return ref


def _dense_from_create_mask(
    document_ids,
    total_seq_len,
    anchor_positions,
    block_size,
    is_causal,
    sliding_window=None,
):
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        is_causal=is_causal,
        sliding_window=sliding_window,
    )
    return create_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=document_ids.device,
    )


def test_create_mask_matches_mask_mod_full_attention():
    """Dense mask equals the mask_mod for the full-attention case."""
    device = torch.device("cpu")
    total_seq_len, block_size = 16, 4
    lengths = torch.tensor([10, 6])  # two packed documents summing to total_seq_len
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([3, 8, 12])

    ref = _reference_dense_from_mask_mod(
        document_ids, total_seq_len, anchor_positions, block_size, is_causal=False
    )
    dense = _dense_from_create_mask(
        document_ids.to(device),
        total_seq_len,
        anchor_positions,
        block_size,
        is_causal=False,
    )

    assert dense.shape == (1, 1, ref.shape[0], ref.shape[1])
    assert torch.equal(dense[0, 0].bool(), ref)


def test_create_mask_matches_mask_mod_sliding_window():
    """Dense mask equals the mask_mod when a sliding window is set."""
    device = torch.device("cpu")
    total_seq_len, block_size = 16, 4
    lengths = torch.tensor([16])  # single document
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([5, 9, 14])
    sliding_window = 4

    ref = _reference_dense_from_mask_mod(
        document_ids,
        total_seq_len,
        anchor_positions,
        block_size,
        is_causal=True,
        sliding_window=sliding_window,
    )
    dense = _dense_from_create_mask(
        document_ids.to(device),
        total_seq_len,
        anchor_positions,
        block_size,
        is_causal=True,
        sliding_window=sliding_window,
    )

    assert torch.equal(dense[0, 0].bool(), ref)


def test_create_mask_each_query_sees_its_own_block():
    """Every query must attend to at least its own synthetic block."""
    device = torch.device("cpu")
    total_seq_len, block_size = 12, 4
    lengths = torch.tensor([12])
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([2, 7, 10])

    dense = _dense_from_create_mask(
        document_ids.to(device),
        total_seq_len,
        anchor_positions,
        block_size,
        is_causal=False,
    )

    assert bool(dense[0, 0].any(dim=-1).all())


@pytest.mark.parametrize("is_causal", [None, False, True])
@pytest.mark.parametrize("sliding_window_non_causal", [False, True])
def test_config_causality_controls_mixed_layers(is_causal, sliding_window_non_causal):
    transformer_config = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        is_causal=is_causal,
        sliding_window=4,
        use_sliding_window=True,
        layer_types=["full_attention", "sliding_attention"],
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=16,
        block_size=4,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=0,
        sliding_window_non_causal=sliding_window_non_causal,
    )
    assert transformer_config.is_causal is is_causal
    config = DFlashSpeculatorConfig.from_dict(config.model_dump(exclude_none=True))
    causal = not sliding_window_non_causal if is_causal is None else is_causal
    assert getattr(config.transformer_layer_config, "is_causal", None) is is_causal
    config.transformer_layer_config._attn_implementation = "eager"
    model = DFlashDraftModel(config)
    saved_config = DFlashSpeculatorConfig.from_dict(
        model.config.model_dump(exclude_none=True)
    )
    assert saved_config.sliding_window_non_causal is sliding_window_non_causal
    assert saved_config.transformer_layer_config.is_causal is causal
    loss_mask = torch.zeros(1, 8, dtype=torch.long)
    loss_mask[0, 2] = 1
    full_mask, sliding_mask, anchors, _ = model._build_attention_mask(
        loss_mask=loss_mask,
        max_anchors=1,
        document_ids=torch.zeros(1, 8, dtype=torch.long),
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(anchors, torch.tensor([2]))
    query_positions = 2 + torch.arange(4)
    key_positions = torch.cat((torch.arange(8), query_positions))
    for mask, window in ((full_mask, None), (sliding_mask, 4)):
        expected = torch.cat(
            ((torch.arange(8) < 2).expand(4, -1), torch.ones(4, 4, dtype=torch.bool)),
            dim=1,
        )
        if causal:
            expected &= key_positions[None, :] <= query_positions[:, None]
        if window is not None:
            expected &= (
                key_positions[None, :] - query_positions[:, None]
            ).abs() < window
        torch.testing.assert_close(mask[0, 0] == 0, expected)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("anchor", [2, 7, 12])
@pytest.mark.parametrize("window", [2, 4, 16])
def test_sliding_window_matches_serving(anchor, window, is_causal):
    """The prefix and synthetic keys share the FlashAttention per-query window."""
    total_seq_len, block_size = 20, 8
    document_ids = torch.zeros(total_seq_len, dtype=torch.long)
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=torch.tensor([anchor]),
        block_size=block_size,
        sliding_window=window,
        is_causal=is_causal,
    )
    actual = create_mask(
        mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device="cpu"
    )[0, 0]
    expected = torch.zeros_like(actual)
    for query in range(block_size):
        absolute_query = anchor + query
        for key in range(anchor):
            expected[query, key] = absolute_query - window < key <= absolute_query
        for key in range(block_size):
            allowed = abs(key - query) < window
            if is_causal:
                allowed = allowed and key <= query
            expected[query, total_seq_len + key] = allowed
    torch.testing.assert_close(actual, expected)
