"""Unit tests for the DFlash anchor-block attention mask."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask
from transformers import Qwen3Config

from speculators import VerifierConfig
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod


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


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("anchor", [2, 7, 12])
@pytest.mark.parametrize("window", [8, 16])
@pytest.mark.parametrize("lengths", [[20], [10, 10]])
def test_sliding_window_matches_serving(anchor, window, is_causal, lengths):
    """The prefix and synthetic keys share the FlashAttention per-query window."""
    total_seq_len, block_size = 20, 8
    document_ids = _lengths_to_document_ids(torch.tensor(lengths), total_seq_len)
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
            expected[query, key] = (
                absolute_query - window < key <= absolute_query
                and document_ids[key] == document_ids[anchor]
            )
        for key in range(block_size):
            allowed = abs(key - query) < window
            if is_causal:
                allowed = allowed and key <= query
            expected[query, total_seq_len + key] = allowed
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("is_causal", [None, False, True])
@pytest.mark.parametrize("non_causal", [False, True])
def test_saved_causality_preserves_legacy_layer_masks(is_causal, non_causal):
    """Legacy mixed layers retain their masks; explicit causality overrides both."""
    decoder = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
    )
    decoder.is_causal = is_causal
    config = DFlashSpeculatorConfig(
        transformer_layer_config=decoder,
        draft_vocab_size=16,
        aux_hidden_state_layer_ids=[0, 1],
        block_size=4,
        mask_token_id=0,
        sliding_window_non_causal=non_causal,
    )
    for _ in range(2):
        config.transformer_layer_config._attn_implementation = "eager"
        model = DFlashDraftModel(config)
        for window in (None, 4):
            mask = model._create_attention_mask(
                document_ids=torch.zeros(1, 12, dtype=torch.long),
                total_seq_len=12,
                anchor_positions=torch.tensor([4]),
                device=torch.device("cpu"),
                sliding_window=window,
            )
            expected_causal = (
                window is not None and not non_causal
                if is_causal is None
                else is_causal
            )
            assert bool(mask[0, 0, 0, 13] == 0) is not expected_causal
        config = DFlashSpeculatorConfig.model_validate(
            config.model_dump(exclude_none=True)
        )


@pytest.mark.parametrize("is_causal", [False, True])
def test_training_flag_is_saved_in_decoder_config(is_causal, monkeypatch):
    """The direct training option persists the flag consumed by vLLM."""
    monkeypatch.setattr(
        "speculators.models.dflash.core.resolve_target_layer_ids",
        lambda *a, **k: [0, 1],
    )
    monkeypatch.setattr(
        VerifierConfig,
        "from_pretrained",
        lambda *a, **k: VerifierConfig(name_or_path=None, architectures=[]),
    )
    decoder = Qwen3Config()
    decoder.is_causal = not is_causal
    kwargs = DFlashDraftModel._build_base_config_kwargs(
        "dflash",
        decoder,
        verifier_name_or_path="target",
        draft_vocab_size=16,
        causal_attention=is_causal,
    )
    saved = DFlashSpeculatorConfig(**kwargs).model_dump(exclude_none=True)
    assert saved["transformer_layer_config"]["is_causal"] is is_causal
