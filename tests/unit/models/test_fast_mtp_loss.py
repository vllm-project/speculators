"""Regression tests for FastMTP loss index correctness (commit 13ce4c2).

These tests verify that at step k:
  - Token embedding uses input_ids[:, k : k + valid_len]  (NOT k+1)
  - Target labels use labels[:, k+1 : k+1+valid_len]     (NOT k+2)
  - valid_len = seq_len - k - 1
  - loss_mask zeros are correctly excluded via label=-100
  - step_weights scale losses proportionally
"""

import pytest
import torch
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from speculators.models.fast_mtp import FastMTPSpeculator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_config():
    return Qwen3NextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        vocab_size=100,
        head_dim=32,
        full_attention_interval=1,
    )


@pytest.fixture(scope="module")
def model(tiny_config):
    return FastMTPSpeculator.from_training_args(
        verifier_config=tiny_config,
        num_speculative_steps=3,
        verifier_name_or_path=None,
    )


# ---------------------------------------------------------------------------
# valid_len shrinks by 1 per step
# ---------------------------------------------------------------------------


def test_valid_len_decreases_by_one_per_step(model) -> None:
    """At step k, logits shape[1] == seq_len - k - 1."""
    seq_len = 10
    B, H = 2, 64
    input_ids = torch.randint(0, 100, (B, seq_len))
    hidden_states = torch.randn(B, seq_len, H)

    logits_list, _, _ = model(input_ids=input_ids, hidden_states=hidden_states)

    for k, logits in enumerate(logits_list):
        expected_len = seq_len - k - 1
        assert logits.shape[1] == expected_len, (
            f"Step {k}: expected valid_len={expected_len}, got {logits.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Embedding index — step 0 embeds position 0, not 1
# ---------------------------------------------------------------------------


def test_step0_embeds_correct_position(model) -> None:
    """Step 0 should embed input_ids[:, 0:valid_len], not [:, 1:]."""
    seq_len = 5
    B, H = 1, 64
    # Set up: two different input_ids to detect which position is embedded
    ids_a = torch.zeros(B, seq_len, dtype=torch.long)  # all token 0
    ids_b = ids_a.clone()
    ids_b[:, 0] = 1  # token 1 at position 0

    hs = torch.randn(B, seq_len, H)

    logits_a, _, _ = model(input_ids=ids_a, hidden_states=hs)
    logits_b, _, _ = model(input_ids=ids_b, hidden_states=hs)

    # If step 0 embeds position 0, changing position 0 must change step 0 logits
    assert not torch.allclose(logits_a[0], logits_b[0]), (
        "Changing input_ids[:, 0] must affect step 0 logits "
        "(step 0 should embed position 0)"
    )


def test_step0_does_not_embed_position_seq_len_minus_one(model) -> None:
    """Step 0 only reads positions 0..valid_len-1; the last position is untouched."""
    seq_len = 5
    B, H = 1, 64
    ids_a = torch.zeros(B, seq_len, dtype=torch.long)
    ids_b = ids_a.clone()
    ids_b[:, seq_len - 1] = 99  # change only the last position

    hs = torch.randn(B, seq_len, H)
    logits_a, _, _ = model(input_ids=ids_a, hidden_states=hs)
    logits_b, _, _ = model(input_ids=ids_b, hidden_states=hs)

    # Step 0 valid_len = seq_len - 1 = 4, so embeds positions 0..3.
    # Position 4 (last) should NOT affect step 0 logits.
    assert torch.allclose(logits_a[0], logits_b[0]), (
        "Changing only the last input_ids position must NOT affect step 0 logits"
    )


# ---------------------------------------------------------------------------
# Target label index — step 0 targets position 1, step 1 targets position 2
# ---------------------------------------------------------------------------


def test_step0_loss_changes_with_labels_at_position_1(model) -> None:
    """Step 0 CE loss uses labels[:, 1:1+valid_len] as targets."""
    seq_len = 5
    B, H = 1, 64
    input_ids = torch.randint(0, 100, (B, seq_len))
    hs = torch.randn(B, seq_len, H)

    labels_a = input_ids.clone()
    labels_b = input_ids.clone()
    labels_b[:, 1] = (labels_b[:, 1] + 1) % 100  # change label at position 1

    _, loss_a, _ = model(input_ids=input_ids, hidden_states=hs, labels=labels_a)
    _, loss_b, _ = model(input_ids=input_ids, hidden_states=hs, labels=labels_b)

    assert not torch.allclose(loss_a, loss_b), (
        "Changing labels at position 1 must affect step 0 loss"
    )


# ---------------------------------------------------------------------------
# loss_mask zeros excluded
# ---------------------------------------------------------------------------


def test_loss_mask_zeros_are_excluded(model) -> None:
    """Positions where loss_mask==0 must not contribute to the CE loss."""
    seq_len = 6
    B, H = 1, 64
    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (B, seq_len))
    hs = torch.randn(B, seq_len, H)

    # All-ones mask (all positions included)
    mask_all = torch.ones(B, seq_len, dtype=torch.long)
    _, loss_all, _ = model(input_ids=input_ids, hidden_states=hs, loss_mask=mask_all)

    # Zero out the first half
    mask_half = mask_all.clone()
    mask_half[:, : seq_len // 2] = 0
    _, loss_half, _ = model(input_ids=input_ids, hidden_states=hs, loss_mask=mask_half)

    # Loss values must differ when different positions are masked
    assert not torch.allclose(loss_all, loss_half), (
        "Zeroing half the loss_mask must change the total loss"
    )


def test_all_zeros_loss_mask_produces_nan_or_inf(model) -> None:
    """If every position is masked, CE denominator is 0 → loss is nan/inf."""
    seq_len = 5
    B, H = 1, 64
    input_ids = torch.randint(0, 100, (B, seq_len))
    hs = torch.randn(B, seq_len, H)
    mask_zero = torch.zeros(B, seq_len, dtype=torch.long)
    _, loss, _ = model(input_ids=input_ids, hidden_states=hs, loss_mask=mask_zero)
    # CE with all ignore_index=-100 targets → nan or zero, not a valid training loss
    assert not (torch.isfinite(loss) and loss > 0), (
        "All-zero loss_mask should produce nan/inf/0 loss, not a positive finite value"
    )


# ---------------------------------------------------------------------------
# step_weights scale losses proportionally
# ---------------------------------------------------------------------------


def test_step_weights_scale_loss_correctly(model) -> None:
    """total_loss == sum(loss_step_k); each step loss already includes its weight."""
    seq_len = 8
    B, H = 1, 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (B, seq_len))
    hs = torch.randn(B, seq_len, H)

    weights = [0.6, 0.3, 0.1]
    logits_list, total_loss, metrics = model(
        input_ids=input_ids, hidden_states=hs, step_weights=weights
    )

    expected = sum(metrics[f"loss_step_{i}"] for i in range(3))
    assert torch.allclose(total_loss, expected, atol=1e-5), (
        f"total_loss={total_loss.item():.6f} != sum(step_losses)={expected.item():.6f}"
    )
