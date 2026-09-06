"""Unit tests for MTPDraftModel forward pass."""

import math

import pytest
import torch

BATCH = 1
SEQ_LEN = 10


# ===== Forward output structure =====


def test_forward_output_structure(mtp_model, seed):
    """Verify logit shapes, loss, and per-step metrics in a single forward pass."""
    num_steps = mtp_model.config.num_speculative_steps
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
    hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
    with torch.no_grad():
        logits_list, total_loss, metrics = mtp_model(
            input_ids=input_ids, hidden_states=hidden_states
        )

    assert len(logits_list) == num_steps
    expected_len = SEQ_LEN - num_steps - 1
    for step in range(num_steps):
        assert logits_list[step].shape == (BATCH, expected_len, vocab_size)

    assert total_loss.dim() == 0
    assert torch.isfinite(total_loss)
    assert total_loss >= 0

    expected_keys = {f"loss_step_{k}" for k in range(num_steps)} | {
        "loss_sum",
        "loss_total",
    }
    expected_keys |= {
        f"reference_prefix_acc_{i}_{kind}"
        for i in range(1, num_steps + 1)
        for kind in ("sum", "total")
    }
    assert set(metrics.keys()) == expected_keys
    for key in expected_keys:
        assert math.isfinite(metrics[key])

    # MTP's first prediction is two tokens after the initial hidden state.
    prefix_matches = torch.ones_like(input_ids[:, :expected_len], dtype=torch.bool)
    for step, logits in enumerate(logits_list):
        prefix_matches &= logits.argmax(-1).eq(
            input_ids[:, step + 2 : step + 2 + expected_len]
        )
        assert metrics[f"reference_prefix_acc_{step + 1}_sum"] == prefix_matches.sum()
        assert metrics[f"reference_prefix_acc_{step + 1}_total"] == expected_len


# ===== Loss masking =====


class TestLossMasking:
    def test_zero_mask_ignores_all_targets(self, mtp_model, seed):
        """All-zero loss_mask sets every target to -100. Loss returns 0.0
        (not NaN) because the denominator is clamped to min=1."""
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        loss_mask = torch.zeros(BATCH, SEQ_LEN)
        with torch.no_grad():
            _, total_loss, metrics = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )
        assert total_loss == 0.0
        assert all(metrics[f"reference_prefix_acc_{i}_total"] == 0 for i in range(1, 4))

    def test_partial_mask_changes_loss(self, mtp_model, seed):
        """Masking some positions should change the loss vs no mask."""
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        with torch.no_grad():
            _, loss_no_mask, _ = mtp_model(
                input_ids=input_ids, hidden_states=hidden_states
            )
            mask = torch.ones(BATCH, SEQ_LEN)
            mask[:, -3:] = 0
            _, loss_partial_mask, _ = mtp_model(
                input_ids=input_ids, hidden_states=hidden_states, loss_mask=mask
            )
        assert loss_no_mask != loss_partial_mask


# ===== Step weights =====


class TestStepWeights:
    def test_zero_weight_zeroes_step_loss(self, mtp_model, seed):
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        with torch.no_grad():
            _, _, metrics = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                step_weights=[1.0, 0.0, 0.0],
            )
        assert metrics["loss_step_0"] > 0
        assert metrics["loss_step_1"] == 0.0
        assert metrics["loss_step_2"] == 0.0


# ===== Short sequence truncation =====


def test_mtp_references_exclude_hidden_state_from_previous_document(mtp_model):
    # Position 4 must not borrow the next document's reference tokens.
    inputs = torch.zeros(1, 12, dtype=torch.long)
    _, _, metrics = mtp_model(
        input_ids=inputs,
        hidden_states=torch.randn(1, 12, mtp_model.config.hidden_size),
        document_ids=torch.tensor([[0] * 5 + [1] * 5 + [-1] * 2]),
    )
    assert [metrics[f"reference_prefix_acc_{i}_total"] for i in range(1, 4)] == [
        6,
        4,
        2,
    ]


@pytest.mark.parametrize("seq_len", [0, 1, 2, 3])
def test_short_mtp_sequence_counts_only_observed_positions(mtp_model, seq_len):
    logits, _, metrics = mtp_model(
        input_ids=torch.zeros(1, seq_len, dtype=torch.long),
        hidden_states=torch.zeros(1, seq_len, mtp_model.config.hidden_size),
    )
    assert len(logits) == max(0, seq_len - 2)
    assert metrics["reference_prefix_acc_1_total"] == int(seq_len == 3)
    assert (
        metrics["reference_prefix_acc_2_total"]
        == metrics["reference_prefix_acc_3_total"]
        == 0
    )
