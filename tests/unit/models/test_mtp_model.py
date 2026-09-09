"""Unit tests for MTPDraftModel forward pass."""

import math

import torch

from speculators.models.mtp.core import _prepare_mtp_position_ids

BATCH = 1
SEQ_LEN = 10


def test_qwen35_position_ids_are_expanded_for_new_transformers():
    position_ids = torch.arange(8).unsqueeze(0)

    text_position_ids, rotary_position_ids = _prepare_mtp_position_ids(
        position_ids, valid_len=6, model_type="qwen3_5_text"
    )

    assert text_position_ids.shape == (1, 6)
    assert rotary_position_ids.shape == (3, 1, 6)
    assert torch.equal(rotary_position_ids[0], text_position_ids)
    assert torch.equal(rotary_position_ids[1], text_position_ids)
    assert torch.equal(rotary_position_ids[2], text_position_ids)


def test_qwen35_position_ids_accept_existing_mrope_layout():
    position_ids = torch.arange(24).reshape(3, 2, 4)

    text_position_ids, rotary_position_ids = _prepare_mtp_position_ids(
        position_ids, valid_len=3, model_type="qwen3_5_moe_text"
    )

    assert torch.equal(text_position_ids, position_ids[0, :, :3])
    assert torch.equal(rotary_position_ids, position_ids[:, :, :3])


def test_non_qwen35_position_ids_remain_2d():
    position_ids = torch.arange(8).unsqueeze(0)

    text_position_ids, rotary_position_ids = _prepare_mtp_position_ids(
        position_ids, valid_len=6, model_type="qwen3"
    )

    assert torch.equal(text_position_ids, position_ids[:, :6])
    assert torch.equal(rotary_position_ids, position_ids[:, :6])


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
    assert set(metrics.keys()) == expected_keys
    for key in expected_keys:
        assert math.isfinite(metrics[key])


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
            _, total_loss, _ = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )
        assert total_loss == 0.0

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


def test_short_sequence_fewer_logits(mtp_model, seed):
    num_steps = mtp_model.config.num_speculative_steps
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    short_len = 3
    input_ids = torch.randint(0, vocab_size, (BATCH, short_len))
    hidden_states = torch.randn(BATCH, short_len, hidden_size)
    with torch.no_grad():
        logits_list, _, _ = mtp_model(input_ids=input_ids, hidden_states=hidden_states)
    assert len(logits_list) < num_steps
    assert len(logits_list) == 1
