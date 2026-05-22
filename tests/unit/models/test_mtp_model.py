"""Unit tests for MTPDraftModel forward pass and resolve_model_type."""

import math

import pytest
import torch

from speculators.models.mtp.model_definitions import (
    mtp_model_classes,
    resolve_model_type,
)

BATCH = 1
SEQ_LEN = 10
NUM_STEPS = 3


@pytest.fixture
def forward_result(mtp_model, seed):
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
    hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
    with torch.no_grad():
        return mtp_model(input_ids=input_ids, hidden_states=hidden_states)


# ===== Forward output structure =====


class TestForwardOutputStructure:
    def test_logits_list_length(self, forward_result):
        logits_list, _, _ = forward_result
        assert len(logits_list) == NUM_STEPS

    @pytest.mark.parametrize("step", range(NUM_STEPS))
    def test_per_step_logit_shapes(self, mtp_model, forward_result, step):
        logits_list, _, _ = forward_result
        vocab_size = mtp_model.config.vocab_size
        expected_len = SEQ_LEN - step - 2
        assert logits_list[step].shape == (BATCH, expected_len, vocab_size)

    def test_loss_is_scalar_finite_nonnegative(self, forward_result):
        _, total_loss, _ = forward_result
        assert total_loss.dim() == 0
        assert torch.isfinite(total_loss)
        assert total_loss >= 0

    def test_metrics_keys(self, forward_result):
        _, _, metrics = forward_result
        expected_keys = {f"loss_step_{k}" for k in range(NUM_STEPS)}
        assert set(metrics.keys()) == expected_keys
        for key in expected_keys:
            assert math.isfinite(metrics[key])


# ===== Loss masking =====


class TestLossMasking:
    def test_zero_mask_ignores_all_targets(self, mtp_model, seed):
        """All-zero loss_mask sets every target to -100. cross_entropy(mean)
        with all-ignored targets returns NaN (0/0), confirming the mask is
        applied to every position."""
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
        assert total_loss.isnan()

    def test_partial_mask_reduces_loss(self, mtp_model, seed):
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


class TestShortSequenceTruncation:
    def test_short_sequence_fewer_logits(self, mtp_model, seed):
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        short_len = 3
        input_ids = torch.randint(0, vocab_size, (BATCH, short_len))
        hidden_states = torch.randn(BATCH, short_len, hidden_size)
        with torch.no_grad():
            logits_list, _, _ = mtp_model(
                input_ids=input_ids, hidden_states=hidden_states
            )
        assert len(logits_list) < NUM_STEPS
        assert len(logits_list) == 1


# ===== resolve_model_type =====


class TestResolveModelType:
    @pytest.mark.parametrize("model_type", list(mtp_model_classes.keys()))
    def test_registered_type_resolves(self, model_type):
        assert resolve_model_type(model_type) == model_type

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported MTP model type"):
            resolve_model_type("nonexistent_model")
