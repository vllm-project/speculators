"""Unit tests for MTPDraftModel forward pass."""

import math

import pytest
import torch
from safetensors import safe_open

import speculators.models.mtp.core as mtp_core
from speculators.models.mtp.core import _prepare_mtp_position_ids, _ttt_mask_mod

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


@pytest.mark.parametrize("train_forward", ["fastmtp", "vllm"])
def test_forward_output_structure(mtp_model, seed, train_forward):
    """Verify draft-token shapes, loss, and per-step metrics in one forward pass."""
    num_steps = mtp_model.config.num_speculative_steps
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
    hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
    with torch.no_grad():
        draft_tokens, total_loss, metrics = mtp_model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            train_forward=train_forward,
        )

    assert len(draft_tokens) == num_steps
    expected_len = SEQ_LEN - num_steps - 1
    for step in range(num_steps):
        # greedy predictions at the supervised positions (all, without a mask)
        assert draft_tokens[step].shape == (BATCH * expected_len,)

    assert total_loss.dim() == 0
    assert torch.isfinite(total_loss)
    assert total_loss >= 0

    expected_keys = {f"loss_step_{k}" for k in range(num_steps)} | {
        "loss_sum",
        "loss_total",
    }
    for k in range(1, num_steps + 1):
        expected_keys |= {
            f"accept_pos_{k}_sum",
            f"accept_pos_{k}_total",
            f"step_{k}_acc_sum",
            f"step_{k}_acc_total",
        }
    assert set(metrics.keys()) == expected_keys
    for key in expected_keys:
        assert math.isfinite(metrics[key])
    # accept at position k requires steps 1..k all correct
    for k in range(1, num_steps):
        assert metrics[f"accept_pos_{k + 1}_sum"] <= metrics[f"accept_pos_{k}_sum"]


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


@pytest.mark.parametrize("train_forward", ["fastmtp", "vllm"])
def test_short_sequence_fewer_steps(mtp_model, seed, train_forward):
    num_steps = mtp_model.config.num_speculative_steps
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    short_len = 3
    input_ids = torch.randint(0, vocab_size, (BATCH, short_len))
    hidden_states = torch.randn(BATCH, short_len, hidden_size)
    with torch.no_grad():
        draft_tokens, _, _ = mtp_model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            train_forward=train_forward,
        )
    assert len(draft_tokens) < num_steps
    assert len(draft_tokens) == 1


def test_unknown_train_forward_raises(mtp_model):
    with pytest.raises(ValueError, match="train_forward"):
        mtp_model(
            input_ids=torch.zeros(BATCH, SEQ_LEN, dtype=torch.long),
            hidden_states=torch.zeros(BATCH, SEQ_LEN, mtp_model.config.hidden_size),
            train_forward="eagle",
        )


# ===== vLLM-aligned (TTT) forward =====


def test_ttt_mask_matches_vllm_mtp_proposer():
    """Step k's query at anchor t sees step-0 context <= t in its own document,
    plus anchor t's own KV from steps 1..k, and no other anchor's draft chain."""
    doc = torch.tensor([1, 1, 1, 2, 2])  # two packed documents
    n, step = doc.numel(), 2
    q = torch.arange(n)[:, None]
    kv = torch.arange(n * (step + 1))[None, :]
    mask = _ttt_mask_mod(doc, n)(0, 0, q, kv)

    same_doc = doc[:, None] == doc[None, :]
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    assert torch.equal(mask[:, :n], causal & same_doc)
    for blk in range(1, step + 1):
        assert torch.equal(mask[:, blk * n : (blk + 1) * n], torch.eye(n).bool())


def test_ttt_forward_is_document_local(mtp_model, seed):
    """Changing one packed document must not change the other's predictions."""
    num_steps = mtp_model.config.num_speculative_steps
    first, second = 4, 4
    n = first + second
    seq_len = n + num_steps + 1
    position_ids = torch.zeros(1, seq_len, dtype=torch.long)
    position_ids[0, :n] = torch.cat([torch.arange(first), torch.arange(second)])
    input_ids = torch.randint(0, mtp_model.config.vocab_size, (1, seq_len))
    hidden_states = torch.randn(1, seq_len, mtp_model.config.hidden_size)

    def run(ids, hs):
        with torch.no_grad():
            tokens, _, _ = mtp_model(
                input_ids=ids,
                hidden_states=hs,
                position_ids=position_ids,
                train_forward="vllm",
            )
        return tokens

    base = run(input_ids, hidden_states)
    ids2, hs2 = input_ids.clone(), hidden_states.clone()
    ids2[0, first + 2 :] = (ids2[0, first + 2 :] + 1) % mtp_model.config.vocab_size
    hs2[0, first:] = torch.randn_like(hs2[0, first:])
    changed = run(ids2, hs2)
    # anchors whose inputs and targets stay inside the first document
    for step in range(num_steps):
        keep = first - step - 1
        assert torch.equal(base[step][:keep], changed[step][:keep])


def test_ttt_forward_applies_frozen_verifier_norm(mtp_model, seed):
    """The vLLM path normalizes hidden states with the verifier's final norm."""
    assert not mtp_model.verifier_norm.weight.requires_grad
    input_ids = torch.randint(0, mtp_model.config.vocab_size, (BATCH, SEQ_LEN))
    hidden_states = torch.randn(BATCH, SEQ_LEN, mtp_model.config.hidden_size)

    def losses():
        with torch.no_grad():
            return [
                mtp_model(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    train_forward=mode,
                )[1]
                for mode in ("fastmtp", "vllm")
            ]

    fastmtp_before, vllm_before = losses()
    # non-uniform: the MTP layer re-normalizes its hidden input, which would
    # cancel a uniform gain
    with torch.no_grad():
        mtp_model.verifier_norm.weight.copy_(
            torch.randn_like(mtp_model.verifier_norm.weight)
        )
    fastmtp_after, vllm_after = losses()
    assert fastmtp_after == fastmtp_before
    assert vllm_after != vllm_before


def test_verifier_norm_is_not_saved(mtp_model, tmp_path):
    mtp_model.save_pretrained(tmp_path)
    keys = set()
    for path in tmp_path.glob("*.safetensors"):
        with safe_open(path, framework="pt") as f:
            keys |= set(f.keys())
    assert keys
    assert "verifier_norm.weight" not in keys


# ===== Chunked LM head + CE =====


def test_head_ce_chunking_matches_single_chunk(mtp_model, seed, monkeypatch):
    """Chunked, checkpointed LM head + CE gives the same loss, predictions and
    grads as one unchunked pass."""
    num_positions = 7
    hidden = torch.randn(num_positions, mtp_model.config.hidden_size)
    targets = torch.randint(0, mtp_model.config.vocab_size, (num_positions,))

    def run(chunk):
        monkeypatch.setattr(mtp_core, "_HEAD_CE_CHUNK", chunk)
        h = hidden.clone().requires_grad_()
        mtp_model.zero_grad()
        loss, pred = mtp_model._head_ce_chunked(h, targets)
        loss.sum().backward()
        return loss.detach(), pred, h.grad, mtp_model.lm_head.weight.grad.clone()

    mtp_model.lm_head.weight.requires_grad_(True)
    full = run(num_positions)
    chunked = run(3)  # uneven last chunk
    for a, b in zip(chunked, full, strict=True):
        torch.testing.assert_close(a, b)
