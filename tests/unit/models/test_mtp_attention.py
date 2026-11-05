"""MTP attention: the vLLM-aligned TTT mask, built from Eagle3's mask helpers.

Draft step k at anchor t must see the step-0 context at positions <= t in its own
document plus only its own anchor's KV from steps 1..k (vLLM's MTP proposer),
with no attention across packed documents (#621), on every attention backend.
"""

import pytest
import torch
from torch import nn

from speculators.models.mtp import MTPDraftModel

BACKENDS = ["simple_flex_attention", "sdpa", "eager"]


def _model(config, impl):
    config.transformer_layer_config._attn_implementation = impl
    torch.manual_seed(0)
    model = MTPDraftModel(config)
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    return model.eval()


def _packed_inputs(model, lengths, seed=1):
    gen = torch.Generator().manual_seed(seed)
    seq_len = sum(lengths)
    document_ids = torch.cat(
        [torch.full((length,), i) for i, length in enumerate(lengths)]
    ).unsqueeze(0)
    position_ids = torch.cat([torch.arange(length) for length in lengths])
    return {
        "input_ids": torch.randint(
            0, model.config.vocab_size, (1, seq_len), generator=gen
        ),
        "hidden_states": torch.randn(
            1, seq_len, model.config.hidden_size, generator=gen
        ),
        "document_ids": document_ids,
        "position_ids": position_ids.unsqueeze(0),
    }


def test_default_attn_impl_is_flex(mtp_model):
    """Same default as Eagle3; sdpa / eager stay available via --draft-attn-impl."""
    tlc = mtp_model.config.transformer_layer_config
    assert tlc._attn_implementation == "simple_flex_attention"


@pytest.mark.parametrize("impl", BACKENDS)
def test_packed_documents_are_independent(mtp_model_config, impl):
    """Changing the second packed document leaves the first one's drafts unchanged."""
    model = _model(mtp_model_config, impl)
    lengths = [7, 6]
    batch = _packed_inputs(model, lengths)
    changed = {k: v.clone() for k, v in batch.items()}
    changed["input_ids"][0, lengths[0] :] = (
        changed["input_ids"][0, lengths[0] :] + 1
    ) % model.config.vocab_size
    changed["hidden_states"][0, lengths[0] :] = torch.randn(
        lengths[1], model.config.hidden_size
    )
    with torch.no_grad():
        base, _, _ = model(**batch)
        new, _, _ = model(**changed)
    for step in range(model.config.num_speculative_steps):
        keep = lengths[0] - step - 2  # anchors with in-document targets
        assert torch.equal(base[step][:keep], new[step][:keep])


def test_backends_agree(mtp_model_config):
    """flex (block mask), sdpa (bool mask) and eager (float mask) give the same loss."""
    losses = {}
    for impl in BACKENDS:
        model = _model(mtp_model_config, impl)
        batch = _packed_inputs(model, [9, 5, 4])
        with torch.no_grad():
            _, loss, metrics = model(**batch)
        losses[impl] = (loss, metrics["accept_pos_1_sum"])
    ref = losses["eager"]
    for impl in BACKENDS:
        torch.testing.assert_close(losses[impl][0], ref[0], rtol=1e-4, atol=1e-4)
        assert losses[impl][1] == ref[1]


def test_ttt_mask_matches_vllm_mtp_proposer(mtp_model_config):
    """Step k's query at anchor t sees step-0 context <= t in its own document,
    plus anchor t's own KV from steps 1..k, and no other anchor's draft chain."""
    from speculators.models.eagle3.attention import (  # noqa: PLC0415
        create_combined_mask_mod,
    )

    model = _model(mtp_model_config, "sdpa")
    doc = torch.tensor([0, 0, 0, 1, 1, -1])  # two packed documents + padding
    n, steps = doc.numel(), 3
    mask = model._create_mask_fn(
        create_combined_mask_mod(doc, n),
        B=None,
        H=None,
        Q_LEN=n,
        KV_LEN=n,
        device="cpu",
    )
    for _ in range(steps - 1):
        mask = model._extend_attn_mask(mask, n)
    mask = mask[0, 0]
    assert mask.shape == (n, n * steps)

    real = doc != -1
    context = (
        torch.ones(n, n, dtype=torch.bool).tril()
        & (doc[:, None] == doc[None, :])
        & real[:, None]
    )
    eye = torch.eye(n, dtype=torch.bool)
    assert torch.equal(mask[:, :n], context | eye)
    for blk in range(1, steps):
        assert torch.equal(mask[:, blk * n : (blk + 1) * n], eye)
