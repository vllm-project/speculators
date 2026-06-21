"""MTP packed-batch attention must stay document-local (#621).

When several documents are packed into one training row, no token may attend
across a document boundary. MTP builds its mask on the dense
``create_causal_mask`` path, where transformers derives the block-diagonal
packed mask from the per-document ``position_ids`` the collate produces (each
document's positions restart at 0, ``attention_mask`` is None). This test pins
that invariant so it can't silently regress (e.g. if positions stop resetting
or packed-mask detection changes) -- a failure mode where loss still decreases.
"""

import torch

import speculators.models.mtp.core as mtp_core


def test_packed_batch_attention_is_document_local(mtp_model, monkeypatch):
    captured = {}
    real = mtp_core.create_causal_mask

    def spy(*args, **kwargs):
        captured["mask"] = real(*args, **kwargs)
        return captured["mask"]

    monkeypatch.setattr(mtp_core, "create_causal_mask", spy)

    lengths = [1, 3, 2]  # three packed documents
    n = sum(lengths)
    # size the row so valid_len (= seq_len - num_steps - 1) covers all documents
    seq_len = n + mtp_model.config.num_speculative_steps + 1
    # per-document position_ids restart at 0 (as the training collate produces)
    position_ids = torch.zeros(1, seq_len, dtype=torch.long)
    position_ids[0, :n] = torch.cat([torch.arange(length) for length in lengths])
    document_ids = torch.cat(
        [torch.full((length,), i) for i, length in enumerate(lengths)]
    )

    with torch.no_grad():
        mtp_model(
            input_ids=torch.randint(0, mtp_model.config.vocab_size, (1, seq_len)),
            hidden_states=torch.randn(1, seq_len, mtp_model.config.hidden_size),
            position_ids=position_ids,
        )

    allow = captured["mask"][0, 0] == 0  # additive mask -> "may attend"
    n_q = allow.shape[0]  # == valid_len == n
    idx = torch.arange(n_q)
    doc = document_ids[:n_q]
    expected = (idx[:, None] >= idx[None, :]) & (doc[:, None] == doc[None, :])
    assert torch.equal(allow, expected)
