"""Chunked candidate selection for DFlash2's selector
(speculators.models.dflash2.chunked_selector).

Everything is checked against the naive, fully-materialized computation:
top-k ids, the differentiable gather-at-given-ids and its gradient (fp64
gradcheck), and the target gather + log-normalizer, so a chunk-size change
can never silently change results.
"""

import torch

from speculators.models.dflash2.chunked_selector import (
    chunked_gather_values,
    chunked_target_gather,
    chunked_topk_ids,
)


def _inputs(rows, hidden, vocab, dtype=torch.float64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn(rows, hidden, generator=gen, dtype=dtype)
    w = torch.randn(vocab, hidden, generator=gen, dtype=dtype)
    return h, w


def test_chunked_topk_ids_matches_naive_topk_indices():
    hidden, weight = _inputs(rows=37, hidden=16, vocab=200)  # deliberately uneven
    top_k, chunk_rows = 5, 8

    naive_ids = (hidden @ weight.T).topk(top_k, dim=-1).indices
    chunked_ids = chunked_topk_ids(hidden, weight, top_k, chunk_rows)

    assert torch.equal(naive_ids, chunked_ids)


def test_chunked_topk_ids_chunk_rows_not_dividing_evenly():
    """The last chunk takes the remainder; rows is not a multiple of chunk_rows."""
    hidden, weight = _inputs(rows=41, hidden=8, vocab=97)
    top_k, chunk_rows = 6, 9  # 41 / 9 -> 4 full chunks + a 5-row remainder

    naive_ids = (hidden @ weight.T).topk(top_k, dim=-1).indices
    chunked_ids = chunked_topk_ids(hidden, weight, top_k, chunk_rows)

    assert torch.equal(naive_ids, chunked_ids)


def test_chunked_topk_ids_default_chunk_rows():
    hidden, weight = _inputs(rows=64, hidden=16, vocab=500)
    top_k = 8

    naive_ids = (hidden @ weight.T).topk(top_k, dim=-1).indices
    chunked_ids = chunked_topk_ids(hidden, weight, top_k, chunk_rows=None)

    assert torch.equal(naive_ids, chunked_ids)


def test_chunked_topk_ids_has_no_gradient():
    hidden, weight = _inputs(rows=10, hidden=8, vocab=50)
    hidden.requires_grad_(True)

    ids = chunked_topk_ids(hidden, weight, top_k=4, chunk_rows=4)
    assert not ids.requires_grad


def test_chunked_gather_values_matches_naive_gather():
    hidden, weight = _inputs(rows=41, hidden=16, vocab=150)
    top_k, chunk_rows = 6, 9

    gen = torch.Generator().manual_seed(2)
    ids = torch.randint(0, weight.shape[0], (41, top_k), generator=gen)

    naive = (hidden @ weight.T).gather(-1, ids)
    chunked = chunked_gather_values(hidden, weight, ids, chunk_rows)

    assert torch.allclose(naive, chunked)


def test_chunked_gather_values_gradient_matches_naive_autograd():
    hidden, weight = _inputs(rows=37, hidden=16, vocab=200)
    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    chunk_rows = 8

    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(0, weight.shape[0], (37, 5), generator=gen)

    naive = (hidden @ weight.T).gather(-1, ids)
    naive.sum().backward()
    naive_d_hidden, naive_d_weight = hidden.grad.clone(), weight.grad.clone()
    hidden.grad, weight.grad = None, None

    chunked = chunked_gather_values(hidden, weight, ids, chunk_rows)
    chunked.sum().backward()

    assert torch.allclose(naive_d_hidden, hidden.grad)
    assert torch.allclose(naive_d_weight, weight.grad)


def test_chunked_gather_values_gradcheck():
    hidden, weight = _inputs(rows=29, hidden=12, vocab=150)
    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    chunk_rows = 9

    gen = torch.Generator().manual_seed(4)
    ids = torch.randint(0, weight.shape[0], (29, 4), generator=gen)

    assert torch.autograd.gradcheck(
        lambda hid, w: chunked_gather_values(hid, w, ids, chunk_rows),
        (hidden, weight),
    )


def test_chunked_gather_values_default_chunk_rows():
    hidden, weight = _inputs(rows=64, hidden=16, vocab=500)
    gen = torch.Generator().manual_seed(5)
    ids = torch.randint(0, weight.shape[0], (64, 8), generator=gen)

    naive = (hidden @ weight.T).gather(-1, ids)
    chunked = chunked_gather_values(hidden, weight, ids, chunk_rows=None)

    assert torch.allclose(naive, chunked)


def test_chunked_target_gather_matches_naive_gather():
    verifier_hidden, verifier_weight = _inputs(rows=41, hidden=20, vocab=150)
    top_k, chunk_rows = 6, 9

    gen = torch.Generator().manual_seed(1)
    candidate_ids = torch.randint(
        0, verifier_weight.shape[0], (41, top_k), generator=gen
    )

    naive_targets = verifier_hidden @ verifier_weight.T
    naive_gathered = naive_targets.gather(-1, candidate_ids)
    naive_lognorm = torch.logsumexp(naive_targets.float(), dim=-1)
    chunked_gathered, chunked_lognorm = chunked_target_gather(
        verifier_hidden, verifier_weight, candidate_ids, chunk_rows
    )

    assert torch.allclose(naive_gathered, chunked_gathered)
    assert torch.allclose(naive_lognorm, chunked_lognorm)


def test_chunked_target_gather_has_no_gradient():
    verifier_hidden, verifier_weight = _inputs(rows=10, hidden=8, vocab=50)
    verifier_hidden.requires_grad_(True)
    candidate_ids = torch.randint(0, 50, (10, 4))

    gathered, lognorm = chunked_target_gather(
        verifier_hidden, verifier_weight, candidate_ids, 4
    )
    assert not gathered.requires_grad
    assert not lognorm.requires_grad


def test_chunked_target_gather_materialize_mode():
    """target_weight=None: targets are already [N, V] logits, as with
    fused_linear_targets="materialize"; only the gather/logsumexp are chunked."""
    gen = torch.Generator().manual_seed(6)
    targets = torch.randn(41, 97, dtype=torch.float64, generator=gen)
    top_k, chunk_rows = 6, 9
    candidate_ids = torch.randint(0, 97, (41, top_k), generator=gen)

    naive_gathered = targets.gather(-1, candidate_ids)
    naive_lognorm = torch.logsumexp(targets.float(), dim=-1)
    chunked_gathered, chunked_lognorm = chunked_target_gather(
        targets, None, candidate_ids, chunk_rows
    )

    assert torch.allclose(naive_gathered, chunked_gathered)
    assert torch.allclose(naive_lognorm, chunked_lognorm)
