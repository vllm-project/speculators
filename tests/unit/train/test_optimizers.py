"""Tests for speculators.train.optimizers (BatchedMuon).

BatchedMuon replaces torch.optim.Muon's per-parameter Newton-Schulz loop with
same-shape ``bmm`` batches and ``foreach`` elementwise ops. These tests pin the
invariant that matters: the parameters, momentum buffers, and applied updates
must match the stock optimizer the repo shipped with, across the tall/wide/square
shape mix the draft model produces (tall matrices exercise the transpose
orientation), both LR adjustments, and the fallback path taken for non-plain
tensor subclasses (DTensor under FSDP2 takes it in production).
"""

import pytest
import torch

from speculators.train.optimizers import BatchedMuon, build_optimizers

# Shape mix mirroring the eagle3 draft model's Muon matrices: same-shape groups
# (x2/x3), tall matrices (stored rows > cols, exercising the .T orientation),
# and a lone square matrix.
SHAPES = [
    (2048, 2048),
    (1024, 3072),
    (3072, 1024),
    (3072, 1024),
    (1024, 3072),
    (1024, 2048),
    (1024, 2048),
    (1024, 2048),
]

MUON_KWARGS = {
    "lr": 1e-3, "momentum": 0.95, "weight_decay": 0.1, "ns_steps": 5,
    "adjust_lr_fn": "match_rms_adamw",
}


def _run_steps(optimizer, params, n_steps, seed=0):
    """Assign fresh random grads and step ``n_steps`` times; returns param clones."""
    torch.manual_seed(seed)
    final = []
    for _ in range(n_steps):
        grads = [
            torch.randn(s, device="cuda", dtype=params[i].dtype) * 0.05
            for i, s in enumerate(SHAPES)
        ]
        for p, g in zip(params, grads, strict=True):
            p.grad = g
        optimizer.step()
    torch.cuda.synchronize()
    final = [p.detach().clone() for p in params]
    for p in params:
        p.grad = None
    return final


def _make_pair(dtype):
    stock = [torch.randn(s, device="cuda", dtype=dtype) * 0.02 for s in SHAPES]
    batched = [p.detach().clone() for p in stock]
    for p in (*stock, *batched):
        p.requires_grad_(True)
    opt_stock = torch.optim.Muon(stock, **MUON_KWARGS)
    opt_batched = BatchedMuon(batched, **MUON_KWARGS)
    return stock, batched, opt_stock, opt_batched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_batched_muon_matches_stock_muon(dtype):
    """Batched step applies the same update as torch.optim.Muon.

    The first-step delta isolates the orthogonalized update (momentum has not
    accumulated yet), so a wrong orientation, norm, or LR adjustment shows up
    at full size there; the three-step comparison then covers the momentum
    recursion and the state carried between steps.
    """
    stock, batched, opt_stock, opt_batched = _make_pair(dtype)

    before = [p.detach().clone() for p in stock]
    delta_stock = [
        a - b for a, b in zip(_run_steps(opt_stock, stock, 1), before, strict=True)
    ]
    delta_batched = [
        a - b for a, b in zip(_run_steps(opt_batched, batched, 1), before, strict=True)
    ]
    for ds, db in zip(delta_stock, delta_batched, strict=True):
        torch.testing.assert_close(db.float(), ds.float(), rtol=1e-2, atol=1e-6)

    # three more steps with accumulated momentum on top of the first
    out_stock = _run_steps(opt_stock, stock, 3, seed=1)
    out_batched = _run_steps(opt_batched, batched, 3, seed=1)
    for ps, pb in zip(out_stock, out_batched, strict=True):
        torch.testing.assert_close(pb.float(), ps.float(), rtol=1e-2, atol=1e-5)

    for p in (*stock, *batched):
        buf_s = opt_stock.state[p].get("momentum_buffer")
        buf_b = opt_batched.state[p].get("momentum_buffer")
        if buf_s is not None and buf_b is not None:
            torch.testing.assert_close(
                buf_b.float(), buf_s.float(), rtol=1e-2, atol=1e-5
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_muon_fallback_matches_stock_muon():
    """Non-plain tensor subclasses take the per-parameter path and match stock.

    DTensor (FSDP2) triggers this in production; a tagged subclass stands in
    for it here since FSDP needs a process group.
    """

    class _Tagged(torch.Tensor):
        pass

    stock, batched, opt_stock, opt_batched = _make_pair(torch.bfloat16)
    tagged = [p.detach().clone().as_subclass(_Tagged) for p in batched]
    for p in tagged:
        p.requires_grad_(True)
    opt_tagged = BatchedMuon(tagged, **MUON_KWARGS)

    out_stock = _run_steps(opt_stock, stock, 3)
    out_tagged = _run_steps(opt_tagged, tagged, 3)
    for ps, pt in zip(out_stock, out_tagged, strict=True):
        torch.testing.assert_close(pt.float(), ps.float(), rtol=1e-2, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_muon_matches_stock_muon_original_lr_adjustment():
    """The "original" LR adjustment is shape-asymmetric, so the batched path
    must key it on the stored (not the oriented) shape."""
    kwargs = {**MUON_KWARGS, "adjust_lr_fn": "original"}
    stock, batched = [], []
    for s in SHAPES:
        p = torch.randn(s, device="cuda", dtype=torch.bfloat16) * 0.02
        stock.append(p)
        batched.append(p.detach().clone())
    for p in (*stock, *batched):
        p.requires_grad_(True)
    opt_stock = torch.optim.Muon(stock, **kwargs)
    opt_batched = BatchedMuon(batched, **kwargs)

    out_stock = _run_steps(opt_stock, stock, 1)
    out_batched = _run_steps(opt_batched, batched, 1)
    for ps, pb in zip(out_stock, out_batched, strict=True):
        torch.testing.assert_close(pb.float(), ps.float(), rtol=1e-2, atol=1e-6)


def test_build_optimizers_muon_uses_batched_subclass(tmp_path):
    """The "muon" option wires in BatchedMuon (a torch.optim.Muon, so
    schedulers/checkpointing treat it identically) plus AdamW for the rest."""

    class _Cfg:
        optimizer = "muon"
        lr = 1e-3
        weight_decay = 0.1
        muon_lr = 1e-3  # the schema resolves this to --lr before build_optimizers
        muon_momentum = 0.95
        muon_weight_decay = 0.1
        muon_ns_steps = 5
        muon_adjust_lr_fn = "match_rms_adamw"

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),  # 2D -> Muon
        torch.nn.LayerNorm(32),  # 1D -> AdamW
    )
    optimizers = build_optimizers(model, _Cfg())
    muon = next(o for o in optimizers if isinstance(o, torch.optim.Muon))
    adamw = next(o for o in optimizers if type(o) is torch.optim.AdamW)
    assert type(muon) is BatchedMuon
    assert isinstance(muon, torch.optim.Muon)
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    adamw_ids = {id(p) for g in adamw.param_groups for p in g["params"]}
    all_ids = {id(p) for p in model.parameters()}
    assert muon_ids | adamw_ids == all_ids
    assert not muon_ids & adamw_ids
