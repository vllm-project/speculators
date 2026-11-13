"""The standalone loss head (``speculators.models.loss_head``) and its golden
fixtures, which pin the fp32 eager loss-head semantics.

Each golden case runs the DFlash or DSpark loss head (``lm_head`` projection, Markov
bias for DSpark, verifier projection, the model's ``compute_metrics``) with
``loss_implementation=eager`` in fp32 on CPU, and pins a fingerprint of the
result in ``data/loss_head_golden.json``: the loss, every scalar metric, and
for each input gradient its norm and its projection onto a fixed random probe.

- ``test_eager_fp32_matches_golden`` fails if the eager reference itself
  drifts, so later loss refactors cannot silently change the semantics that
  the fused path is measured against.
- ``test_fused_fp32_matches_eager_reference`` checks today's fused path at the
  head level (D-PACE, the DSpark Markov bias and composite losses included).
- ``test_fused_bf16_error_vs_eager_reference`` bounds the fused path's bf16
  error against the fp32 reference.

Regenerate after an intentional semantic change with:

    SPECULATORS_REGENERATE_GOLDEN=1 pytest tests/unit/models/test_loss_head.py \
        -k regenerate
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from speculators.models.loss_head import (
    HeadConfig,
    HeadInputs,
    loss_head,
    make_inputs,
)
from tests.conftest import requires_cuda

GOLDEN_PATH = Path(__file__).parent / "data" / "loss_head_golden.json"

_SMALL = {"rows": 48, "vocab": 1000, "hidden": 64, "block_size": 8, "markov_rank": 8}


def _case(model: str, loss: str, weighting: str = "fixed-exp-decay") -> HeadConfig:
    return HeadConfig(
        model=model,
        loss=loss,
        weighting=weighting,
        implementation="eager",
        dtype="float32",
        **_SMALL,
    )


CASES: dict[str, HeadConfig] = {
    **{
        f"dflash-{loss}": _case("dflash", loss)
        for loss in ("kl_div", "rkl", "jsd", "ce", "tv", "nla", "lk_hybrid")
    },
    "dflash-kl_tv": _case("dflash", '{"kl_div": 0.5, "tv": 0.5}'),
    "dflash-ce-dpace": _case("dflash", "ce", "dpace"),
    "dflash-kl_tv-dpace": _case("dflash", '{"kl_div": 0.5, "tv": 0.5}', "dpace"),
    "dspark-kl_div": _case("dspark", "kl_div"),
    "dspark-ce_tv": _case("dspark", '{"ce": 0.1, "tv": 0.9}'),
    "dspark-ce-dpace": _case("dspark", "ce", "dpace"),
    "dspark-lk_hybrid-dpace": _case("dspark", "lk_hybrid", "dpace"),
}

_GRAD_NAMES = ("hidden", "markov_input", "markov_weight")


def _inputs(cfg: HeadConfig, device: str, dtype: torch.dtype) -> HeadInputs:
    """Case inputs drawn on CPU in fp32 (device-independent), then moved."""
    cpu = make_inputs(cfg, "cpu")

    def move(t: torch.Tensor | None, grad: bool) -> torch.Tensor | None:
        if t is None:
            return None
        moved = t.detach().to(
            device=device, dtype=dtype if t.is_floating_point() else t.dtype
        )
        return moved.requires_grad_(grad)

    return HeadInputs(
        hidden=move(cpu.hidden, True),  # type: ignore[arg-type]
        weight=move(cpu.weight, False),  # type: ignore[arg-type]
        verifier_hidden=move(cpu.verifier_hidden, False),  # type: ignore[arg-type]
        verifier_weight=move(cpu.verifier_weight, False),  # type: ignore[arg-type]
        loss_mask=move(cpu.loss_mask, False),  # type: ignore[arg-type]
        markov_input=move(cpu.markov_input, True),
        markov_weight=move(cpu.markov_weight, True),
    )


def _run(cfg: HeadConfig, inputs: HeadInputs) -> tuple[torch.Tensor, dict, dict]:
    loss, metrics = loss_head(cfg, inputs)
    loss.backward()
    grads = {
        name: getattr(inputs, name).grad.detach().float().cpu()
        for name in _GRAD_NAMES
        if getattr(inputs, name) is not None
    }
    return loss.detach().float().cpu(), metrics, grads


def _reference(
    cfg: HeadConfig, input_dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, dict, dict]:
    """fp32 eager on CPU, from inputs first rounded to ``input_dtype``.

    Rounding the inputs to the dtype a candidate path runs in leaves only the
    candidate's own arithmetic error (e.g. bf16 logits) in the comparison.
    """
    inputs = _inputs(cfg, "cpu", input_dtype)
    for field in dataclasses.fields(inputs):
        t = getattr(inputs, field.name)
        if t is not None and t.is_floating_point():
            setattr(
                inputs, field.name, t.detach().float().requires_grad_(t.requires_grad)
            )
    return _run(cfg, inputs)


def _probe(name: str, shape: torch.Size) -> torch.Tensor:
    seed = 1000 + _GRAD_NAMES.index(name)
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in sorted(metrics.items())
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }


def _fingerprint(cfg: HeadConfig) -> dict:
    loss, metrics, grads = _reference(cfg)
    return {
        "loss": float(loss),
        "metrics": _scalar_metrics(metrics),
        "grads": {
            name: {
                "norm": float(grad.norm()),
                "probe": float((grad * _probe(name, grad.shape)).sum()),
            }
            for name, grad in grads.items()
        },
    }


def _cfg(**overrides: Any) -> HeadConfig:
    base = HeadConfig(
        model="dflash",
        rows=50,
        vocab=300,
        hidden=16,
        block_size=8,
        markov_rank=4,
        loss="kl_div",
        weighting="fixed-exp-decay",
        implementation="eager",
        dtype="float32",
    )
    return dataclasses.replace(base, **overrides)


def test_num_rows_rounds_down_to_whole_blocks():
    assert _cfg(rows=50, block_size=8).num_rows == 48


@pytest.mark.parametrize("model", ["dflash", "dspark"])
@pytest.mark.parametrize("weighting", ["fixed-exp-decay", "dpace"])
def test_eager_loss_head_runs_on_cpu(model, weighting):
    cfg = _cfg(model=model, weighting=weighting)
    inputs = make_inputs(cfg, "cpu")
    assert inputs.hidden.shape == (48, 16)
    loss, metrics = loss_head(cfg, inputs)
    loss.backward()
    assert torch.isfinite(loss)
    assert "full_acc_sum" in metrics
    trainable = inputs.trainable()
    assert len(trainable) == (3 if model == "dspark" else 1)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in trainable)


def test_dflash_inputs_mask_the_anchor_slot():
    """DFlash (sample_from_anchor=False) never trains slot 0 of a block."""
    dflash = make_inputs(_cfg(model="dflash"), "cpu").loss_mask.view(-1, 8)
    dspark = make_inputs(_cfg(model="dspark"), "cpu").loss_mask.view(-1, 8)
    assert not dflash[:, 0].any()
    assert dspark[:, 0].any()


def test_golden_file_covers_every_case():
    golden = json.loads(GOLDEN_PATH.read_text())
    assert sorted(golden["cases"]) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_eager_fp32_matches_golden(name):
    expected = json.loads(GOLDEN_PATH.read_text())["cases"][name]
    actual = _fingerprint(CASES[name])

    def approx(value: float):
        return pytest.approx(value, rel=1e-4, abs=1e-6)

    assert actual["loss"] == approx(expected["loss"])
    # Expected keys only: a new metric must not break this test.
    assert expected["metrics"].keys() <= actual["metrics"].keys()
    for key, value in expected["metrics"].items():
        assert actual["metrics"][key] == approx(value), key
    assert actual["grads"].keys() == expected["grads"].keys()
    for grad_name, stats in expected["grads"].items():
        for stat, value in stats.items():
            assert actual["grads"][grad_name][stat] == approx(value), (
                f"{grad_name}.{stat}"
            )


def _assert_head_close(cfg, dtype, loss_tol, grad_tol, implementation="fused"):
    ref_loss, ref_metrics, ref_grads = _reference(cfg, input_dtype=dtype)
    fused_cfg = dataclasses.replace(cfg, implementation=implementation)
    device = "cuda"
    loss, metrics, grads = _run(fused_cfg, _inputs(fused_cfg, device, dtype))
    torch.testing.assert_close(loss, ref_loss, **loss_tol)
    for key in ref_metrics:
        if key.endswith("_loss_sum"):
            torch.testing.assert_close(
                metrics[key].float().cpu(), ref_metrics[key].float().cpu(), **loss_tol
            )
    assert grads.keys() == ref_grads.keys()
    for name, ref_grad in ref_grads.items():
        # Scale atol to the gradient so one tolerance serves every case.
        atol = grad_tol["atol"] * float(ref_grad.abs().max())
        torch.testing.assert_close(
            grads[name], ref_grad, rtol=grad_tol["rtol"], atol=atol, msg=name
        )


# fused_linear (no materialized draft logits) covers DFlash only.
IMPLEMENTATIONS = [
    "fused",
    "fused_linear",
    "fused_linear_fp32",
    "fused_linear_materialize",
]


def _skip_unsupported(name: str, implementation: str) -> None:
    if implementation.startswith("fused_linear") and CASES[name].model != "dflash":
        pytest.skip("fused_linear supports DFlash only")


@requires_cuda
# fp32 inputs give fp32 logit tiles either way: fused_linear_fp32 adds nothing here.
@pytest.mark.parametrize(
    "implementation", [i for i in IMPLEMENTATIONS if i != "fused_linear_fp32"]
)
@pytest.mark.parametrize("name", sorted(CASES))
def test_fused_fp32_matches_eager_reference(name, implementation):
    _skip_unsupported(name, implementation)
    _assert_head_close(
        CASES[name],
        torch.float32,
        loss_tol={"rtol": 1e-4, "atol": 1e-5},
        grad_tol={"rtol": 1e-3, "atol": 1e-4},
        implementation=implementation,
    )


# Today's bf16 path vs the fp32 reference on the same bf16-rounded inputs, so
# only the path's own arithmetic (bf16 logits, bf16 grad_in) is measured.
# Error = max |g - g_ref| / max |g_ref| over every input gradient. Bounds are
# 2x the value measured on ROCm (MI350P, gfx950, torch 2.13.0+rocm10.0); other
# GPUs or CUDA builds may need them widened. TV-based terms are the least
# accurate: their gradient switches on dp <= tp, which bf16 logits flip near
# ties, so the 0.1-0.4 bounds there only catch gross breakage, not precision.
# Every implementation must stay below these.
BF16_LOSS_RTOL = 2e-3
# Relative L2 error ||g - g_ref|| / ||g_ref||: unlike the max error below, it
# is not driven by a few outliers, so its bounds can be tighter.
BF16_GRAD_L2_ERR_BOUND = {
    "dflash-ce": 9e-3,
    "dflash-ce-dpace": 8e-3,
    "dflash-jsd": 2.3e-2,
    "dflash-kl_div": 2.4e-2,
    "dflash-kl_tv": 3.9e-2,
    "dflash-kl_tv-dpace": 5.3e-2,
    "dflash-lk_hybrid": 6.4e-2,
    "dflash-nla": 0.17,
    "dflash-rkl": 3e-2,
    "dflash-tv": 0.15,
    "dspark-ce-dpace": 1.1e-2,
    "dspark-ce_tv": 6.5e-2,
    "dspark-kl_div": 3.5e-2,
    "dspark-lk_hybrid-dpace": 7e-2,
}
BF16_GRAD_ERR_BOUND = {
    "dflash-ce": 9e-3,
    "dflash-ce-dpace": 8e-3,
    "dflash-jsd": 2.3e-2,
    "dflash-kl_div": 2.3e-2,
    "dflash-kl_tv": 7.3e-2,
    "dflash-kl_tv-dpace": 0.1,
    "dflash-lk_hybrid": 0.12,
    "dflash-nla": 0.36,
    "dflash-rkl": 6e-2,
    "dflash-tv": 0.28,
    "dspark-ce-dpace": 1.7e-2,
    "dspark-ce_tv": 0.14,
    "dspark-kl_div": 4e-2,
    "dspark-lk_hybrid-dpace": 0.15,
}


def test_bf16_bounds_cover_every_case():
    assert sorted(BF16_GRAD_ERR_BOUND) == sorted(CASES)
    assert sorted(BF16_GRAD_L2_ERR_BOUND) == sorted(CASES)


@functools.cache
def _bf16_fused_and_reference(
    name: str, implementation: str
) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """``(loss, ref_loss, grads, ref_grads)``, shared by both bf16 tests."""
    cfg = CASES[name]
    ref_loss, _, ref_grads = _reference(cfg, input_dtype=torch.bfloat16)
    fused_cfg = dataclasses.replace(cfg, implementation=implementation)
    loss, _, grads = _run(fused_cfg, _inputs(fused_cfg, "cuda", torch.bfloat16))
    assert grads.keys() == ref_grads.keys()
    for grad_name, ref in ref_grads.items():
        assert torch.isfinite(ref).all(), f"{grad_name}: non-finite reference gradient"
    return loss, ref_loss, grads, ref_grads


@requires_cuda
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_fused_bf16_error_vs_eager_reference(name, implementation):
    """fused_linear (fp32 logits) must stay within today's measured bounds."""
    _skip_unsupported(name, implementation)
    loss, ref_loss, grads, ref_grads = _bf16_fused_and_reference(name, implementation)
    torch.testing.assert_close(loss, ref_loss, rtol=BF16_LOSS_RTOL, atol=0.0)
    bound = BF16_GRAD_ERR_BOUND[name]
    for grad_name, ref in ref_grads.items():
        error = float((grads[grad_name] - ref).abs().max())
        limit = bound * float(ref.abs().max())
        # Checked per gradient and written so that a NaN error fails.
        assert error <= limit, f"{grad_name}: error {error:.3g} > limit {limit:.3g}"


@requires_cuda
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_fused_bf16_l2_error_vs_eager_reference(name, implementation):
    _skip_unsupported(name, implementation)
    _, _, grads, ref_grads = _bf16_fused_and_reference(name, implementation)
    bound = BF16_GRAD_L2_ERR_BOUND[name]
    for grad_name, ref in ref_grads.items():
        error = float((grads[grad_name] - ref).norm())
        limit = bound * float(ref.norm())
        assert error <= limit, f"{grad_name}: L2 error {error:.3g} > limit {limit:.3g}"


@pytest.mark.skipif(
    not os.environ.get("SPECULATORS_REGENERATE_GOLDEN"),
    reason="set SPECULATORS_REGENERATE_GOLDEN=1 to rewrite the golden file",
)
def test_regenerate_golden():
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "fp32 eager loss-head fingerprints; regenerate with "
            "`SPECULATORS_REGENERATE_GOLDEN=1 pytest "
            "tests/unit/models/test_loss_head.py -k regenerate`"
        ),
        "torch_version": torch.__version__,
        "cases": {name: _fingerprint(cfg) for name, cfg in sorted(CASES.items())},
    }
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2) + "\n")
