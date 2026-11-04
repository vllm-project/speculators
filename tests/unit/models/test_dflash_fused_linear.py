"""DFlash with loss_implementation="fused_linear": one real training step.

A tiny DFlash model is run once through the materialized fused path and once
through fused_linear (the hidden-state path that never builds the draft
logits), both configured through ``get_trainer_kwargs`` as the trainer does.
The loss, every metric and every trainable parameter's gradient must agree.
"""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel

pytestmark = pytest.mark.usefixtures("narrowest_tile")

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused losses need a CUDA or ROCm device"
)


@pytest.fixture
def aot_eager_forward(monkeypatch):
    """Compile DFlash.forward with aot_eager: Dynamo and AOTAutograd trace the
    op as in training, without Inductor codegen (~25 s per cold compile)."""
    original = getattr(DFlashDraftModel.forward, "_torchdynamo_orig_callable", None)
    if original is not None:
        compiled = torch.compile(original, backend="aot_eager")
        monkeypatch.setattr(DFlashDraftModel, "forward", compiled)


HIDDEN = 64
VOCAB = 1000
BLOCK = 4
SEQ_LEN = 256
MAX_ANCHORS = 32  # N = MAX_ANCHORS * BLOCK = 128 loss rows


def _model(sample_from_anchor: bool, dtype: torch.dtype) -> DFlashDraftModel:
    tl_config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
    )
    torch.manual_seed(0)
    model = DFlashDraftModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
        # lm_head / verifier_lm_head at a scale that gives non-trivial softmaxes
        torch.nn.init.normal_(model.lm_head.weight, std=0.3)
        model.verifier_lm_head.weight.copy_(
            model.lm_head.weight + 0.1 * torch.randn_like(model.lm_head.weight)
        )
        torch.nn.init.ones_(model.verifier_norm.weight)
    return model.to(device="cuda", dtype=dtype).train()  # type: ignore[call-arg]


def _inputs(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(1)
    loss_mask = torch.ones(1, SEQ_LEN, device="cuda")
    loss_mask[0, 40:70] = 0
    return {
        "hidden_states": torch.randn(
            1, SEQ_LEN, 2 * HIDDEN, device="cuda", generator=gen
        ).to(dtype),
        "input_ids": torch.randint(
            0, VOCAB, (1, SEQ_LEN), device="cuda", generator=gen
        ),
        "loss_mask": loss_mask,
        "verifier_last_hidden_states": torch.randn(
            1, SEQ_LEN, HIDDEN, device="cuda", generator=gen
        ).to(dtype),
        "document_ids": torch.zeros(1, SEQ_LEN, device="cuda", dtype=torch.long),
    }


def _step(
    model,
    inputs,
    spec,
    implementation,
    weighting,
    targets="recompute",
    max_anchors=MAX_ANCHORS,
):
    train_kwargs, _ = DFlashDraftModel.get_trainer_kwargs(
        loss_fn=spec,
        loss_implementation=implementation,
        per_position_loss_weight=weighting,
        max_anchors=max_anchors,
        fused_linear_targets=targets,
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(2)  # anchor sampling
    _, loss, metrics = model(**inputs, **train_kwargs)
    loss.backward()
    grads = {
        name: p.grad.detach().float().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach().float(), metrics, grads


def _compare(
    model,
    inputs,
    spec,
    weighting,
    loss_rtol,
    grad_rtol,
    targets="recompute",
    max_anchors=MAX_ANCHORS,
):
    loss, metrics, grads = _step(
        model, inputs, spec, "fused", weighting, max_anchors=max_anchors
    )
    loss_fl, metrics_fl, grads_fl = _step(
        model, inputs, spec, "fused_linear", weighting, targets, max_anchors
    )

    torch.testing.assert_close(loss_fl, loss, rtol=loss_rtol, atol=1e-6)
    assert metrics_fl.keys() == metrics.keys()
    for key, value in metrics.items():
        if key.endswith("_loss_sum") or key == "loss_sum":
            torch.testing.assert_close(
                metrics_fl[key].float(), value.float(), rtol=loss_rtol, atol=1e-6
            )
        else:  # accuracy counts from the argmaxes
            torch.testing.assert_close(metrics_fl[key], value, msg=key)
    assert grads_fl.keys() == grads.keys()
    assert "lm_head.weight" not in grads  # frozen in both paths
    assert len(grads) > 5
    for name, grad in grads.items():
        atol = grad_rtol * float(grad.abs().max()) + 1e-12
        torch.testing.assert_close(
            grads_fl[name], grad, rtol=grad_rtol, atol=atol, msg=name
        )


SPECS = ["kl_div", '{"kl_div": 0.5, "tv": 0.5}', "ce"]


@requires_gpu
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("weighting", ["fixed-exp-decay", "dpace"])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("targets", ["recompute", "materialize"])
def test_training_step_matches_fused_eager(
    spec, weighting, sample_from_anchor, targets, monkeypatch
):
    # 1 MiB budget -> 64-row chunks: the 128 loss rows span two chunks
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")
    model = _model(sample_from_anchor, torch.float32)
    with torch.compiler.set_stance("force_eager"):
        _compare(model, _inputs(torch.float32), spec, weighting, 1e-5, 1e-3, targets)


@requires_gpu
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_recomputed_targets_cover_the_full_sequence_branch(sample_from_anchor):
    """max_anchors * block == seq_len takes DFlash's full-sequence target branch
    (roll then gather); the recompute path gathers verifier rows first."""
    model = _model(sample_from_anchor, torch.float32)
    with torch.compiler.set_stance("force_eager"):
        _compare(
            model,
            _inputs(torch.float32),
            '{"kl_div": 0.5, "tv": 0.5}',
            "fixed-exp-decay",
            1e-5,
            1e-3,
            max_anchors=SEQ_LEN // BLOCK,
        )


@requires_gpu
@pytest.mark.usefixtures("aot_eager_forward")
@pytest.mark.parametrize("spec", ["kl_div", '{"kl_div": 0.5, "tv": 0.5}'])
@pytest.mark.parametrize("targets", ["recompute", "materialize"])
def test_training_step_matches_fused_compiled(spec, targets, monkeypatch):
    """DFlash.forward is torch.compile'd on GPU; the op must trace as one node."""
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")
    torch._dynamo.reset()
    model = _model(False, torch.float32)
    _compare(
        model, _inputs(torch.float32), spec, "fixed-exp-decay", 1e-5, 1e-3, targets
    )


@requires_gpu
def test_training_step_bf16():
    """bf16 model: fused_linear's fp32 logits differ from the bf16-logit path by
    bf16 rounding only; gradients must point the same way."""
    model = _model(False, torch.bfloat16)
    inputs = _inputs(torch.bfloat16)
    with torch.compiler.set_stance("force_eager"):
        loss, _, grads = _step(model, inputs, "kl_div", "fused", "fixed-exp-decay")
        loss_fl, _, grads_fl = _step(
            model, inputs, "kl_div", "fused_linear", "fixed-exp-decay"
        )
    torch.testing.assert_close(loss_fl, loss, rtol=2e-2, atol=1e-4)
    for name, grad in grads.items():
        assert torch.isfinite(grads_fl[name]).all(), name
        cosine = torch.nn.functional.cosine_similarity(
            grads_fl[name].flatten(), grad.flatten(), dim=0
        )
        assert cosine > 0.99, (name, float(cosine))


@requires_gpu
@pytest.mark.usefixtures("aot_eager_forward")
@pytest.mark.parametrize("params", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("compiled", [False, True])
def test_training_step_under_autocast(params, compiled):
    """As the trainer runs it: autocast(bf16), fp32 params (DDP) or bf16 params
    (FSDP param_dtype). fused_linear keeps the logits fp32, so it differs from
    the bf16-logit path by bf16 rounding only."""
    if compiled:
        torch._dynamo.reset()
    model = _model(False, params)
    inputs = _inputs(torch.bfloat16)
    stance = "default" if compiled else "force_eager"
    results = {}
    for implementation in ("fused", "fused_linear"):
        with (
            torch.compiler.set_stance(stance),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            results[implementation] = _step(
                model, inputs, "kl_div", implementation, "fixed-exp-decay"
            )
    loss, _, grads = results["fused"]
    loss_fl, _, grads_fl = results["fused_linear"]
    torch.testing.assert_close(loss_fl, loss, rtol=1e-2, atol=1e-4)
    assert grads_fl.keys() == grads.keys()
    for name, grad in grads.items():
        assert torch.isfinite(grads_fl[name]).all(), name
        cosine = torch.nn.functional.cosine_similarity(
            grads_fl[name].flatten(), grad.flatten(), dim=0
        )
        assert cosine > 0.99, (name, float(cosine))
