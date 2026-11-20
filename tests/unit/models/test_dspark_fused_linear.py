"""DSpark with loss_implementation="fused_linear" (``fused_linear_reduced``)
against the materialized ``fused`` path: same loss, metrics and gradients,
the trainable ``markov_w2``'s included -- for the loss head alone and for a
whole training step."""

import dataclasses

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.loss_head import HeadConfig, loss_head, make_inputs
from tests.unit.models.test_dflash_fused_linear import (
    BLOCK,
    HIDDEN,
    MAX_ANCHORS,
    VOCAB,
)
from tests.unit.models.test_dflash_fused_linear import _inputs as _dflash_inputs
from tests.unit.models.test_fused_losses import DEVICE, requires_accelerator

pytestmark = [requires_accelerator, pytest.mark.usefixtures("narrowest_tile")]

BASE = HeadConfig(
    model="dspark",
    rows=512,
    vocab=3000,
    hidden=64,
    block_size=8,
    markov_rank=16,
    loss="kl_div",
    weighting="fixed-exp-decay",
    implementation="fused",
    dtype="float32",
)
CASES = [
    ("kl_div", "fixed-exp-decay"),
    ("ce", "dpace"),
    ('{"ce": 0.1, "tv": 0.9}', "fixed-exp-decay"),
    ("nla", "fixed-exp-decay"),
]


def _run(cfg):
    inputs = make_inputs(cfg, DEVICE or "cpu", seed=1)
    loss, metrics = loss_head(cfg, inputs)
    loss.backward()
    return loss, metrics, [t.grad for t in inputs.trainable()]


@pytest.mark.parametrize(("loss", "weighting"), CASES)
@pytest.mark.parametrize("targets", ["fused_linear", "fused_linear_materialize"])
def test_matches_materialized(loss, weighting, targets):
    cfg = dataclasses.replace(BASE, loss=loss, weighting=weighting)
    ref, ref_metrics, ref_grads = _run(cfg)
    out, metrics, grads = _run(dataclasses.replace(cfg, implementation=targets))
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6)
    for key in ("accept_rate_sum", "full_acc_sum", "eal_sum"):
        torch.testing.assert_close(metrics[key], ref_metrics[key], rtol=1e-4, atol=1e-4)
    for g, r in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(g, r, rtol=1e-3, atol=1e-4 * float(r.abs().max()))


def _dspark(head_type):
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
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        markov_head_type=head_type,
    )
    torch.manual_seed(0)
    model = DSparkDraftModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
        # full vocab: lm_head is the verifier head, as after load_verifier_weights
        torch.nn.init.normal_(model.lm_head.weight, std=0.3)
        model.verifier_lm_head.weight.copy_(model.lm_head.weight)
        torch.nn.init.ones_(model.verifier_norm.weight)
    return model.to(device=DEVICE).train()  # type: ignore[call-arg]


def _model_step(model, implementation, spec, weighting, targets):
    kwargs, _ = DSparkDraftModel.get_trainer_kwargs(
        loss_fn=spec,
        loss_implementation=implementation,
        per_position_loss_weight=weighting,
        max_anchors=MAX_ANCHORS,
        fused_linear_targets=targets,
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(2)  # anchor sampling
    _, loss, metrics = model(**_dflash_inputs(torch.float32), **kwargs)
    loss.backward()
    grads = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    return loss.detach(), metrics, grads


MODEL_CASES = [
    ("kl_div", "fixed-exp-decay", "vanilla"),
    ("ce", "dpace", "rnn"),
    ('{"ce": 0.1, "tv": 0.9}', "fixed-exp-decay", "gated"),
]


@pytest.mark.parametrize(("spec", "weighting", "head_type"), MODEL_CASES)
@pytest.mark.parametrize("targets", ["recompute", "materialize"])
def test_training_step_matches_fused(spec, weighting, head_type, targets, monkeypatch):
    """A DSpark training step: loss, metrics and every gradient (markov_w2's
    included) agree with the materialized path."""
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")  # 2 chunks
    model = _dspark(head_type)
    with torch.compiler.set_stance("force_eager"):
        ref = _model_step(model, "fused", spec, weighting, targets)
        out = _model_step(model, "fused_linear", spec, weighting, targets)
    torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=1e-6)
    assert out[1].keys() == ref[1].keys()
    for key, value in ref[1].items():
        torch.testing.assert_close(out[1][key], value, rtol=1e-4, atol=1e-5, msg=key)
    assert out[2].keys() == ref[2].keys()
    assert "markov_head.markov_w2.weight" in ref[2]
    for name, grad in ref[2].items():
        atol = 1e-3 * float(grad.abs().max()) + 1e-12
        torch.testing.assert_close(out[2][name], grad, rtol=1e-3, atol=atol, msg=name)


def test_compiled_training_step_matches_eager(monkeypatch):
    """Dynamo and AOTAutograd trace the op as in training (aot_eager: no
    Inductor codegen)."""
    original = DSparkDraftModel.forward._torchdynamo_orig_callable  # type: ignore[attr-defined]
    monkeypatch.setattr(
        DSparkDraftModel, "forward", torch.compile(original, backend="aot_eager")
    )
    model = _dspark("vanilla")
    with torch.compiler.set_stance("force_eager"):
        ref = _model_step(model, "fused_linear", "kl_div", "dpace", "recompute")
    out = _model_step(model, "fused_linear", "kl_div", "dpace", "recompute")
    torch.testing.assert_close(out[0], ref[0])
    for name, grad in ref[2].items():
        torch.testing.assert_close(out[2][name], grad, msg=name)
