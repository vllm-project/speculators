"""DFlash2 with loss_implementation="fused_linear": one real training step.

A tiny DFlash2 model is run once through the materialized path and once
through fused_linear (the hidden-state path that never builds the unary
logits or, with recomputed targets, the target logits), both configured
through ``get_trainer_kwargs`` as the trainer does. The loss, every metric
and every trainable parameter's gradient must agree -- including the
candidate selector's codebooks, which only the chunked path reaches via
``chunked_topk_ids`` / ``chunked_gather_values`` instead of a plain
``.topk()`` / ``.gather()`` on materialized logits.
"""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.core import DFlash2DraftModel

pytestmark = pytest.mark.usefixtures("narrowest_tile")

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused losses need a CUDA or ROCm device"
)

HIDDEN = 64
VOCAB = 1000
BLOCK = 4
SEQ_LEN = 256
MAX_ANCHORS = 32  # N = MAX_ANCHORS * BLOCK = 128 loss rows
TOP_K = 8


def _model(sample_from_anchor: bool, dtype: torch.dtype) -> DFlash2DraftModel:
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
    config = DFlash2SpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=16,
        selector_top_k=TOP_K,
    )
    torch.manual_seed(0)
    model = DFlash2DraftModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
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
    selector_miss_policy="replace",
):
    train_kwargs, _ = DFlash2DraftModel.get_trainer_kwargs(
        loss_fn=spec,
        loss_implementation=implementation,
        per_position_loss_weight=weighting,
        max_anchors=max_anchors,
        fused_linear_targets=targets,
        selector_miss_policy=selector_miss_policy,
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
    selector_miss_policy="replace",
):
    loss, metrics, grads = _step(
        model,
        inputs,
        spec,
        "fused",
        weighting,
        max_anchors=max_anchors,
        selector_miss_policy=selector_miss_policy,
    )
    loss_fl, metrics_fl, grads_fl = _step(
        model,
        inputs,
        spec,
        "fused_linear",
        weighting,
        targets,
        max_anchors,
        selector_miss_policy,
    )

    torch.testing.assert_close(loss_fl, loss, rtol=loss_rtol, atol=1e-6)
    assert metrics_fl.keys() == metrics.keys()
    for key, value in metrics.items():
        if key.endswith("_loss_sum") or key == "loss_sum":
            torch.testing.assert_close(
                metrics_fl[key].float(), value.float(), rtol=loss_rtol, atol=1e-6
            )
        else:  # accuracy / recall / mass counts from the argmaxes and gathers
            torch.testing.assert_close(metrics_fl[key], value, msg=key)
    assert grads_fl.keys() == grads.keys()
    assert "lm_head.weight" not in grads  # frozen in both paths
    assert any("candidate_selector" in name for name in grads)  # selector reached
    assert len(grads) > 5
    for name, grad in grads.items():
        atol = grad_rtol * float(grad.abs().max()) + 1e-12
        torch.testing.assert_close(
            grads_fl[name], grad, rtol=grad_rtol, atol=atol, msg=name
        )


@requires_gpu
@pytest.mark.parametrize(
    ("spec", "weighting", "sample_from_anchor"),
    [
        ("kl_div", "fixed-exp-decay", False),
        ("ce", "dpace", True),
    ],
)
@pytest.mark.parametrize("targets", ["recompute", "materialize"])
@pytest.mark.parametrize("selector_miss_policy", ["replace", "strict"])
def test_training_step_matches_fused_eager(
    spec, weighting, sample_from_anchor, targets, selector_miss_policy, monkeypatch
):
    # 1 MiB budget -> small row chunks: the 128 loss rows span several chunks.
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")
    model = _model(sample_from_anchor, torch.float32)
    with torch.compiler.set_stance("force_eager"):
        _compare(
            model,
            _inputs(torch.float32),
            spec,
            weighting,
            1e-5,
            1e-3,
            targets,
            selector_miss_policy=selector_miss_policy,
        )


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
            "kl_div",
            "fixed-exp-decay",
            1e-5,
            1e-3,
            max_anchors=SEQ_LEN // BLOCK,
        )
