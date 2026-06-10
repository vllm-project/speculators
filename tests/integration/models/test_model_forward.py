"""Integration tests for draft model forward passes with real weights.

Covers DFlash, Eagle3, PEagle, and MTP models with shared parametrized tests
for training, multi-batch, and vocab boundary scenarios, plus model-specific
parameter variation tests.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import pytest
import torch

from speculators.models.mtp import shift_batch_mtp
from speculators.models.mtp.core import compute_step_weights
from tests.conftest import requires_cuda, requires_transformers_version
from tests.integration.conftest import (
    HIDDEN_SIZE,
    TINY_QWEN3_5_KWARGS,
    VOCAB_SIZE,
    make_batch,
    make_dflash_model,
    make_eagle3_model,
    make_mtp_model,
    make_peagle_model,
    make_sample,
)

MAX_LEN = 128
HIDDEN_MULTIPLIER = 3
LOSS_MASK_CASES = ["all", "none", "random", "alternating"]

SAMPLE_CONFIGS = [
    pytest.param([128], id="single_sample"),
    pytest.param([64, 64], id="two_equal"),
    pytest.param([32, 96], id="two_unequal"),
    pytest.param([8] * 20, id="twenty_tiny"),
]

MULTI_BATCH_CONFIGS: list[list[int]] = [
    [16, 16, 8, 10, 15, 12],
    [32, 32, 32],
    [64, 64],
    [32, 3, 17],
    [128],
    [16],
    [],
]

# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

_requires_qwen3_5 = requires_transformers_version("5.2.0")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[..., Any]
    forward_kwargs: dict[str, Any] = field(default_factory=dict)
    hidden_size: int = HIDDEN_SIZE
    hidden_multiplier: int = HIDDEN_MULTIPLIER
    batch_factory: Callable[..., Any] = make_batch


DFLASH_SPEC = ModelSpec(name="dflash", factory=make_dflash_model)
EAGLE3_SPEC = ModelSpec(
    name="eagle3", factory=make_eagle3_model, forward_kwargs={"ttt_steps": 2}
)
PEAGLE_SPEC = ModelSpec(name="peagle", factory=make_peagle_model)
MTP_SPEC = ModelSpec(
    name="mtp",
    factory=make_mtp_model,
    forward_kwargs={"step_weights": [0.51, 0.31, 0.18]},
    hidden_size=TINY_QWEN3_5_KWARGS["hidden_size"],
    hidden_multiplier=1,
    batch_factory=partial(make_batch, num_target_layers=1, preprocess=shift_batch_mtp),
)

ALL_SPECS = [
    pytest.param(DFLASH_SPEC, id="dflash"),
    pytest.param(EAGLE3_SPEC, id="eagle3"),
    pytest.param(PEAGLE_SPEC, id="peagle"),
    pytest.param(MTP_SPEC, id="mtp", marks=_requires_qwen3_5),
]

VOCAB_SPECS = [
    pytest.param(DFLASH_SPEC, id="dflash"),
    pytest.param(EAGLE3_SPEC, id="eagle3"),
    pytest.param(PEAGLE_SPEC, id="peagle"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(
    seq_lengths: list[int],
    loss_mask_pattern: str = "all",
    vocab_size: int = VOCAB_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    hidden_multiplier: int = HIDDEN_MULTIPLIER,
    boundary_token_ids: list[int] | None = None,
) -> list[dict[str, torch.Tensor]]:
    return [
        make_sample(
            seq_len=sl,
            hidden_size=hidden_size,
            hidden_multiplier=hidden_multiplier,
            vocab_size=vocab_size,
            loss_mask_pattern=loss_mask_pattern,
            include_verifier_states=True,
            boundary_token_ids=boundary_token_ids,
        )
        for sl in seq_lengths
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=ALL_SPECS)
def model_and_spec(request):
    spec: ModelSpec = request.param
    model = spec.factory()
    yield model, spec
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def draft_vocab_model(request):
    spec: ModelSpec = request.param
    model = spec.factory(draft_vocab_size=32)
    t2d = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    t2d[:32] = True
    d2t = torch.arange(32, dtype=torch.long)
    model.load_vocab_mappings(t2d.to("cuda"), d2t.to("cuda"))
    yield model, spec
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Shared tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestTraining:
    """Forward + backward pass across all models."""

    @pytest.mark.parametrize("seq_lengths", SAMPLE_CONFIGS)
    @pytest.mark.parametrize("loss_mask_pattern", LOSS_MASK_CASES)
    def test_forward_backward(self, model_and_spec, seq_lengths, loss_mask_pattern):
        model, spec = model_and_spec
        samples = _make_samples(
            seq_lengths,
            loss_mask_pattern=loss_mask_pattern,
            hidden_size=spec.hidden_size,
            hidden_multiplier=spec.hidden_multiplier,
        )
        batch = spec.batch_factory(
            max_len=MAX_LEN, samples=samples, hidden_size=spec.hidden_size
        )
        draft_tokens, loss, metrics = model(**batch, **spec.forward_kwargs)

        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        assert "loss_sum" in metrics
        assert "loss_total" in metrics
        loss.backward()


@requires_cuda
class TestMultiBatch:
    """Run multiple batches back-to-back to test statefulness and cache clearing."""

    def test_varying_batches(self, model_and_spec):
        model, spec = model_and_spec
        torch.compiler.reset()
        for seq_lengths in MULTI_BATCH_CONFIGS:
            samples = _make_samples(
                seq_lengths,
                hidden_size=spec.hidden_size,
                hidden_multiplier=spec.hidden_multiplier,
            )
            batch = spec.batch_factory(
                max_len=MAX_LEN, samples=samples, hidden_size=spec.hidden_size
            )
            draft_tokens, loss, metrics = model(**batch, **spec.forward_kwargs)
            assert loss.isfinite(), f"Loss not finite for seq_lengths={seq_lengths}"
            loss.backward()

    def test_varying_loss_masks_across_batches(self, model_and_spec):
        model, spec = model_and_spec
        torch.compiler.reset()
        for pattern in LOSS_MASK_CASES:
            samples = _make_samples(
                [64, 64],
                loss_mask_pattern=pattern,
                hidden_size=spec.hidden_size,
                hidden_multiplier=spec.hidden_multiplier,
            )
            batch = spec.batch_factory(
                max_len=MAX_LEN, samples=samples, hidden_size=spec.hidden_size
            )
            draft_tokens, loss, metrics = model(**batch, **spec.forward_kwargs)
            assert loss.isfinite(), f"Loss not finite for loss_mask_pattern={pattern}"
            loss.backward()


@requires_cuda
class TestVocabBoundary:
    """Tests with draft vocab mapping."""

    @pytest.mark.parametrize("draft_vocab_model", VOCAB_SPECS, indirect=True)
    def test_boundary_tokens(self, draft_vocab_model):
        model, spec = draft_vocab_model
        samples = _make_samples([128], vocab_size=32, boundary_token_ids=[0, 31])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = model(**batch, **spec.forward_kwargs)

        assert loss.isfinite()
        loss.backward()


# ---------------------------------------------------------------------------
# Model-specific parameter tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestDFlashParams:
    @pytest.mark.parametrize("block_size", [2, 4, 8])
    def test_varying_block_size(self, block_size):
        model = make_dflash_model(block_size=block_size, max_anchors=4)
        samples = _make_samples([128])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = model(**batch)

        assert loss.isfinite()
        loss.backward()

    @pytest.mark.parametrize("max_anchors", [2, 8, 16])
    def test_varying_max_anchors(self, max_anchors):
        model = make_dflash_model(max_anchors=max_anchors)
        samples = _make_samples([128])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = model(**batch)

        assert loss.isfinite()
        loss.backward()


@requires_cuda
class TestEagle3Params:
    @pytest.mark.parametrize("ttt_steps", [1, 3, 5])
    def test_varying_ttt_steps(self, ttt_steps):
        model = make_eagle3_model()
        samples = _make_samples([128])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = model(**batch, ttt_steps=ttt_steps)

        assert len(draft_tokens) == ttt_steps
        for dt in draft_tokens:
            assert dt.shape == (1, MAX_LEN)
            assert dt.dtype == torch.long
        assert loss.isfinite()
        loss.backward()


@requires_cuda
class TestPEagleParams:
    @pytest.mark.parametrize("num_depths", [2, 4, 8])
    def test_varying_num_depths(self, num_depths):
        model = make_peagle_model(num_depths=num_depths)
        samples = _make_samples([128])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = model(**batch)

        assert loss.isfinite()
        loss.backward()

    @pytest.mark.parametrize("down_sample_ratio", [0.3, 0.7, 1.0])
    def test_varying_down_sample_ratio(self, down_sample_ratio):
        model = make_peagle_model(down_sample_ratio=down_sample_ratio)
        samples = _make_samples([128])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = model(**batch)

        assert loss.isfinite()
        loss.backward()


@requires_cuda
@requires_transformers_version("5.2.0")
class TestMTPParams:
    @pytest.mark.parametrize("num_speculative_steps", [1, 2, 5])
    def test_varying_num_speculative_steps(self, num_speculative_steps):
        model = make_mtp_model(num_speculative_steps=num_speculative_steps)
        step_weights = compute_step_weights(num_steps=num_speculative_steps)
        samples = _make_samples(
            [128],
            hidden_size=TINY_QWEN3_5_KWARGS["hidden_size"],
            hidden_multiplier=1,
            vocab_size=TINY_QWEN3_5_KWARGS["vocab_size"],
        )
        batch = make_batch(
            max_len=MAX_LEN,
            samples=samples,
            hidden_size=TINY_QWEN3_5_KWARGS["hidden_size"],
            num_target_layers=1,
            preprocess=shift_batch_mtp,
        )
        logits_list, loss, metrics = model(**batch, step_weights=step_weights)

        assert loss.isfinite()
        loss.backward()
