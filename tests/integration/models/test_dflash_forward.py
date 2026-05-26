"""Integration tests for DFlashDraftModel forward passes with real weights."""

import pytest
import torch

from tests.conftest import requires_cuda
from tests.integration.conftest import (
    HIDDEN_SIZE,
    VOCAB_SIZE,
    make_batch,
    make_dflash_model,
    make_sample,
)

MAX_LEN = 128
NUM_TARGET_LAYERS = 2

LOSS_MASK_CASES = ["all", "none", "random", "alternating"]


def _make_samples(
    seq_lengths: list[int],
    loss_mask_pattern: str = "all",
    vocab_size: int = VOCAB_SIZE,
    boundary_token_ids: list[int] | None = None,
) -> list[dict[str, torch.Tensor]]:
    return [
        make_sample(
            seq_len=sl,
            hidden_size=HIDDEN_SIZE,
            hidden_multiplier=NUM_TARGET_LAYERS,
            vocab_size=vocab_size,
            loss_mask_pattern=loss_mask_pattern,
            include_verifier_states=True,
            boundary_token_ids=boundary_token_ids,
        )
        for sl in seq_lengths
    ]


SAMPLE_CONFIGS = [
    pytest.param([128], id="single_sample"),
    pytest.param([64, 64], id="two_equal"),
    pytest.param([32, 96], id="two_unequal"),
    pytest.param([16] * 8, id="eight_tiny"),
    pytest.param([8] * 20, id="twenty_tiny"),
]


@requires_cuda
class TestDFlashTraining:
    """Forward pass in training mode (DFlash always requires verifier states)."""

    @pytest.mark.parametrize("seq_lengths", SAMPLE_CONFIGS)
    @pytest.mark.parametrize("loss_mask_pattern", LOSS_MASK_CASES)
    def test_forward_backward_produces_valid_outputs(
        self, dflash_model, seq_lengths, loss_mask_pattern
    ):
        samples = _make_samples(seq_lengths, loss_mask_pattern=loss_mask_pattern)
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = dflash_model(**batch)

        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        assert "loss_sum" in metrics
        assert "loss_total" in metrics

        loss.backward()

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
class TestDFlashMultiBatch:
    """Run multiple batches back-to-back through the same model to test
    statefulness, cache clearing, and varying batch compositions."""

    def test_multi_then_single(self, dflash_model):
        """Multi-sample batches, then single-sample."""
        batch_configs: list[list[int]] = [
            [32, 32, 32],
            [64, 64],
            [128],
            [16],
            # Empty batch ([]) excluded: create_empty_sample dtype bug (#527)
        ]
        for seq_lengths in batch_configs:
            samples = _make_samples(seq_lengths)
            batch = make_batch(
                max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE
            )
            draft_tokens, loss, metrics = dflash_model(**batch)
            assert loss.isfinite(), f"Loss not finite for seq_lengths={seq_lengths}"
            loss.backward()

    def test_alternating_batch_sizes(self, dflash_model):
        """Alternate between large multi-sample and single-sample batches."""
        batch_configs: list[list[int]] = [
            [16] * 8,
            [128],
            [32, 32, 32, 32],
            [64],
            [8] * 16,
            [128],
        ]
        for seq_lengths in batch_configs:
            samples = _make_samples(seq_lengths)
            batch = make_batch(
                max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE
            )
            draft_tokens, loss, metrics = dflash_model(**batch)
            assert loss.isfinite()
            loss.backward()

    def test_varying_loss_masks_across_batches(self, dflash_model):
        """Each batch uses a different loss mask pattern."""
        for pattern in LOSS_MASK_CASES:
            samples = _make_samples([64, 64], loss_mask_pattern=pattern)
            batch = make_batch(
                max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE
            )
            draft_tokens, loss, metrics = dflash_model(**batch)
            assert loss.isfinite(), f"Loss not finite for loss_mask_pattern={pattern}"
            loss.backward()


@requires_cuda
class TestDFlashVocabBoundary:
    """Tests with draft vocab mapping."""

    @pytest.fixture
    def dflash_draft_vocab_model(self):
        model = make_dflash_model(draft_vocab_size=32)
        t2d = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        t2d[:32] = True
        d2t = torch.arange(32, dtype=torch.long)
        model.load_vocab_mappings(t2d.to("cuda"), d2t.to("cuda"))
        yield model
        del model
        torch.cuda.empty_cache()

    def test_boundary_tokens(self, dflash_draft_vocab_model):
        samples = _make_samples([128], vocab_size=32, boundary_token_ids=[0, 31])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = dflash_draft_vocab_model(**batch)

        assert loss.isfinite()
        loss.backward()
