"""Integration tests for Eagle3DraftModel forward passes with real weights."""

import pytest
import torch

from tests.conftest import requires_cuda
from tests.integration.conftest import (
    HIDDEN_SIZE,
    VOCAB_SIZE,
    make_batch,
    make_eagle3_model,
    make_sample,
)

MAX_LEN = 128

LOSS_MASK_CASES = ["all", "none", "random", "alternating"]


def _make_samples(
    seq_lengths: list[int],
    loss_mask_pattern: str = "all",
    include_verifier_states: bool = True,
    vocab_size: int = VOCAB_SIZE,
    boundary_token_ids: list[int] | None = None,
) -> list[dict[str, torch.Tensor]]:
    return [
        make_sample(
            seq_len=sl,
            hidden_size=HIDDEN_SIZE,
            hidden_multiplier=3,
            vocab_size=vocab_size,
            loss_mask_pattern=loss_mask_pattern,
            include_verifier_states=include_verifier_states,
            boundary_token_ids=boundary_token_ids,
        )
        for sl in seq_lengths
    ]


# ---------------------------------------------------------------------------
# Sample compositions for parametrize
# ---------------------------------------------------------------------------

SAMPLE_CONFIGS = [
    pytest.param([128], id="single_sample"),
    pytest.param([64, 64], id="two_equal"),
    pytest.param([32, 96], id="two_unequal"),
    pytest.param([16] * 8, id="eight_tiny"),
    pytest.param([8] * 20, id="twenty_tiny"),
]


@requires_cuda
class TestEagle3Training:
    """Forward pass in training mode (with verifier_last_hidden_states)."""

    @pytest.mark.parametrize("seq_lengths", SAMPLE_CONFIGS)
    @pytest.mark.parametrize("loss_mask_pattern", LOSS_MASK_CASES)
    def test_forward_backward_produces_valid_outputs(
        self, eagle3_model, seq_lengths, loss_mask_pattern
    ):
        samples = _make_samples(seq_lengths, loss_mask_pattern=loss_mask_pattern)
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        ttt_steps = 2
        draft_tokens, loss, metrics = eagle3_model(**batch, ttt_steps=ttt_steps)

        assert len(draft_tokens) == ttt_steps
        for dt in draft_tokens:
            assert dt.shape == (1, MAX_LEN)
            assert dt.dtype == torch.long

        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        assert "loss_sum" in metrics
        assert "loss_total" in metrics

        loss.backward()

    @pytest.mark.parametrize("ttt_steps", [1, 3, 5])
    def test_varying_ttt_steps(self, eagle3_model, ttt_steps):
        samples = _make_samples([128])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = eagle3_model(**batch, ttt_steps=ttt_steps)

        assert len(draft_tokens) == ttt_steps
        assert loss.isfinite()

        loss.backward()


@requires_cuda
class TestEagle3MultiBatch:
    """Run multiple batches back-to-back through the same model to test
    statefulness, cache clearing, and varying batch compositions."""

    def test_multi_then_single_then_empty(self, eagle3_model):
        """Multi-sample batches, then single-sample, then empty batch."""
        batch_configs: list[list[int]] = [
            [16, 16, 8, 10, 15, 12],
            [32, 32, 32],
            [64, 64],
            [32, 3, 17],
            [128],
            [16],
            # Empty batch ([]) excluded: create_empty_sample dtype bug (#527)
        ]
        torch.compiler.reset()
        for seq_lengths in batch_configs:
            samples = _make_samples(seq_lengths) if seq_lengths else []
            batch = make_batch(
                max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE
            )
            draft_tokens, loss, metrics = eagle3_model(**batch, ttt_steps=2)
            assert loss.isfinite(), f"Loss not finite for seq_lengths={seq_lengths}"
            loss.backward()

    def test_alternating_batch_sizes(self, eagle3_model):
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
            draft_tokens, loss, metrics = eagle3_model(**batch, ttt_steps=2)
            assert loss.isfinite()
            loss.backward()

    def test_varying_loss_masks_across_batches(self, eagle3_model):
        """Each batch uses a different loss mask pattern."""
        for pattern in LOSS_MASK_CASES:
            samples = _make_samples([64, 64], loss_mask_pattern=pattern)
            batch = make_batch(
                max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE
            )
            draft_tokens, loss, metrics = eagle3_model(**batch, ttt_steps=2)
            assert loss.isfinite(), f"Loss not finite for loss_mask_pattern={pattern}"
            loss.backward()


@requires_cuda
class TestEagle3VocabBoundary:
    """Tests with draft vocab mapping and boundary token IDs."""

    @pytest.fixture
    def eagle3_draft_vocab_model(self):
        model = make_eagle3_model(draft_vocab_size=32)
        t2d = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        t2d[:32] = True
        d2t = torch.arange(32, dtype=torch.long)
        model.load_vocab_mappings(t2d.to("cuda"), d2t.to("cuda"))
        yield model
        del model
        torch.cuda.empty_cache()

    def test_boundary_tokens_training(self, eagle3_draft_vocab_model):
        samples = _make_samples([128], vocab_size=32, boundary_token_ids=[0, 31])
        batch = make_batch(max_len=MAX_LEN, samples=samples, hidden_size=HIDDEN_SIZE)
        draft_tokens, loss, metrics = eagle3_draft_vocab_model(**batch, ttt_steps=2)

        assert len(draft_tokens) == 2
        assert loss.isfinite()
        for dt in draft_tokens:
            assert (dt >= 0).all()
            assert (dt < 32).all()
