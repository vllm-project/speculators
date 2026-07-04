import copy

import pytest
import torch
from torch import nn

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.domino import DominoHead
from speculators.proposals.greedy import GreedyTokenProposalConfig
from tests.conftest import requires_cuda
from tests.integration.conftest import TINY_QWEN3_CONFIG, _fill_nan_weights


class TestDominoHeadInit:
    def test_creates_submodules(self):
        head = DominoHead(
            hidden_size=64, gru_hidden_dim=128, emb_dim=32, draft_vocab_size=100
        )
        assert isinstance(head.prefix_gru, nn.GRU)
        assert head.prefix_gru.input_size == 64
        assert head.prefix_gru.hidden_size == 128
        assert head.prefix_gru.num_layers == 1
        assert head.prefix_gru.batch_first is True
        assert head.prefix_gru.bias is False

        assert len(head.embed_proj) == 3
        assert isinstance(head.embed_proj[0], nn.Linear)
        assert head.embed_proj[0].in_features == 64 + 128
        assert head.embed_proj[0].out_features == 32
        assert head.embed_proj[0].bias is None
        assert isinstance(head.embed_proj[1], nn.SiLU)
        assert isinstance(head.embed_proj[2], nn.Linear)
        assert head.embed_proj[2].in_features == 32
        assert head.embed_proj[2].out_features == 100


class TestDominoHeadForward:
    HIDDEN_SIZE = 64
    GRU_DIM = 128
    EMB_DIM = 32
    VOCAB_SIZE = 100
    NUM_ANCHORS = 4
    BLOCK_SIZE = 8

    @pytest.fixture
    def head(self):
        return DominoHead(
            hidden_size=self.HIDDEN_SIZE,
            gru_hidden_dim=self.GRU_DIM,
            emb_dim=self.EMB_DIM,
            draft_vocab_size=self.VOCAB_SIZE,
        )

    @pytest.fixture
    def embed_tokens(self):
        emb = nn.Embedding(self.VOCAB_SIZE, self.HIDDEN_SIZE)
        nn.init.normal_(emb.weight, std=0.02)
        return emb

    @pytest.fixture
    def inputs(self, embed_tokens):
        hidden = torch.randn(1, self.NUM_ANCHORS, self.BLOCK_SIZE, self.HIDDEN_SIZE)
        logits = torch.randn(1, self.NUM_ANCHORS, self.BLOCK_SIZE, self.VOCAB_SIZE)
        prev_ids = torch.randint(
            0, self.VOCAB_SIZE, (1, self.NUM_ANCHORS, self.BLOCK_SIZE)
        )
        return hidden, logits, prev_ids, embed_tokens

    def test_output_shape(self, head, inputs):
        hidden, logits, prev_ids, embed_tokens = inputs
        out = head(hidden, logits, prev_ids, suffix_start=1, embed_tokens=embed_tokens)
        assert out.shape == (1, self.NUM_ANCHORS, self.BLOCK_SIZE, self.VOCAB_SIZE)

    def test_suffix_positions_differ_from_base(self, head, inputs):
        hidden, logits, prev_ids, embed_tokens = inputs
        suffix_start = 2
        out = head(
            hidden,
            logits,
            prev_ids,
            suffix_start=suffix_start,
            embed_tokens=embed_tokens,
        )
        assert torch.equal(out[:, :, :suffix_start], logits[:, :, :suffix_start])
        assert not torch.equal(out[:, :, suffix_start:], logits[:, :, suffix_start:])

    def test_additive_deltas(self, head, inputs):
        hidden, logits, prev_ids, embed_tokens = inputs
        suffix_start = 1
        out = head(
            hidden,
            logits,
            prev_ids,
            suffix_start=suffix_start,
            embed_tokens=embed_tokens,
        )
        delta = out[:, :, suffix_start:] - logits[:, :, suffix_start:]
        assert delta.abs().sum() > 0

    def test_anchor_position_uncorrected_when_suffix_start_gt_zero(self, head, inputs):
        hidden, logits, prev_ids, embed_tokens = inputs
        out = head(hidden, logits, prev_ids, suffix_start=1, embed_tokens=embed_tokens)
        assert torch.equal(out[:, :, 0:1], logits[:, :, 0:1])
        assert not torch.equal(out[:, :, 1:], logits[:, :, 1:])

    def test_suffix_start_zero_corrects_all_positions(self, head, inputs):
        hidden, logits, prev_ids, embed_tokens = inputs
        out = head(hidden, logits, prev_ids, suffix_start=0, embed_tokens=embed_tokens)
        assert not torch.equal(out[:, :, 0:1], logits[:, :, 0:1])
        assert not torch.equal(out[:, :, 1:], logits[:, :, 1:])

    def test_suffix_start_one_corrects_positions_1_through_end(self, head, inputs):
        hidden, logits, prev_ids, embed_tokens = inputs
        out = head(hidden, logits, prev_ids, suffix_start=1, embed_tokens=embed_tokens)
        assert torch.equal(out[:, :, :1], logits[:, :, :1])
        assert not torch.equal(out[:, :, 1:], logits[:, :, 1:])

    def test_suffix_start_equals_block_size_no_correction(self, head, inputs):
        hidden, logits, prev_ids, embed_tokens = inputs
        out = head(
            hidden,
            logits,
            prev_ids,
            suffix_start=self.BLOCK_SIZE,
            embed_tokens=embed_tokens,
        )
        assert torch.equal(out, logits)

    def test_gru_accepts_correct_shapes(self, head, inputs, embed_tokens):
        hidden, logits, prev_ids, _ = inputs
        prev_embeds = embed_tokens(prev_ids)
        flat_batch = 1 * self.NUM_ANCHORS
        prev_2d = prev_embeds.view(flat_batch, self.BLOCK_SIZE, self.HIDDEN_SIZE)
        gru_out, h_n = head.prefix_gru(prev_2d)
        assert gru_out.shape == (flat_batch, self.BLOCK_SIZE, self.GRU_DIM)
        assert h_n.shape == (1, flat_batch, self.GRU_DIM)

    def test_backpropagation(self, head, inputs):
        hidden, logits, prev_ids, embed_tokens = inputs
        out = head(hidden, logits, prev_ids, suffix_start=2, embed_tokens=embed_tokens)
        loss = out.sum()
        loss.backward()
        assert head.prefix_gru.weight_ih_l0.grad is not None
        assert head.prefix_gru.weight_hh_l0.grad is not None
        assert head.embed_proj[0].weight.grad is not None
        assert head.embed_proj[2].weight.grad is not None


class TestDominoHeadShiftLabel:
    HIDDEN_SIZE = 64
    GRU_DIM = 128
    EMB_DIM = 32
    VOCAB_SIZE = 100
    NUM_ANCHORS = 4
    BLOCK_SIZE = 8

    @pytest.fixture
    def head(self):
        return DominoHead(
            hidden_size=self.HIDDEN_SIZE,
            gru_hidden_dim=self.GRU_DIM,
            emb_dim=self.EMB_DIM,
            draft_vocab_size=self.VOCAB_SIZE,
        )

    @pytest.fixture
    def embed_tokens(self):
        emb = nn.Embedding(self.VOCAB_SIZE, self.HIDDEN_SIZE)
        nn.init.normal_(emb.weight, std=0.02)
        return emb

    @pytest.fixture
    def inputs(self, embed_tokens):
        hidden = torch.randn(1, self.NUM_ANCHORS, self.BLOCK_SIZE, self.HIDDEN_SIZE)
        logits = torch.randn(1, self.NUM_ANCHORS, self.BLOCK_SIZE, self.VOCAB_SIZE)
        prev_ids = torch.randint(
            0, self.VOCAB_SIZE, (1, self.NUM_ANCHORS, self.BLOCK_SIZE)
        )
        return hidden, logits, prev_ids, embed_tokens

    # suffix_start = pure_draft_prefix_len
    # if not shift_label: suffix_start = 1 + pure_draft_prefix_len
    # (see core.py DFlashDraftModel._backbone_forward)
    @pytest.mark.parametrize(
        ("pure_draft_prefix_len", "shift_label"),
        [
            (2, True),
            (2, False),
            (0, True),
            (0, False),
        ],
    )
    def test_suffix_start_applied_via_domino_head(
        self, head, inputs, pure_draft_prefix_len, shift_label
    ):
        hidden, logits, prev_ids, embed_tokens = inputs
        suffix_start = (
            pure_draft_prefix_len if shift_label else 1 + pure_draft_prefix_len
        )
        out = head(
            hidden,
            logits,
            prev_ids,
            suffix_start=suffix_start,
            embed_tokens=embed_tokens,
        )
        assert torch.equal(out[:, :, :suffix_start], logits[:, :, :suffix_start])
        if suffix_start < self.BLOCK_SIZE:
            assert not torch.equal(
                out[:, :, suffix_start:], logits[:, :, suffix_start:]
            )


# ---------------------------------------------------------------------------
# Helpers for core.py integration tests
# ---------------------------------------------------------------------------


def _make_tiny_domino_model(
    block_size: int = 4,
    shift_label: bool = True,
    lambda_base_start: float = 1.0,
    lambda_base_decay_steps: int = 0,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> DFlashDraftModel:
    transformer_config = copy.deepcopy(TINY_QWEN3_CONFIG)
    config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=64,
        block_size=block_size,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        projector_type="domino",
        shift_label=shift_label,
        lambda_base_start=lambda_base_start,
        lambda_base_decay_steps=lambda_base_decay_steps,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )
    model = DFlashDraftModel(config)
    _fill_nan_weights(model)
    return model.to(device=device, dtype=dtype) if device else model  # type: ignore[call-arg]


def _make_synthetic_data(
    seq_len: int = 32,
    hidden_size: int = 64,
    num_target_layers: int = 3,
    loss_mask_all: bool = True,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    loss_mask = torch.ones(1, seq_len, dtype=dtype, device=device)
    if not loss_mask_all:
        loss_mask[:, -8:] = 0
    return {
        "hidden_states": torch.randn(
            1, seq_len, num_target_layers * hidden_size, dtype=dtype, device=device
        ),
        "input_ids": torch.randint(0, 64, (1, seq_len), device=device),
        "loss_mask": loss_mask,
        "verifier_last_hidden_states": torch.randn(
            1, seq_len, hidden_size, dtype=dtype, device=device
        ),
        "document_ids": torch.ones(1, seq_len, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Core model target alignment tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestBackboneTargetAlignment:
    def test_shift_targets_true_shifts_by_one(self):
        torch.manual_seed(42)
        model = _make_tiny_domino_model(block_size=4)
        data = _make_synthetic_data(seq_len=32, loss_mask_all=False)
        loss_mask = data["loss_mask"]

        _, _, shifted_targets, _, shifted_anchored_idx = model._backbone_forward(
            data["hidden_states"],
            data["input_ids"],
            loss_mask,
            data["verifier_last_hidden_states"],
            data["document_ids"],
            shift_targets=True,
            max_anchors=8,
        )

        bs = model.block_size
        anchor_indices = shifted_anchored_idx[::bs]
        pos0_shifted = shifted_targets[:, ::bs, :]

        raw_logits = model.verifier_lm_head(
            model.verifier_norm(data["verifier_last_hidden_states"])
        )
        rolled_logits = torch.roll(raw_logits, 1, dims=1)
        expected_shifted = rolled_logits[:, anchor_indices + 1, :]

        assert torch.equal(pos0_shifted, expected_shifted), (
            "With shift_targets=True, position 0 targets must equal "
            "verifier_logits at anchor+1"
        )

    def test_shift_targets_false_no_shift(self):
        torch.manual_seed(42)
        model = _make_tiny_domino_model(block_size=4)
        data = _make_synthetic_data(seq_len=32, loss_mask_all=False)
        loss_mask = data["loss_mask"]

        _, _, unshifted_targets, _, unshifted_anchored_idx = model._backbone_forward(
            data["hidden_states"],
            data["input_ids"],
            loss_mask,
            data["verifier_last_hidden_states"],
            data["document_ids"],
            shift_targets=False,
            max_anchors=8,
        )

        bs = model.block_size
        anchor_indices = unshifted_anchored_idx[::bs]
        pos0_unshifted = unshifted_targets[:, ::bs, :]

        raw_logits = model.verifier_lm_head(
            model.verifier_norm(data["verifier_last_hidden_states"])
        )
        rolled_logits = torch.roll(raw_logits, 1, dims=1)
        expected = rolled_logits[:, anchor_indices, :]

        assert torch.equal(pos0_unshifted, expected), (
            "With shift_targets=False, position 0 targets must equal "
            "verifier_logits at anchor positions"
        )

    def test_shift_targets_true_oob_clamped_and_masked(self):
        torch.manual_seed(42)
        block_size = 8
        seq_len = 32
        model = _make_tiny_domino_model(block_size=block_size)
        loss_mask = torch.ones(1, seq_len, dtype=torch.bfloat16, device="cuda:0")
        loss_mask[:, -(block_size - 1) :] = 0  # force anchors near the end
        data = _make_synthetic_data(seq_len=seq_len, loss_mask_all=False)
        data["loss_mask"] = loss_mask

        _, _, _, aligned_mask, anchored_idx = model._backbone_forward(
            data["hidden_states"],
            data["input_ids"],
            loss_mask,
            data["verifier_last_hidden_states"],
            data["document_ids"],
            shift_targets=True,
            max_anchors=4,
        )

        # Positions where anchored_block_indices + 1 >= verifier_logits.shape[1]
        # must have aligned_loss_mask == 0
        max_idx = (
            seq_len - 1
        )  # verifier_logits shape is [1, seq_len, V], indexed 0..seq_len-1
        shifted = anchored_idx + 1
        oob = shifted >= max_idx
        if oob.any():
            assert (aligned_mask[:, oob] == 0).all(), (
                "OOB positions must have loss mask zeroed"
            )


# ---------------------------------------------------------------------------
# Domino loss mask tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestDominoLossMask:
    def test_position_zero_in_shift_label_true(self):
        torch.manual_seed(42)
        model = _make_tiny_domino_model(block_size=4, shift_label=True)
        data = _make_synthetic_data(seq_len=32, loss_mask_all=False)
        loss_mask = data["loss_mask"]

        _, _, _, aligned_mask, anchored_idx = model._backbone_forward(
            data["hidden_states"],
            data["input_ids"],
            loss_mask,
            data["verifier_last_hidden_states"],
            data["document_ids"],
            shift_targets=True,
            max_anchors=8,
        )

        bs = model.block_size

        # aligned_loss_mask has position 0 zeroed
        assert (aligned_mask[:, ::bs] == 0).all(), (
            "aligned_loss_mask must zero position 0 of every block"
        )

        # Construct domino_loss_mask the same way forward() does
        domino_mask = aligned_mask.clone()
        anchor_pos = anchored_idx[::bs]
        domino_mask[:, ::bs] = loss_mask[:, anchor_pos]

        # With shift_label=True and loss_mask=1 at anchors, position 0 must be unmasked
        assert (domino_mask[:, ::bs] != 0).any(), (
            "domino_loss_mask must include position 0 when shift_label=True"
        )

    def test_position_zero_excluded_when_shift_label_false(self):
        torch.manual_seed(42)
        model = _make_tiny_domino_model(block_size=4, shift_label=False)
        data = _make_synthetic_data(seq_len=32, loss_mask_all=False)
        loss_mask = data["loss_mask"]

        _, _, _, aligned_mask, anchored_idx = model._backbone_forward(
            data["hidden_states"],
            data["input_ids"],
            loss_mask,
            data["verifier_last_hidden_states"],
            data["document_ids"],
            shift_targets=True,
            max_anchors=8,
        )

        bs = model.block_size

        # When shift_label=False, domino_loss_mask = aligned_loss_mask
        # Position 0 must still be zero
        assert (aligned_mask[:, ::bs] == 0).all(), (
            "With shift_label=False, domino_loss_mask must still zero position 0"
        )


# ---------------------------------------------------------------------------
# Lambda base decay tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestLambdaBaseDecay:
    def test_decay_steps_zero_no_decay(self):
        torch.compiler.reset()
        model = _make_tiny_domino_model(
            block_size=4,
            lambda_base_start=1.0,
            lambda_base_decay_steps=0,
        )
        data = _make_synthetic_data(seq_len=32, loss_mask_all=False)

        loss_mask = data["loss_mask"]

        # Call with a large global_step — lambda should stay at 1.0
        _, _, metrics = model(
            data["hidden_states"],
            data["input_ids"],
            loss_mask,
            data["verifier_last_hidden_states"],
            data["document_ids"],
            global_step=1000,
            max_anchors=8,
        )

        # loss = (1-lambda) * final + lambda * base
        # With lambda=1.0: loss = base_loss
        loss_sum = metrics["loss_sum"]
        base_sum = metrics["base_loss_sum"]
        assert torch.allclose(loss_sum, base_sum, rtol=1e-4), (
            "With decay_steps=0, lambda must stay at 1.0 (loss == base_loss)"
        )

    def test_decay_steps_positive_decays(self):
        torch.compiler.reset()
        decay_steps = 100
        model = _make_tiny_domino_model(
            block_size=4,
            lambda_base_start=1.0,
            lambda_base_decay_steps=decay_steps,
        )
        data = _make_synthetic_data(seq_len=32, loss_mask_all=False)
        loss_mask = data["loss_mask"]

        # Call with global_step = decay_steps (progress=1.0, lambda=0.0)
        _, _, metrics_full_decay = model(
            data["hidden_states"],
            data["input_ids"],
            loss_mask,
            data["verifier_last_hidden_states"],
            data["document_ids"],
            global_step=decay_steps,
            max_anchors=8,
        )

        # With lambda=0: loss = final_loss
        assert torch.allclose(
            metrics_full_decay["loss_sum"],
            metrics_full_decay["final_loss_sum"],
            rtol=1e-4,
        ), "At progress=1.0, lambda must be 0.0 (loss == final_loss)"

    def test_decay_midpoint_loss_is_blend(self):
        torch.compiler.reset()
        decay_steps = 100
        model = _make_tiny_domino_model(
            block_size=4,
            lambda_base_start=1.0,
            lambda_base_decay_steps=decay_steps,
        )
        data = _make_synthetic_data(seq_len=32, loss_mask_all=False)
        loss_mask = data["loss_mask"]

        # Call with global_step = decay_steps // 2 (progress=0.5, lambda=0.5)
        _, _, metrics_mid = model(
            data["hidden_states"],
            data["input_ids"],
            loss_mask,
            data["verifier_last_hidden_states"],
            data["document_ids"],
            global_step=decay_steps // 2,
            max_anchors=8,
        )

        ls = metrics_mid["loss_sum"]
        fls = metrics_mid["final_loss_sum"]
        bls = metrics_mid["base_loss_sum"]

        lo = torch.min(fls, bls)
        hi = torch.max(fls, bls)
        # With lambda=0.5, loss must be between final_loss and base_loss
        assert lo < ls < hi or hi < ls < lo, (
            f"loss ({ls.item():.6f}) must be between "
            f"final ({fls.item():.6f}) and base ({bls.item():.6f})"
        )
        # Loss at midpoint must not equal either extreme
        assert ls != fls, "Loss must not equal pure final_loss at midpoint"
        assert ls != bls, "Loss must not equal pure base_loss at midpoint"
