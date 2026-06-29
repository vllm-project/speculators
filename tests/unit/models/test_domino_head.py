import pytest
import torch
from torch import nn

from speculators.models.dflash.domino import DominoHead


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
    def test_shift_label_true_suffix_start_equals_prefix_len(self):
        pure_draft_prefix_len = 2
        suffix_start = pure_draft_prefix_len
        assert suffix_start == 2

    def test_shift_label_false_suffix_start_is_prefix_len_plus_one(self):
        pure_draft_prefix_len = 2
        suffix_start = 1 + pure_draft_prefix_len
        assert suffix_start == 3

    def test_suffix_start_zero_when_prefix_len_zero_and_shift_label_true(self):
        pure_draft_prefix_len = 0
        suffix_start = pure_draft_prefix_len
        assert suffix_start == 0

    def test_suffix_start_one_when_prefix_len_zero_and_shift_label_false(self):
        pure_draft_prefix_len = 0
        suffix_start = 1 + pure_draft_prefix_len
        assert suffix_start == 1
