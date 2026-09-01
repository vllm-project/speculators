"""Unit tests for DSpark Markov and confidence heads."""

from types import SimpleNamespace

import pytest
import torch

from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.train.optimizers import (
    build_optimizers,
    split_named_params_for_muon,
)


class TestMarkovHead:
    def _head(self, head_type="vanilla", r=8, vv=50, dv=20, h=16):
        torch.manual_seed(0)
        return MarkovHead(
            verifier_vocab_size=vv,
            draft_vocab_size=dv,
            markov_rank=r,
            hidden_size=h,
            head_type=head_type,
        )

    @pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
    def test_block_bias_shape(self, head_type):
        head = self._head(head_type)
        n, b, h = 3, 4, 16
        prev = torch.randint(0, 50, (n, b))
        hidden = torch.randn(n, b, h)
        bias = head.block_bias(prev_token_ids=prev, hidden_states=hidden)
        assert bias.shape == (n, b, 20)
        assert torch.isfinite(bias).all()

    def test_vanilla_is_low_rank_factorization(self):
        head = self._head("vanilla")
        prev = torch.randint(0, 50, (2, 4))
        hidden = torch.zeros(2, 4, 16)
        bias = head.block_bias(prev_token_ids=prev, hidden_states=hidden)
        expected = head.markov_w2(head.markov_w1(prev))
        assert torch.allclose(bias, expected, atol=1e-5)

    def test_bias_depends_on_prev_token(self):
        head = self._head("vanilla")
        hidden = torch.zeros(1, 1, 16)
        bias_a = head.block_bias(
            prev_token_ids=torch.tensor([[1]]), hidden_states=hidden
        )
        bias_b = head.block_bias(
            prev_token_ids=torch.tensor([[2]]), hidden_states=hidden
        )
        assert not torch.allclose(bias_a, bias_b)

    def test_lookup_embedding_has_small_initialization(self):
        head = self._head("vanilla")

        assert head.markov_w1.weight.std().item() == pytest.approx(0.01, rel=0.2)

    def test_vocab_factors_use_adamw_under_muon_optimizer(self):
        head = self._head("gated")

        muon, adamw, excluded = split_named_params_for_muon(head)

        assert {name for name, _ in muon} == {"gate_proj.weight"}
        assert {name for name, _ in adamw} == {"gate_proj.bias"}
        assert {name for name, _ in excluded} == {
            "markov_w1.weight",
            "markov_w2.weight",
        }

    def test_vocab_factors_keep_muon_lr_not_the_base_lr(self):
        """The Markov factors are skipped by Muon because a vocabulary index is not a
        feature axis -- not because they want a 10x smaller step than their neighbours.
        """
        head = self._head("vanilla")
        config = SimpleNamespace(
            optimizer="muon",
            lr=3e-4,
            weight_decay=0.01,
            muon_lr=3e-3,
            muon_momentum=0.95,
            muon_weight_decay=0.1,
            muon_ns_steps=5,
            muon_adjust_lr_fn="match_rms_adamw",
        )

        groups = {
            name: group
            for opt in build_optimizers(head, config)
            for group in opt.param_groups
            for name in group.get("param_names") or []
        }

        for name in ("markov_w1.weight", "markov_w2.weight"):
            assert groups[name]["lr"] == config.muon_lr
            assert groups[name]["weight_decay"] == config.muon_weight_decay

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError):
            MarkovHead(
                verifier_vocab_size=50,
                draft_vocab_size=20,
                markov_rank=0,
                hidden_size=16,
            )

    def test_invalid_head_type_raises(self):
        with pytest.raises(ValueError):
            MarkovHead(
                verifier_vocab_size=50,
                draft_vocab_size=20,
                markov_rank=8,
                hidden_size=16,
                head_type="bogus",
            )


class TestConfidenceHead:
    def test_output_shape(self):
        head = ConfidenceHead(input_dim=24)
        features = torch.randn(3, 4, 24)
        out = head(features)
        assert out.shape == (3, 4)
        assert torch.isfinite(out).all()
