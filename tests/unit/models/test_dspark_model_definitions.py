"""Unit tests for DSpark Markov and confidence heads."""

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.proposals.greedy import GreedyTokenProposalConfig


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


def test_markov_head_is_covered_by_the_weight_init_sweep():
    # The heads are attached after DFlash's __init__ has run the init sweep, so
    # without a second pass the Markov embedding keeps nn.Embedding's N(0, 1)
    # default -- ~50x the initializer_range every other draft matrix gets, landing
    # as a large random bias directly on the draft logits.
    transformer_config = Qwen3Config(
        vocab_size=2048,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    model = DSparkDraftModel(
        DSparkSpeculatorConfig(
            transformer_layer_config=transformer_config,
            draft_vocab_size=512,
            block_size=4,
            aux_hidden_state_layer_ids=[0, 1, 2],
            mask_token_id=0,
            markov_rank=16,
            markov_head_type="vanilla",
            speculators_config=SpeculatorsConfig(
                algorithm="dspark",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["Qwen3ForCausalLM"]
                ),
            ),
        )
    )

    # `fc` is a matrix the first sweep certainly covered; compare against it rather
    # than hardcoding initializer_range.
    reference = model.fc.weight.std().item()
    assert model.markov_head.markov_w1.weight.std().item() == pytest.approx(
        reference, rel=0.3
    )
